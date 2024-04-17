pub(crate) mod data_transformer;
pub mod input;

use crate::tvae::data_transformer::{ColumnTrainInfo, DataTransformer};
use crate::tvae::input::{ColumnDataRef, SampledColumnData};
use crate::utils::{self, Pdf};
use base64::Engine;
use std::io::Cursor;
use tch::nn::{Adam, Module, OptimizerConfig};
use tch::{nn, Device, Kind, Tensor};

const JSON_DATA_TRANSFORMER_FIELD: &str = "data_transformer";
const JSON_BATCH_SIZE_FIELD: &str = "batch_size";
const JSON_NN_DATA_FIELD: &str = "nn_data";

const LOSS_FACTOR: f32 = 2.0;
const L2_SCALE: f64 = 0.0; //1e-5;
const EMBEDDING_DIM: i64 = 128;
const COMPRESS_DIMS: [i64; 2] = [128, 128];
const DECOMPRESS_DIMS: [i64; 2] = [128, 128];
const NUM_SAMPLE_CYCLES: usize = 8;

struct Encoder {
    seq: nn::Sequential,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl Encoder {
    pub fn new(vs: &nn::Path, data_dim: i64, compress_dims: &[i64], embedding_dim: i64) -> Self {
        let vs = vs / "encoder";
        let mut seq = nn::seq();
        let mut curr_dim = data_dim;

        for (i, dim) in compress_dims.iter().enumerate() {
            seq = seq
                .add(nn::linear(
                    &vs / format!("seq{}", i),
                    curr_dim,
                    *dim,
                    Default::default(),
                ))
                .add_fn(Tensor::mish);
            curr_dim = *dim;
        }

        let fc1 = nn::linear(&vs / "fc1", curr_dim, embedding_dim, Default::default());
        let fc2 = nn::linear(&vs / "fc2", curr_dim, embedding_dim, Default::default());

        Self { seq, fc1, fc2 }
    }

    pub fn encode(&self, input: &Tensor) -> (Tensor, Tensor, Tensor) {
        let feature = self.seq.forward(input);
        let mu = self.fc1.forward(&feature);
        let log_var = self.fc2.forward(&feature);
        let std = (0.5_f32 * &log_var).exp();

        (mu, std, log_var)
    }
}

struct Decoder {
    seq: nn::Sequential,
}

impl Decoder {
    pub fn new(vs: &nn::Path, embedding_dim: i64, decompress_dims: &[i64], data_dim: i64) -> Self {
        let vs = vs / "decoder";
        let mut seq = nn::seq();
        let mut curr_dim = embedding_dim;

        for (i, dim) in decompress_dims.iter().enumerate() {
            seq = seq
                .add(nn::linear(
                    &vs / format!("seq{}", i),
                    curr_dim,
                    *dim,
                    Default::default(),
                ))
                .add_fn(Tensor::mish);
            curr_dim = *dim;
        }

        seq = seq.add(nn::linear(
            &vs / "seqLast",
            curr_dim,
            data_dim,
            Default::default(),
        ));

        Self { seq }
    }

    pub fn decode(&self, input: &Tensor) -> Tensor {
        self.seq.forward(input)
    }
}

fn calc_loss(
    recon_x: &Tensor,
    x: &Tensor,
    mu: &Tensor,
    log_var: &Tensor,
    output_info: &[ColumnTrainInfo],
) -> (Tensor, Tensor) {
    let mut start = 0_i64;
    let mut loss = vec![];

    for column_info in output_info {
        for span_info in column_info.output_spans() {
            let end = start + span_info.dim;

            let x_slice = x.slice(1, start, end, 1);
            let recon_x_slice = recon_x.slice(1, start, end, 1);

            let cross_loss = recon_x_slice.cross_entropy_loss::<&_>(
                &x_slice.argmax(-1, false),
                Some(&column_info.balance_weights().to(x_slice.device())),
                tch::Reduction::Sum,
                -100,
                0.0,
            );
            loss.push(cross_loss);

            start = end;
        }
    }

    assert_eq!(start, recon_x.size()[1]);

    let kld = -0.5_f32
        * (1_f32 + log_var - mu.pow_tensor_scalar(2) - log_var.exp()).sum(tch::Kind::Float);

    let s = loss.iter().sum::<Tensor>() / x.size()[0];
    let kld_norm = &kld / x.size()[0];

    (s, kld_norm)
}

pub struct TVAE {
    vs: nn::VarStore,
    device: Device,
    decoder: Decoder,
    batch_size: usize,
    transformer: DataTransformer,
}

pub type DoStop = bool;
pub type Realness = f32;
pub type CorrelationRealness = f32;

impl TVAE {
    /// `flow_control`: Fn(epoch, loss) -> DoStop
    pub fn fit<F: Fn(usize, f64) -> DoStop>(
        data: &[ColumnDataRef],
        batch_size: usize,
        device: Device,
        flow_control: F,
    ) -> Self {
        let vs = nn::VarStore::new(device);
        assert!(!data.is_empty());

        let transformer = DataTransformer::prepare(data);
        let n_rows = data[0].len() as i64;

        let n_tiles = batch_size.div_ceil(n_rows as usize).max(2);
        let aligned_n_rows = (n_rows as usize).next_multiple_of(batch_size) as i64;

        let train_data = Tensor::cat(
            &data
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let transformed = transformer.transform(i, v);

                    transformed
                        .repeat([n_tiles as i64, 1])
                        .slice(0, 0, aligned_n_rows, 1)
                })
                .collect::<Vec<_>>(),
            1,
        )
        .totype(tch::Kind::Float)
        .to(device);

        let data_dim = transformer
            .train_infos()
            .iter()
            .map(|v| v.total_dim())
            .sum::<i64>();

        let encoder = Encoder::new(&vs.root(), data_dim, &COMPRESS_DIMS, EMBEDDING_DIM);
        let decoder = Decoder::new(&vs.root(), EMBEDDING_DIM, &DECOMPRESS_DIMS, data_dim);

        let mut optimizer = Adam::default().wd(L2_SCALE).build(&vs, 7e-4).unwrap();
        let mut epoch = 0;

        loop {
            let shuffle_perm =
                Tensor::randperm(aligned_n_rows, (tch::Kind::Int64, tch::Device::Cpu));
            let curr_train_data = train_data.index(&[Some(shuffle_perm)]).to(device);

            let mut total_loss = 0.0;
            let mut loss_count = 0.0;

            for batch_start in (0..aligned_n_rows).step_by(batch_size) {
                let batch_real =
                    curr_train_data.slice(0, batch_start, batch_start + batch_size as i64, 1);

                optimizer.zero_grad();

                let (mu, std, log_var) = encoder.encode(&batch_real);

                let random_deviations = std.randn_like();
                let latents = &random_deviations * &std + &mu;
                let batch_reconstructed = decoder.decode(&latents);

                let (recon_loss, kl_loss) = calc_loss(
                    &batch_reconstructed,
                    &batch_real,
                    &mu,
                    &log_var,
                    &transformer.train_infos(),
                );

                let loss: Tensor = LOSS_FACTOR * &recon_loss + &kl_loss;

                loss.backward();
                optimizer.step();

                total_loss += loss.double_value(&[]);
                loss_count += 1.0;
            }

            let loss = total_loss / loss_count;

            if flow_control(epoch, loss) {
                break;
            }
            epoch += 1;
        }

        Self {
            vs,
            decoder,
            batch_size,
            device,
            transformer,
        }
    }

    pub fn save(&self) -> serde_json::Value {
        let transformer_data = self.transformer.save();
        let batch_size = serde_json::Value::Number(self.batch_size.into());

        let mut nn_data_cursor = Cursor::new(Vec::with_capacity(1024 * 1024));
        self.vs.save_to_stream(&mut nn_data_cursor).unwrap();

        let nn_data_base64 = serde_json::Value::String(
            base64::engine::general_purpose::STANDARD_NO_PAD.encode(nn_data_cursor.get_ref()),
        );

        let mut map = serde_json::Map::new();
        map.insert(JSON_DATA_TRANSFORMER_FIELD.to_owned(), transformer_data);
        map.insert(JSON_BATCH_SIZE_FIELD.to_owned(), batch_size);
        map.insert(JSON_NN_DATA_FIELD.to_owned(), nn_data_base64);

        serde_json::Value::Object(map)
    }

    pub fn load(device: Device, data: serde_json::Value) -> Self {
        let data = data.as_object().unwrap();
        let transformer_data = data.get(JSON_DATA_TRANSFORMER_FIELD).unwrap();
        let batch_size = data.get(JSON_BATCH_SIZE_FIELD).unwrap().as_u64().unwrap();
        let nn_data_base64 = data.get(JSON_NN_DATA_FIELD).unwrap().as_str().unwrap();

        let transformer = DataTransformer::load(transformer_data);

        let nn_data = base64::engine::general_purpose::STANDARD_NO_PAD
            .decode(nn_data_base64)
            .unwrap();

        let data_dim = transformer
            .train_infos()
            .iter()
            .map(|v| v.total_dim())
            .sum::<i64>();

        let mut vs = nn::VarStore::new(device);
        let decoder = Decoder::new(&vs.root(), EMBEDDING_DIM, &DECOMPRESS_DIMS, data_dim);
        vs.freeze();

        let mut cursor = Cursor::new(nn_data);
        vs.load_from_stream(&mut cursor).unwrap();

        Self {
            vs,
            device,
            decoder,
            batch_size: batch_size as usize,
            transformer,
        }
    }

    pub fn sample(&self, samples: usize) -> (Vec<SampledColumnData>, CorrelationRealness) {
        let n_columns = self.transformer.train_infos().len();
        let mut generated_columns = Vec::<SampledColumnData>::with_capacity(n_columns);

        let latents = Tensor::randn([samples as i64, EMBEDDING_DIM], (Kind::Float, self.device));
        let generated = self.decoder.decode(&latents);
        let mut generated_indexed = self.transformer.inverse_samples_to_indices(&generated);

        let target_pdfs: Vec<Pdf> = self
            .transformer
            .column_infos()
            .iter()
            .map(|v| v.pdf().clone())
            .collect();
        let mut curr_pdfs: Vec<Pdf> = target_pdfs
            .iter()
            .map(|v| Pdf::new(vec![0; v.buckets().len()]))
            .collect();

        for cycle in 0..NUM_SAMPLE_CYCLES {
            let replacements = self.decoder.decode(&Tensor::randn(
                [samples as i64, EMBEDDING_DIM],
                (Kind::Float, self.device),
            ));
            let new_generated_indexed = generated_indexed.to(Device::Cpu);
            let replacements_indexed = self
                .transformer
                .inverse_samples_to_indices(&replacements)
                .to(Device::Cpu);
            let mut new_pdfs = curr_pdfs.clone();

            for row_idx in 0..samples {
                let mut curr_sample = new_generated_indexed.get(row_idx as i64);
                let replacement = replacements_indexed.get(row_idx as i64);

                if cycle == 0 {
                    for (col_idx, curr) in new_generated_indexed
                        .get(row_idx as i64)
                        .iter::<i64>()
                        .unwrap()
                        .enumerate()
                    {
                        // Fill the initial pdf up
                        new_pdfs[col_idx].add(curr as usize, 1);
                    }
                    continue;
                }

                let total_influence = new_generated_indexed
                    .get(row_idx as i64)
                    .iter::<i64>()
                    .unwrap()
                    .zip(
                        replacements_indexed
                            .get(row_idx as i64)
                            .iter::<i64>()
                            .unwrap(),
                    )
                    .enumerate()
                    .map(|(col_idx, (curr, replacement))| {
                        utils::calc_change_influence(
                            &target_pdfs[col_idx],
                            &new_pdfs[col_idx],
                            replacement as usize,
                            curr as usize,
                        )
                    })
                    .sum::<f32>();

                if total_influence >= 0.0 {
                    for (col_idx, (curr, replacement)) in new_generated_indexed
                        .get(row_idx as i64)
                        .iter::<i64>()
                        .unwrap()
                        .zip(
                            replacements_indexed
                                .get(row_idx as i64)
                                .iter::<i64>()
                                .unwrap(),
                        )
                        .enumerate()
                    {
                        new_pdfs[col_idx].add(replacement as usize, 1);
                        new_pdfs[col_idx].remove(curr as usize, 1).unwrap();
                    }

                    curr_sample.copy_(&replacement);
                }
            }

            let curr_dist = curr_pdfs
                .iter()
                .zip(&target_pdfs)
                .map(|(a, b)| a.hard_distance(b))
                .sum::<f32>()
                / (curr_pdfs.len() as f32);

            let new_dist = new_pdfs
                .iter()
                .zip(&target_pdfs)
                .map(|(a, b)| a.hard_distance(b))
                .sum::<f32>()
                / (new_pdfs.len() as f32);

            if cycle == 0 || new_dist < curr_dist {
                curr_pdfs = new_pdfs;
                generated_indexed = new_generated_indexed;
            }

            println!("Cycle {cycle}, {new_dist}");
        }

        for (i, _) in self.transformer.train_infos().iter().enumerate() {
            let generated_column = generated_indexed
                .slice(1, i as i64, i as i64 + 1, 1)
                .squeeze();

            let inverse_data = self
                .transformer
                .inverse_transform_indexed(i, &generated_column);
            let col_info = &self.transformer.column_infos()[i];
            let realness = 1.0 - col_info.calc_similarity(&inverse_data);
            // println!("out sim{i} {}", 1.0 - realness);

            generated_columns.push(SampledColumnData::from_regular(inverse_data, realness));
        }

        let real_corr_mat = self.transformer.correlation_matrix();
        let generated_corr_mat = utils::calc_correlation_matrix(
            &generated_columns
                .iter()
                .map(|v| v.data_as_ref())
                .collect::<Vec<_>>(),
        );

        let corr_realness = 1.0
            - real_corr_mat
                .iter()
                .zip(&generated_corr_mat)
                .map(|(v1, v2)| {
                    if v1.is_finite() && v2.is_finite() {
                        ((v1 + 1.0) - (v2 + 1.0)).abs().min(1.0)
                    } else {
                        1.0
                    }
                })
                .sum::<f32>()
                / real_corr_mat.len() as f32;

        (generated_columns, corr_realness)
    }
}
