mod data_transformer;
pub mod input;

use crate::tvae::data_transformer::{ColumnTrainInfo, DataTransformer};
use crate::tvae::input::{ColumnDataRef, SampledColumnData};
use crate::utils;
use base64::Engine;
use std::io::Cursor;
use tch::nn::{Adam, Module, OptimizerConfig};
use tch::{nn, Device, Tensor};

const JSON_DATA_TRANSFORMER_FIELD: &str = "data_transformer";
const JSON_BATCH_SIZE_FIELD: &str = "batch_size";
const JSON_NN_DATA_FIELD: &str = "nn_data";

const LOSS_FACTOR: f32 = 2.0;
const L2_SCALE: f64 = 1e-5;
const EMBEDDING_DIM: i64 = 128;
const COMPRESS_DIMS: [i64; 2] = [128, 128];
const DECOMPRESS_DIMS: [i64; 2] = [128, 128];

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
                .add_fn(Tensor::relu);
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
    sigma: Tensor,
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
                .add_fn(Tensor::relu);
            curr_dim = *dim;
        }

        seq = seq.add(nn::linear(
            &vs / "seqLast",
            curr_dim,
            data_dim,
            Default::default(),
        ));

        let sigma = vs.entry("sigma").or_var(&[data_dim], nn::Init::Const(0.1));

        Self { seq, sigma }
    }

    pub fn decode(&self, input: &Tensor) -> (Tensor, Tensor) {
        let out = self.seq.forward(input);
        (out, self.sigma.shallow_clone())
    }
}

fn calc_loss(
    recon_x: &Tensor,
    x: &Tensor,
    sigmas: &Tensor,
    mu: &Tensor,
    log_var: &Tensor,
    output_info: &[ColumnTrainInfo],
    factor: f32,
) -> (Tensor, Tensor) {
    let mut st = 0_i64;
    let mut loss = vec![];

    for column_info in output_info {
        for span_info in column_info.output_spans() {
            let ed = st + span_info.dim;

            if span_info.activation != "softmax" {
                let x_slice = x.slice(1, Some(st), Some(st + 1), 1);
                let recon_x_slice = recon_x.slice(1, Some(st), Some(st + 1), 1);

                let std = sigmas.get(st);
                let eq = x_slice - recon_x_slice.tanh();

                loss.push(
                    (eq.pow_tensor_scalar(2) / 2 / std.pow_tensor_scalar(2)).sum(tch::Kind::Float),
                );
                loss.push(std.log() * x.size()[0]);
            } else {
                let x_slice = x.slice(1, Some(st), Some(ed), 1);
                let recon_x_slice = recon_x.slice(1, Some(st), Some(ed), 1);

                loss.push(recon_x_slice.cross_entropy_loss::<&_>(
                    &x_slice.argmax(Some(-1), false),
                    None,
                    tch::Reduction::Sum,
                    -100,
                    0.0,
                ));
            }

            st = ed;
        }
    }

    assert_eq!(st, recon_x.size()[1]);

    let kld = -0.5_f32
        * (1_i32 + log_var - mu.pow_tensor_scalar(2) - log_var.exp()).sum(tch::Kind::Float);

    let s = loss.iter().sum::<Tensor>() * factor as f64 / x.size()[0];
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

fn next_multiple_of(n: usize, multiple: usize) -> usize {
    assert!(multiple > 0);
    ((n + multiple - 1) / multiple) * multiple
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

        let train_data = Tensor::cat(
            &data
                .iter()
                .enumerate()
                .map(|(i, v)| transformer.transform(i, v))
                .collect::<Vec<_>>(),
            1,
        )
        .totype(tch::Kind::Float);

        let data_dim = transformer
            .train_infos()
            .iter()
            .map(|v| v.total_dim())
            .sum::<i64>();

        let encoder = Encoder::new(&vs.root(), data_dim, &COMPRESS_DIMS, EMBEDDING_DIM);
        let decoder = Decoder::new(&vs.root(), EMBEDDING_DIM, &DECOMPRESS_DIMS, data_dim);

        let mut optimizer = Adam::default().wd(L2_SCALE).build(&vs, 1e-3).unwrap();
        let mut epoch = 0;

        loop {
            let shuffle_perm = Tensor::randperm(n_rows, (tch::Kind::Int64, tch::Device::Cpu));
            let curr_train_data = train_data.index(&[Some(shuffle_perm)]).to(device);

            let mut total_loss = 0.0;
            let mut loss_count = 0.0;

            for batch_start in (0..n_rows).step_by(batch_size) {
                optimizer.zero_grad();

                let batch_real = curr_train_data.slice(
                    0,
                    Some(batch_start),
                    Some((batch_start + batch_size as i64).min(n_rows)),
                    1,
                );

                let (mu, std, log_var) = encoder.encode(&batch_real);

                let random_deviations = std.randn_like();
                let latents = &random_deviations * &std + &mu;
                let (batch_reconstructed, sigmas) = decoder.decode(&latents);

                let (loss1, loss2) = calc_loss(
                    &batch_reconstructed,
                    &batch_real,
                    &sigmas,
                    &mu,
                    &log_var,
                    &transformer.train_infos(),
                    LOSS_FACTOR,
                );
                let loss = &loss1 + &loss2;

                loss.backward();
                optimizer.step();

                total_loss += loss.double_value(&[]);
                loss_count += 1.0;

                let _ = decoder.sigma.data().clamp_(0.01, 1.0);
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

        let mut nn_data = Vec::with_capacity(1024 * 1024);
        self.vs.save_to_stream(&mut nn_data).unwrap();

        let nn_data_base64 = serde_json::Value::String(
            base64::engine::general_purpose::STANDARD_NO_PAD.encode(nn_data),
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
        vs.load_from_stream(Cursor::new(nn_data)).unwrap();

        Self {
            vs,
            device,
            decoder,
            batch_size: batch_size as usize,
            transformer,
        }
    }

    pub fn sample(&self, samples: usize) -> (Vec<SampledColumnData>, CorrelationRealness) {
        let n_steps = next_multiple_of(samples, self.batch_size) / self.batch_size;
        let n_columns = self.transformer.train_infos().len();
        let mut generated_columns = Vec::<SampledColumnData>::with_capacity(n_columns);
        let mut raw_data = Vec::with_capacity(n_steps * self.batch_size);

        for _ in 0..n_steps {
            let mut noise = Tensor::zeros(
                [self.batch_size as i64, EMBEDDING_DIM],
                (tch::Kind::Float, self.device),
            );
            let _ = noise.normal_(0.0, 1.0);

            let (fake, _sigmas) = self.decoder.decode(&noise);
            let fake = fake.tanh();

            raw_data.push(fake.detach().to(Device::Cpu));
        }

        let generated = Tensor::cat(&raw_data, 0);
        let mut start_idx = 0;

        for (i, train_info) in self.transformer.train_infos().iter().enumerate() {
            let end_idx = start_idx + train_info.total_dim();
            let generated_column = generated.slice(0, None, Some(samples as i64), 1).slice(
                1,
                Some(start_idx),
                Some(end_idx),
                1,
            );

            let inverse_data = self.transformer.inverse_transform(i, &generated_column);
            let col_info = &self.transformer.column_infos()[i];
            let realness = 1.0 - col_info.calc_l1_distance(&inverse_data);

            generated_columns.push(SampledColumnData::from_regular(inverse_data, realness));
            start_idx = end_idx;
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
                        (v1.abs() - v2.abs()).abs()
                    } else {
                        1.0
                    }
                })
                .sum::<f32>()
                / real_corr_mat.len() as f32;

        (generated_columns, corr_realness)
    }
}
