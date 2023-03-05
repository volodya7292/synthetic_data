mod data_transform;
pub mod input;

use crate::tvae::data_transform::{ColumnTrainInfo, DataTransformer};
use crate::tvae::input::ColumnData;
use tch::nn::{Adam, Module, OptimizerConfig};
use tch::{nn, Tensor};

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
            let ed = st + span_info.dim as i64;

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
    device: tch::Device,
    decoder: Decoder,
    batch_size: usize,
    embedding_dim: i64,
    transformer: DataTransformer,
}

fn next_multiple_of(n: usize, multiple: usize) -> usize {
    assert!(multiple > 0);
    ((n + multiple - 1) / multiple) * multiple
}

impl TVAE {
    pub fn fit(data: &[ColumnData], epochs: usize, batch_size: usize, device: tch::Device) -> Self {
        let vs = nn::VarStore::new(device);
        assert!(data.len() > 0);

        let loss_factor = 2.0_f32;
        let l2scale = 1e-5;
        let embedding_dim = 128;
        let compress_dims = [128, 128];
        let decompress_dims = [128, 128];

        let transformer = DataTransformer::prepare(data);
        let n_rows = transformer.n_rows() as i64;

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

        let encoder = Encoder::new(&vs.root(), data_dim, &compress_dims, embedding_dim);
        let decoder = Decoder::new(&vs.root(), embedding_dim, &decompress_dims, data_dim);

        let mut optimizer = Adam::default().wd(l2scale).build(&vs, 1e-3).unwrap();

        for i in 0..epochs {
            print!("Epoch #{}... ", i + 1);

            let shuffle_perm = Tensor::randperm(n_rows, (tch::Kind::Int64, device));
            let curr_train_data = train_data.index(&[Some(shuffle_perm)]);

            let mut total_loss = 0.0;
            let mut loss_count = 0.0;

            for batch_start in (0..n_rows).step_by(batch_size) {
                optimizer.zero_grad();

                let batch = curr_train_data.slice(
                    0,
                    Some(batch_start),
                    Some((batch_start + batch_size as i64).min(n_rows)),
                    1,
                );
                let real = batch.to(device);

                let (mu, std, log_var) = encoder.encode(&real);

                let eps = std.randn_like();
                let emb = &eps * &std + &mu;
                let (rec, sigmas) = decoder.decode(&emb);

                let (loss1, loss2) = calc_loss(
                    &rec,
                    &real,
                    &sigmas,
                    &mu,
                    &log_var,
                    transformer.train_infos(),
                    loss_factor,
                );
                let loss = &loss1 + &loss2;

                total_loss += loss.double_value(&[]);
                loss_count += 1.0;

                optimizer.backward_step(&loss);

                let _ = decoder.sigma.data().clamp_(0.01, 1.0);
            }

            println!("loss: {}", total_loss / loss_count);
        }

        Self {
            decoder,
            batch_size,
            embedding_dim,
            device,
            transformer,
        }
    }

    pub fn sample(&self, samples: usize) -> Vec<ColumnData> {
        let n_steps = next_multiple_of(samples, self.batch_size) / self.batch_size;
        let n_columns = self.transformer.train_infos().len();
        let mut generated_columns = Vec::<ColumnData>::with_capacity(n_columns);
        let mut raw_data = Vec::with_capacity(n_steps * self.batch_size);

        for _ in 0..n_steps {
            let mut noise = Tensor::zeros(
                &[self.batch_size as i64, self.embedding_dim],
                (tch::Kind::Float, self.device),
            );
            let _ = noise.normal_(0.0, 1.0);

            let (fake, _sigmas) = self.decoder.decode(&noise);
            let fake = fake.tanh();

            raw_data.push(fake.detach().to(tch::Device::Cpu));
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

            generated_columns.push(inverse_data);

            start_idx = end_idx;
        }

        generated_columns
    }
}
