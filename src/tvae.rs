use tch::nn::{Adam, Module, OptimizerConfig};
use tch::{nn, Tensor};

pub enum CellData {
    Discrete(u32),
    Real(f32),
}

pub enum ColumnData {
    Discrete(Vec<i32>),
    Continuous(Vec<f32>),
}

impl ColumnData {
    pub fn discrete_to_tensor(data: &[i32]) -> Tensor {
        let mut uniques = data.to_vec();
        uniques.sort_unstable();
        uniques.dedup();

        let n_rows = data.len() as i64;
        let n_uniques = uniques.len() as i64;

        let data_tensor = Tensor::of_slice(data);
        let uniques_tensor = Tensor::of_slice(&uniques);

        let data_x_uniques = data_tensor.broadcast_to(&[n_uniques, n_rows]);
        let uniques_x_data = uniques_tensor.broadcast_to(&[n_rows, n_uniques]);

        let hot_vectors = data_x_uniques.transpose(0, 1).eq_tensor(&uniques_x_data);

        hot_vectors.totype(tch::Kind::Int8)
    }

    pub fn to_train_data(&self) -> Tensor {
        match self {
            ColumnData::Discrete(d) => Self::discrete_to_tensor(d),
            ColumnData::Continuous(_) => todo!(),
        }
    }

    pub fn info(&self) -> ColumnInfo {
        match self {
            ColumnData::Discrete(d) => {
                let mut uniques = d.to_vec();
                uniques.sort_unstable();
                uniques.dedup();

                /*

                     ColumnTransformInfo(
                column_name=column_name, column_type='discrete', transform=ohe,
                output_info=[SpanInfo(num_categories, 'softmax')],
                output_dimensions=num_categories)
                     */

                ColumnInfo {
                    spans: vec![SpanInfo {
                        dim: uniques.len(),
                        func: "softmax",
                    }],
                    output_dims: uniques.len() as i64,
                }
            }
            ColumnData::Continuous(_) => {
                todo!()
            }
        }
    }

    pub fn len(&self) -> usize {
        match self {
            ColumnData::Discrete(data) => data.len(),
            ColumnData::Continuous(data) => data.len(),
        }
    }
}

pub struct Row {
    columns: Vec<CellData>,
}

pub struct SpanInfo {
    dim: usize,
    func: &'static str,
}

pub struct ColumnInfo {
    spans: Vec<SpanInfo>,
    output_dims: i64,
}

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

// NOTE: python x[:, 3] = rust x.slice(0, 0, len(x)).slice(1, 3, 4)

fn calc_loss(
    recon_x: &Tensor,
    x: &Tensor,
    sigmas: &Tensor,
    mu: &Tensor,
    log_var: &Tensor,
    output_info: &[ColumnInfo],
    factor: f32,
) -> (Tensor, Tensor) {
    let mut st = 0_i64;
    let mut loss = vec![];

    for column_info in output_info {
        for span_info in &column_info.spans {
            let ed = st + span_info.dim as i64;

            if span_info.func != "softmax" {
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
    encoder: Encoder,
    decoder: Decoder,
    batch_size: usize,
    embedding_dim: i64,
    device: tch::Device,
}

fn next_multiple_of(n: i64, multiple: i64) -> i64 {
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

        let column_infos: Vec<_> = data.iter().map(|column| column.info()).collect();

        let train_column_data: Vec<_> = data.iter().map(|column| column.to_train_data()).collect();
        let train_data = Tensor::cat(&train_column_data, 1).totype(tch::Kind::Float);

        let data_dim = column_infos.iter().map(|v| v.output_dims).sum::<i64>();

        let encoder = Encoder::new(&vs.root(), data_dim, &compress_dims, embedding_dim);
        let decoder = Decoder::new(&vs.root(), embedding_dim, &decompress_dims, data_dim);

        let mut optimizer = Adam::default().wd(l2scale).build(&vs, 1e-3).unwrap();

        let n_rows = train_data.size()[0];
        // let n_rows_aligned = next_multiple_of(n_rows, batch_size as i64);

        for i in 0..epochs {
            print!("Epoch #{}... ", i);

            let shuffle_perm = Tensor::randperm(n_rows, (tch::Kind::Int64, device));
            let curr_train_data = train_data.index(&[Some(shuffle_perm)]);

            let mut total_loss = 0.0;

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
                    &column_infos,
                    loss_factor,
                );
                let loss = loss1 + loss2;

                total_loss += loss.double_value(&[]);

                optimizer.backward_step(&loss);

                let _ = decoder.sigma.data().clamp_(0.01, 1.0);
            }

            println!("loss: {}", total_loss);
        }

        Self {
            encoder,
            decoder,
            batch_size,
            embedding_dim,
            device,
        }
    }

    // pub fn sample(&self, n_batches: usize) -> Vec<Row> {
    pub fn sample(&self, samples: usize) -> Vec<i32> {
        let steps = samples / self.batch_size + 1;
        let mut rows = Vec::<Row>::with_capacity(steps * self.batch_size);
        let mut raw_data = Vec::with_capacity(rows.len());
        let mut sigmas = Tensor::new();

        for _ in 0..steps {
            let mut noise = Tensor::zeros(
                &[self.batch_size as i64, self.embedding_dim],
                (tch::Kind::Float, self.device),
            );
            let _ = noise.normal_(0.0, 1.0);

            let (fake, _sigmas) = self.decoder.decode(&noise);
            let fake = fake.tanh();

            raw_data.push(fake.detach().to(tch::Device::Cpu));
            sigmas = _sigmas;
        }

        let data = Tensor::cat(&raw_data, 0);

        let indices = data.argmax(Some(1), false);

        // TODO: map indices to unique items

        // println!("{}", indices);

        indices
            .iter::<i64>()
            .unwrap()
            .map(|v| v as i32)
            .take(samples)
            .collect()

        // rows
    }
}
