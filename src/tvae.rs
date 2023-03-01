use rand::prelude::SliceRandom;
use tch::nn::{Adam, Module, OptimizerConfig};
use tch::{nn, Tensor};

pub enum CellData {
    Discrete(u32),
    Real(f64),
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

fn make_encoder(
    vs: &nn::Path,
    data_dim: i64,
    compress_dims: &[i64],
    embedding_dim: i64,
) -> impl Module {
    let vs = vs / "encoder";

    let mut seq = nn::seq();
    let mut curr_dim = data_dim;

    for (i, dim) in compress_dims.iter().enumerate() {
        seq = seq
            .add(nn::linear(
                &vs / format!("seq{}", i),
                data_dim,
                curr_dim,
                Default::default(),
            ))
            .add_fn(Tensor::relu);
        curr_dim = *dim;
    }

    let fc1 = nn::linear(&vs / "fc1", data_dim, embedding_dim, Default::default());
    let fc2 = nn::linear(&vs / "fc2", data_dim, embedding_dim, Default::default());

    nn::func(move |input| {
        let feature = seq.forward(input);
        let mu = fc1.forward(&feature);
        let log_var = fc2.forward(&feature);
        let std = (0.5_f32 * &log_var).exp();

        Tensor::stack(&[mu, log_var, std], 0)
    })
}

fn build_decoder(
    vs: &nn::Path,
    embedding_dim: i64,
    decompress_dims: &[i64],
    data_dim: i64,
) -> impl Module {
    let vs = vs / "decoder";

    let mut seq = nn::seq();
    let mut curr_dim = embedding_dim;

    for (i, dim) in decompress_dims.iter().chain(&[data_dim]).enumerate() {
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

    let sigma = Tensor::ones(&[data_dim], (tch::Kind::Float, vs.device())) * 0.1;

    nn::func(move |input| {
        let out = seq.forward(input);
        Tensor::stack(&[out, sigma.shallow_clone()], 0)
    })
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
) -> Tensor {
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

    let kld = -0.5 * (log_var - mu.pow_tensor_scalar(2) - log_var.exp() + 1).sum(tch::Kind::Float);

    let s = loss.iter().sum::<Tensor>() * factor as f64 / x.size()[0];
    let kld_norm = kld / x.size()[0];

    Tensor::stack(&[s, kld_norm], 0)
}

pub struct TVAE {}

impl TVAE {
    pub fn new() -> Self {
        // let vs = nn::VarStore::new(tch::Device::Cpu);
        // let encoder = encoder(&vs.root(), 5, &[128, 128], 128);

        Self {}
    }

    pub fn fit(&mut self, data: &[ColumnData], epochs: usize, batch_size: usize) {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        assert!(data.len() > 0);

        // tch::data::Iter2::new

        let l2scale = 1e-5;
        let embedding_dim = 128;
        let compress_dims = [128, 128];
        let decompress_dims = [128, 128];

        let column_infos: [ColumnInfo; 0] = []; // TODO

        let train_column_data: Vec<_> = data.iter().map(|column| column.to_train_data()).collect();
        let train_data = Tensor::cat(&train_column_data, 1).totype(tch::Kind::Float);

        let data_dim = column_infos.iter().map(|v| v.output_dims).sum::<i64>();

        let encoder = make_encoder(&vs.root(), data_dim, &compress_dims, embedding_dim);
        let decoder = build_decoder(&vs.root(), embedding_dim, &decompress_dims, data_dim);

        let mut optimizer = Adam::default().wd(l2scale).build(&vs, 1e-3).unwrap();
        optimizer.zero_grad();

        let mut train_indices = (0..train_data.size()[0]).collect::<Vec<_>>();
        let mut rng = rand::thread_rng();

        for i in 0..epochs {
            train_indices.shuffle(&mut rng);

            for ids_batch in train_indices.chunks(batch_size) {
                let batch = Tensor::stack(
                    &ids_batch
                        .iter()
                        .map(|v| train_data.get(*v))
                        .collect::<Vec<_>>(),
                    0,
                );

                // TODO
                // let batch =
            }

            todo!()
        }
    }

    pub fn sample() -> impl Iterator<Item = Row> {
        std::iter::once(Row { columns: vec![] })
    }
}
