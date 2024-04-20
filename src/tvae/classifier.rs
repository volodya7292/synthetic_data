use tch::{
    nn::{self, Adam, Module, OptimizerConfig},
    Device, Tensor,
};

use super::{
    data_transformer::{ColumnInfo, ColumnTrainInfo, DataTransformer},
    input::{ColumnData, ColumnDataRef},
    DoStop,
};

const L_RATE: f64 = 1e-3;

pub struct Model {
    seq: nn::Sequential,
    out_layer: nn::Linear,
}

impl Model {
    pub fn new(vs: &nn::Path, in_dim: i64, out_dim: i64) -> Self {
        let seq = nn::seq()
            .add(nn::linear(vs / "l1", in_dim, 128, Default::default()))
            .add_fn(Tensor::mish)
            .add(nn::linear(vs / "l2", 128, 128, Default::default()))
            .add_fn(Tensor::mish)
            .add(nn::linear(vs / "l3", 128, 128, Default::default()))
            .add_fn(Tensor::mish);

        let out_layer = nn::linear(vs / "out", 128, out_dim, Default::default());

        Self { seq, out_layer }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let f = self.seq.forward(x);
        self.out_layer.forward(&f)
    }
}

fn calc_loss(pred_x: &Tensor, x: &Tensor, output_info: &[ColumnTrainInfo]) -> Tensor {
    let mut start = 0_i64;
    let mut loss = vec![];

    for column_info in output_info {
        let balance_weights = column_info.balance_weights().to(x.device());

        for span_info in column_info.output_spans() {
            let end = start + span_info.dim;

            let x_slice = x.slice(1, start, end, 1);
            let pred_x_slice = pred_x.slice(1, start, end, 1);

            let cross_loss = pred_x_slice.cross_entropy_loss::<&_>(
                &x_slice.argmax(-1, false),
                Some(&balance_weights),
                // None,
                tch::Reduction::Mean,
                -100,
                0.0,
            );

            loss.push(cross_loss);

            start = end;
        }
    }

    assert_eq!(start, pred_x.size()[1]);
    let s = loss.iter().sum::<Tensor>();

    s
}

fn calc_accuracy(pred_x: &Tensor, x: &Tensor, output_info: &[ColumnTrainInfo]) -> Tensor {
    let mut start = 0_i64;
    let mut accuracies = vec![];

    for column_info in output_info {
        for span_info in column_info.output_spans() {
            let end = start + span_info.dim;

            let x_slice = x.slice(1, start, end, 1);
            let pred_x_slice = pred_x.slice(1, start, end, 1);

            let accuracy = pred_x_slice.accuracy_for_logits(&x_slice.argmax(-1, false));
            accuracies.push(accuracy);

            start = end;
        }
    }

    assert_eq!(start, pred_x.size()[1]);
    let s = accuracies.iter().sum::<Tensor>();

    s
}

pub struct Classificator {
    vs: nn::VarStore,
    device: Device,
    model: Model,
    batch_size: usize,
    in_transformer: DataTransformer,
    out_transformer: DataTransformer,
}

impl Classificator {
    /// `flow_control`: Fn(epoch, loss, accuracy) -> DoStop
    pub fn fit<F: Fn(usize, f64, f64) -> DoStop>(
        in_train: &[ColumnDataRef],
        out_train: ColumnDataRef,
        in_test: &[ColumnDataRef],
        out_test: ColumnDataRef,
        batch_size: usize,
        device: Device,
        flow_control: F,
    ) -> Self {
        let mut vs = nn::VarStore::new(device);
        assert!(!in_train.is_empty());
        assert_eq!(in_train[0].len(), out_train.len());

        let in_transformer = DataTransformer::prepare(in_train);
        let out_transformer = DataTransformer::prepare(&[out_train]);
        let n_rows = in_train[0].len() as i64;

        let n_tiles = batch_size.div_ceil(n_rows as usize).max(2);
        let aligned_n_rows = (n_rows as usize).next_multiple_of(batch_size) as i64;

        let train_data = Tensor::cat(
            &in_train
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let transformed = in_transformer.transform(i, v);

                    transformed
                        .repeat([n_tiles as i64, 1])
                        .slice(0, 0, aligned_n_rows, 1)
                })
                .collect::<Vec<_>>(),
            1,
        )
        .totype(tch::Kind::Float)
        .to(device);

        let target_data = {
            let transformed = out_transformer.transform(0, &out_train);
            transformed
                .repeat([n_tiles as i64, 1])
                .slice(0, 0, aligned_n_rows, 1)
        }
        .totype(tch::Kind::Float)
        .to(device);

        let test_in_data = Tensor::cat(
            &in_test
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let transformed = in_transformer.transform(i, v);

                    transformed
                        .repeat([n_tiles as i64, 1])
                        .slice(0, 0, aligned_n_rows, 1)
                })
                .collect::<Vec<_>>(),
            1,
        )
        .totype(tch::Kind::Float)
        .to(device);

        let test_out_data = {
            let transformed = out_transformer.transform(0, &out_test);
            transformed
                .repeat([n_tiles as i64, 1])
                .slice(0, 0, aligned_n_rows, 1)
        }
        .totype(tch::Kind::Float)
        .to(device);

        let in_data_dim = in_transformer
            .train_infos()
            .iter()
            .map(|v| v.total_dim())
            .sum::<i64>();

        let out_data_dim = out_transformer
            .train_infos()
            .iter()
            .map(|v| v.total_dim())
            .sum::<i64>();

        let model = Model::new(&vs.root(), in_data_dim, out_data_dim);
        // let encoder = Encoder::new(&vs.root(), data_dim, &COMPRESS_DIMS, EMBEDDING_DIM);
        // let decoder = Decoder::new(&vs.root(), EMBEDDING_DIM, &DECOMPRESS_DIMS, data_dim);

        let mut optimizer = Adam::default().build(&vs, L_RATE).unwrap();
        let mut epoch = 0;

        loop {
            let shuffle_perm =
                Tensor::randperm(aligned_n_rows, (tch::Kind::Int64, tch::Device::Cpu));
            let curr_train_data = train_data.index(&[Some(&shuffle_perm)]).to(device);
            let curr_target_data = target_data.index(&[Some(&shuffle_perm)]).to(device);

            let mut total_loss = 0.0;
            let mut loss_count = 0.0;

            for batch_start in (0..aligned_n_rows).step_by(batch_size) {
                let batch_train =
                    curr_train_data.slice(0, batch_start, batch_start + batch_size as i64, 1);
                let batch_target =
                    curr_target_data.slice(0, batch_start, batch_start + batch_size as i64, 1);

                let pred = model.forward(&batch_train);

                // let (mu, std, log_var) = encoder.encode(&batch_real);

                // let random_deviations = std.randn_like();
                // let latents = &random_deviations * &std + &mu;
                // let batch_reconstructed = decoder.decode(&latents);

                // let (recon_loss, kl_loss) = calc_loss(
                //     &batch_reconstructed,
                //     &batch_real,
                //     &mu,
                //     &log_var,
                //     &transformer.train_infos(),
                // );

                let loss = calc_loss(&pred, &batch_target, &out_transformer.train_infos());

                // let loss: Tensor = &recon_loss + kl_factor * &kl_loss;

                optimizer.zero_grad();
                loss.backward();
                optimizer.step();

                total_loss += loss.double_value(&[]);
                loss_count += 1.0;
            }

            // println!(
            //     "{} {} {}",
            //     total_recon_loss / loss_count,
            //     total_kl_loss / loss_count,
            //     kl_factor
            // );
            let loss = total_loss / loss_count;

            let accuracy = {
                vs.freeze();
                let pred_x = model.forward(&test_in_data);
                let accuracy =
                    calc_accuracy(&pred_x, &test_out_data, &out_transformer.train_infos());
                vs.unfreeze();
                accuracy.double_value(&[])
            };

            if flow_control(epoch, loss, accuracy) {
                break;
            }
            epoch += 1;
        }

        vs.freeze();

        Self {
            vs,
            model,
            batch_size,
            device,
            in_transformer,
            out_transformer,
        }
    }

    pub fn calc_f1(&self, real_in: &[ColumnDataRef], real_out: ColumnDataRef) -> f32 {
        let real_in = Tensor::cat(
            &real_in
                .iter()
                .enumerate()
                .map(|(i, v)| self.in_transformer.transform(i, v))
                .collect::<Vec<_>>(),
            1,
        )
        .totype(tch::Kind::Float)
        .to(self.device);

        // let real_out_tensor = self.out_transformer.transform(0, &real_out);

        let pred = self.model.forward(&real_in);
        let pred = self.out_transformer.inverse_transform(0, &pred);

        // else {
        // let ColumnData::Discrete(pred) = self.out_transformer.inverse_transform(0, &pred) else {
        //     panic!("Our should be discrete");
        // };

        let target_pdf = self.out_transformer.column_infos()[0].target_pdf();
        // let pred_pdf = self.out_transformer.column_infos()[0].calc_data_pdf(&pred);

        let ColumnDataRef::Discrete(real_out_data) = real_out else {
            panic!("Out should be discrete");
        };
        let ColumnData::Discrete(pred_out_data) = pred else {
            panic!("Out should be discrete");
        };
        let ColumnInfo::Discrete {
            unique_categories, ..
        } = &self.out_transformer.column_infos()[0]
        else {
            panic!("Out should be discrete");
        };

        (0..target_pdf.buckets().len())
            .map(|idx| {
                let cat = unique_categories[idx];

                let iter = real_out_data
                    .iter()
                    .zip(pred_out_data.iter())
                    .filter(|(real, pred)| **real != **pred || **real == cat);

                let v_tp = iter
                    .clone()
                    .filter(|(real, pred)| **real == cat && **pred == cat)
                    .count();

                let v_fp = iter
                    .clone()
                    .filter(|(real, pred)| **real != cat && **pred == cat)
                    .count();

                let v_fn = iter
                    .clone()
                    .filter(|(real, pred)| **real == cat && **pred != cat)
                    .count();

                dbg!(cat, v_tp, v_fp, v_fn);

                (2 * v_tp) as f32 / (2 * v_tp + v_fp + v_fn) as f32
            })
            .sum::<f32>()
            / target_pdf.buckets().len() as f32

        // for v in self.out_transformer.column_infos() {

        //     // v.target_pdf()
        // }

        // 0.0

        // let real_in_trans = real_in.iter().map(|v|)self.in_transformer.transform(column_index, data);

        // f1 = (2 * tp) / (2 * tp + fp + fn);

        // tp: synth 1, real 1
        // fp: synth 1, real 0
        // fn: synth 0, real 1
    }
}
