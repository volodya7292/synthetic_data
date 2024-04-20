use crate::tvae::{
    classifier::Classificator,
    data_transformer::DataTransformer,
    input::{ColumnData, ColumnDataRef},
    TVAE,
};
use serde::Deserialize;
use std::{fs, time::Instant};

#[derive(Deserialize)]
struct FullNet {
    netData: String,
}

#[derive(Deserialize)]
struct NetData {
    netInternals: String,
}

#[test]
fn train_works() {
    let mut data = csv::Reader::from_path("testdata/brain_stroke.csv").unwrap();
    let age_col_id = data
        .headers()
        .unwrap()
        .iter()
        .position(|v| v == "age")
        .unwrap();
    let mut ages = vec![];

    for rec in data.records() {
        let rec = rec.unwrap();
        let age_str = rec.get(age_col_id).unwrap();
        let Ok(age) = age_str.parse::<i32>() else {
            continue;
        };
        ages.push(age);
    }
    assert!(ages.len() > 10);

    let t0 = Instant::now();
    let net = TVAE::fit(
        &[ColumnDataRef::Discrete(&ages)],
        500,
        tch::Device::Cpu,
        |epoch, loss| {
            println!("epoch {epoch}, loss {loss}");
            epoch >= 100
        },
    );
    let t1 = Instant::now();
    println!("train time {}s", (t1 - t0).as_secs_f64());

    net.sample(5000);
}

#[test]
fn sample_works() {
    let data_str = fs::read_to_string("testdata/testnet.json").unwrap();
    let full_net: FullNet = serde_json::from_str(&data_str).unwrap();
    let net_data: NetData = serde_json::from_str(&full_net.netData).unwrap();

    let net = TVAE::load(
        tch::Device::Cpu,
        serde_json::from_str(&net_data.netInternals).unwrap(),
    );

    net.sample(5000);
}

fn load_datasets(
    path_real: &str,
    synth_path: &str,
    num_cols: usize,
    continuous_cols: &[usize],
) -> (DataTransformer, Vec<ColumnData>, Vec<ColumnData>) {
    let mut data = csv::Reader::from_path(path_real).unwrap();
    let mut model = csv::Reader::from_path(synth_path).unwrap();

    let mut real_cols = vec![ColumnData::Discrete(vec![]); num_cols];
    let mut model_cols = vec![ColumnData::Discrete(vec![]); num_cols];

    for i in 0..num_cols {
        if continuous_cols.contains(&i) {
            real_cols[i] = ColumnData::Continuous(vec![]);
            model_cols[i] = ColumnData::Continuous(vec![]);
        }
    }

    for rec in data.records() {
        let rec = rec.unwrap();

        for (i, v) in rec.iter().enumerate() {
            let col = &mut real_cols[i];
            match col {
                ColumnData::Discrete(d) => d.push(v.parse::<f32>().unwrap() as i32),
                ColumnData::Continuous(d) => d.push(v.parse::<f32>().unwrap()),
            }
        }
    }

    for rec in model.records() {
        let rec = rec.unwrap();

        for (i, v) in rec.iter().enumerate() {
            let col = &mut model_cols[i];
            match col {
                ColumnData::Discrete(d) => d.push(v.parse::<f32>().unwrap() as i32),
                ColumnData::Continuous(d) => d.push(v.parse::<f32>().unwrap()),
            }
        }
    }

    for col in &mut real_cols {
        col.pshuffle();
    }
    for col in &mut model_cols {
        col.pshuffle();
    }

    let cols: Vec<_> = real_cols.iter().map(|v| v.as_ref()).collect();

    (DataTransformer::prepare(&cols), real_cols, model_cols)
}

fn eval_dataset(
    path_real: &str,
    path_synth: &str,
    num_cols: usize,
    continuous_cols: &[usize],
    out_col: usize,
    num_epochs: usize,
) {
    let (original, real, model) = load_datasets(path_real, path_synth, num_cols, continuous_cols);

    let uni_dist_hard = original.avg_univariate_hard_dist(&model);
    let uni_dist_l1 = original.avg_univariate_l1_dist(&model);
    let uni_kl_div = DataTransformer::univariate_kl_div(&original, &model);
    let corr_dist = original.pearson_dist(&model);

    let synth_in = model
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != out_col)
        .map(|(_, v)| v.as_ref())
        .collect::<Vec<_>>();
    let synth_out = model[out_col].as_ref();
    let real_in = real
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != out_col)
        .map(|(_, v)| v.as_ref())
        .collect::<Vec<_>>();
    let real_out = real[out_col].as_ref();

    let p = 0.8;//0.8_f32;

    let real_in_train: Vec<_> = real
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != out_col)
        .map(|(_, v)| v.take_train_part(p))
        .collect();
    let real_out_train = real[out_col].take_train_part(p);

    let real_in_test: Vec<_> = real
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != out_col)
        .map(|(_, v)| v.take_test_part(p))
        .collect();
    let real_out_test = real[out_col].take_test_part(p);

    let classifier_real = Classificator::fit(
        &real_in_train,
        real_out_train,
        &real_in_test,
        real_out_test,
        32,
        tch::Device::Cpu,
        |epoch, loss, acc| {
            println!("real class loss {loss} | acc {acc}");
            epoch >= num_epochs
        },
    );
    let ident_f1 = classifier_real.calc_f1(&real_in_test, real_out_test);
    dbg!(ident_f1);

    let classifier = Classificator::fit(
        &synth_in,
        synth_out,
        &real_in,
        real_out,
        32,
        tch::Device::Cpu,
        |epoch, loss, acc| {
            println!("class loss {loss} | acc {acc}");
            epoch >= num_epochs
        },
    );
    let f1 = classifier.calc_f1(&real_in, real_out);

    dbg!(uni_dist_hard, uni_dist_l1, uni_kl_div, corr_dist, f1);
}

#[test]
fn eval_brain() {
    eval_dataset(
        "testdata/brain_stroke_d.csv",
        "testdata/brain_stroke_sdv_d.csv",
        11,
        &[1, 7, 8],
        10,
        50,
    );
}

#[test]
fn eval_diabetes() {
    eval_dataset(
        "testdata/diabetes.csv",
        "testdata/diabetes_ours2.csv",
        22,
        &[4, 14, 15, 16, 19],
        0,
        20,
    );
}

#[test]
fn eval_credit() {
    eval_dataset(
        "testdata/creditcard_reduced.csv",
        "testdata/creditcard_reduced_mostly.csv",
        30,
        &[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28,
        ],
        29,
        5,
    );
}
