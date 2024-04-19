use crate::tvae::{
    data_transformer::DataTransformer,
    input::{ColumnData, ColumnDataRef},
    TVAE,
};
use serde::Deserialize;
use std::{fs, iter, time::Instant};

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

#[test]
fn trans() {
    fn load(real: &str, model: &str) -> (DataTransformer, Vec<ColumnData>) {
        let mut real = csv::Reader::from_path(real).unwrap();
        let mut model = csv::Reader::from_path(model).unwrap();

        let mut continuous = vec![vec![]; 29];
        let mut out_class = vec![];

        let mut model_cols: Vec<_> = iter::repeat(ColumnData::Continuous(vec![]))
            .take(29)
            .chain(iter::once(ColumnData::Discrete(vec![])))
            .collect();

        for rec in real.records() {
            let rec = rec.unwrap();

            for (i, v) in rec.iter().enumerate() {
                if i == 29 {
                    out_class.push(v.parse::<i32>().unwrap());
                } else {
                    continuous[i].push(v.parse::<f32>().unwrap());
                }
            }
        }
        for rec in model.records() {
            let rec = rec.unwrap();
            for (i, v) in rec.iter().enumerate() {
                let col = &mut model_cols[i];
                match col {
                    ColumnData::Discrete(d) => d.push(v.parse::<i32>().unwrap()),
                    ColumnData::Continuous(d) => d.push(v.parse::<f32>().unwrap()),
                }
            }
        }

        let cols: Vec<_> = continuous
            .iter()
            .map(|v| ColumnDataRef::Continuous(v))
            .chain(iter::once(ColumnDataRef::Discrete(&out_class)))
            .collect();

        (DataTransformer::prepare(&cols), model_cols)
    }

    let (original, model) = load(
        "testdata/creditcard_reduced.csv",
        "testdata/creditcard_reduced_tonic.csv",
    );

    let uni_dist_hard = original.avg_univariate_hard_dist(&model);
    let uni_dist_l1 = original.avg_univariate_l1_dist(&model);
    let uni_kl_div = DataTransformer::univariate_kl_div(&original, &model);
    let corr_dist = original.pearson_dist(&model);

    dbg!(uni_dist_hard, uni_dist_l1, uni_kl_div, corr_dist);
}
