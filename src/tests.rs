use crate::tvae::{data_transformer::DataTransformer, input::ColumnDataRef, TVAE};
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
    fn load(path: &str) -> DataTransformer {
        let mut data = csv::Reader::from_path(path).unwrap();

        let mut continuous = vec![vec![]; 29];
        let mut out_class = vec![];

        for rec in data.records() {
            let rec = rec.unwrap();

            for (i, v) in rec.iter().enumerate() {
                if i == 29 {
                    out_class.push(v.parse::<i32>().unwrap());
                } else {
                    continuous[i].push(v.parse::<f32>().unwrap());
                }
            }
        }

        let cols: Vec<_> = continuous
            .iter()
            .map(|v| ColumnDataRef::Continuous(v))
            .chain(iter::once(ColumnDataRef::Discrete(&out_class)))
            .collect();

        DataTransformer::prepare(&cols)
    }

    let original = load("testdata/creditcard_reduced.csv");
    let mostly = load("testdata/creditcard_reduced_mostly.csv");

    let uni_simil = original.avg_univariate_similarity_hard(&mostly);
    let corr_simil = original.avg_pearson_similarity(&mostly);

    dbg!(uni_simil, corr_simil);
}
