use serde::Deserialize;
use std::fs;

use crate::tvae::{input::ColumnDataRef, TVAE};

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

    let net = TVAE::fit(
        &[ColumnDataRef::Discrete(&ages)],
        32,
        tch::Device::Cpu,
        |epoch, loss| {
            println!("epoch {epoch}, loss {loss}");
            epoch >= 100
        },
    );

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
