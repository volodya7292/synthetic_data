use serde::Deserialize;
use std::fs;

use crate::tvae::TVAE;

#[derive(Deserialize)]
struct FullNet {
    netData: String,
}

#[derive(Deserialize)]
struct NetData {
    netInternals: String,
}

#[test]
fn it_works() {
    let data_str = fs::read_to_string("testdata/testnet.json").unwrap();
    let full_net: FullNet = serde_json::from_str(&data_str).unwrap();
    let net_data: NetData = serde_json::from_str(&full_net.netData).unwrap();

    let net = TVAE::load(
        tch::Device::Cpu,
        serde_json::from_str(&net_data.netInternals).unwrap(),
    );

    net.sample(5000);
}
