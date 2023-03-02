#[cfg(test)]
mod tests;
mod tvae;

use crate::tvae::{ColumnData, TVAE};
use rand::thread_rng;
use tch::Tensor;

fn main() {
    // println!("{}", tch::utils::has_mps());

    // let device =  tch::Device::Mps;

    // let device = if tch::utils::has_mps() {
    //     tch::Device::Mps
    // } else {
    //     println!("No acceleration available");
    //     tch::Device::Cpu
    // };
    let device = tch::Device::Cpu;

    let norm = Tensor::randn(&[5_000], (tch::Kind::Float, tch::Device::Cpu));

    // let a = Tensor::randn()

    let test_data = norm
        .iter::<f64>()
        .unwrap()
        .map(|v| (v.clamp(-3.0, 2.9999) + 3.0) / 6.0 * 10.0)
        .map(|v| v.floor() as i32)
        .collect::<Vec<_>>();



    let mut net = TVAE::fit(&[ColumnData::Discrete(test_data.clone())], 300, 1500, device);
    let generated = net.sample(5_000);

    println!("Real dist:");
    for u in 0..10 {
        let count = test_data.iter().filter(|v| **v == u).count();
        println!("{} - {}", u, count);
    }

    println!("Fake dist:");
    for u in 0..10 {
        let count = generated.iter().filter(|v| **v == u).count();
        println!("{} - {}", u, count);
    }
}
