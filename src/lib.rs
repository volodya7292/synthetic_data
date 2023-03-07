#[cfg(test)]
mod tests;
mod tvae;

use crate::tvae::input::ColumnData;
use crate::tvae::TVAE;
use tch::Tensor;

#[no_mangle]
pub unsafe extern "C" fn main2() {
    let device = tch::Device::Cpu;
    let test_data_continuous = {
        let mut norm = Tensor::zeros(&[5_000], (tch::Kind::Float, tch::Device::Cpu));
        let _ = norm.normal_(3.0, 1.0);
        norm.iter::<f64>()
            .unwrap()
            // .map(|v| (v / 6.0 + 0.5).clamp(0.0, 1.0) as f32)
            .map(|v| (v / 6.0).clamp(0.0, 1.0) as f32)
            .collect::<Vec<_>>()
    };

    let test_data_discrete = {
        // let norm = Tensor::rand(&[5_000], (tch::Kind::Float, tch::Device::Cpu));
        // norm.iter::<f64>()
        //     .unwrap()
        //     .map(|v| (v * 10.0))
        //     .map(|v| v.floor() as i32)
        //     .collect::<Vec<_>>()

        test_data_continuous
            .iter()
            .map(|&v| if v >= 0.7 && v < 0.8 { 5 } else { 6 })
            .collect::<Vec<_>>()
    };

    let net = TVAE::fit(
        &[
            ColumnData::Discrete(test_data_discrete.clone()),
            ColumnData::Continuous(test_data_continuous.clone()),
        ],
        2000,
        500,
        device,
    );
    let generated = net.sample(5_000);
    let ColumnData::Discrete(generated_column0) = &generated[0] else {
            unreachable!()
        };
    let ColumnData::Continuous(generated_column1) = &generated[1] else {
            unreachable!()
        };

    {
        println!("DISCRETE: Real dist:");
        for u in 0..10 {
            let count = test_data_discrete.iter().filter(|v| **v == u).count();
            println!("{} - {}", u, count);
        }

        println!("DISCRETE: Fake dist:");
        for u in 0..10 {
            let count = generated_column0.iter().filter(|v| **v == u).count();
            println!("{} - {}", u, count);
        }
    }

    {
        let buckets_real: Vec<_> = (0..10)
            .map(|buck| {
                let buck = buck as f32;
                test_data_continuous
                    .iter()
                    .map(|v| *v * 10.0)
                    .filter(|v| *v > buck && *v <= (buck + 1.0))
                    .count()
            })
            .collect();
        let buckets_generated: Vec<_> = (0..10)
            .map(|buck| {
                let buck = buck as f32;
                generated_column1
                    .iter()
                    .map(|v| *v * 10.0)
                    .filter(|v| *v >= buck && *v < (buck + 1.0))
                    .count()
            })
            .collect();

        println!(
            "CONTINUOUS: Real dist ({}):",
            buckets_real.iter().sum::<usize>()
        );
        for u in 0..10 {
            println!("{} - {}", u, buckets_real[u]);
        }

        println!(
            "CONTINUOUS: Fake dist ({}):",
            buckets_generated.iter().sum::<usize>()
        );
        for u in 0..10 {
            println!("{} - {}", u, buckets_generated[u]);
        }
    }

    let mut real_regular_count = 0;
    let mut real_specific_count = 0;

    for (&c0, &c1) in test_data_discrete.iter().zip(test_data_continuous.iter()) {
        if c1 >= 0.7 && c1 < 0.8 && c0 == 5 {
            real_specific_count += 1;
        } else {
            real_regular_count += 1;
        }
    }

    let mut gen_regular_count = 0;
    let mut gen_specific_count = 0;

    for (&c0, &c1) in generated_column0.iter().zip(generated_column1.iter()) {
        if c1 >= 0.7 && c1 < 0.8 && c0 == 5 {
            gen_specific_count += 1;
        } else {
            gen_regular_count += 1;
        }
    }

    println!(
        "REAL: regular {}, specific {} ",
        real_regular_count, real_specific_count
    );
    println!(
        "GENERATED: regular {}, specific {} ",
        gen_regular_count, gen_specific_count
    );
}
