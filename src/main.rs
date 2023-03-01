mod tvae;
#[cfg(test)]
mod tests;

use crate::tvae::{ColumnData, TVAE};
use tch::Tensor;

fn main() {
    let tensor = Tensor::stack(
        &[
            Tensor::stack(&[Tensor::of_slice(&[1, 2]), Tensor::of_slice(&[3, 4])], 0),
            Tensor::stack(&[Tensor::of_slice(&[5, 6]), Tensor::of_slice(&[7, 8])], 0),
        ],
        0,
    );
    // let tensor = Tensor::stack(&[Tensor::of_slice(&[1, 2]), Tensor::of_slice(&[3, 4])], 0);
    println!("{}", &tensor);
    // println!("{:?}", &tensor.size());
    println!(
        "{}",
        // t[:, 1]
        tensor
            .slice_copy(1, Some(1), Some(2), 1)
            // .flatten(1, 2)
            .to_string(100)
            .unwrap()


    );
    // let t = Tensor::of_slice2(&[&Tensor::of_slice(&[])]);

    println!("------------------------------------");

    let mut gan = TVAE::new();
    gan.fit(&[ColumnData::Discrete(vec![1, 2, 3])], 10, 500);
}
