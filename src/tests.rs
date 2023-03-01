use tch::Tensor;
use crate::tvae::ColumnData;

#[test]
fn column_data_discrete_to_tensor_works() {
    let data = [3, 2, 1, 2];
    let hot_vecs = ColumnData::discrete_to_tensor(&data);

    assert_eq!(hot_vecs.get(0), Tensor::of_slice(&[0, 0, 1]));
    assert_eq!(hot_vecs.get(1), Tensor::of_slice(&[0, 1, 0]));
    assert_eq!(hot_vecs.get(2), Tensor::of_slice(&[1, 0, 0]));
    assert_eq!(hot_vecs.get(3), Tensor::of_slice(&[0, 1, 0]));
}
