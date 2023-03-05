use crate::tvae::input::ColumnData;
use tch::Tensor;

pub struct SpanInfo {
    pub dim: i64,
    pub activation: &'static str,
}

pub struct ColumnTrainInfo {
    output_spans: Vec<SpanInfo>,
}

impl ColumnTrainInfo {
    pub fn output_spans(&self) -> &[SpanInfo] {
        &self.output_spans
    }

    pub fn total_dim(&self) -> i64 {
        self.output_spans.iter().map(|v| v.dim).sum::<i64>()
    }
}

pub enum ColumnInfo {
    Discrete { unique_categories: Vec<i32> },
    Continuous { min: f32, max: f32 },
}

pub struct DataTransformer {
    fit_infos: Vec<ColumnInfo>,
    train_infos: Vec<ColumnTrainInfo>,
    n_rows: usize,
}

impl DataTransformer {
    pub fn prepare(columns: &[ColumnData]) -> Self {
        if columns.is_empty() {
            return Self {
                fit_infos: vec![],
                train_infos: vec![],
                n_rows: 0,
            };
        }

        let n_rows = columns[0].len();
        if columns.iter().any(|v| v.len() != n_rows) {
            panic!("All columns must have the same number of rows");
        }

        let columns: Vec<_> = columns
            .iter()
            .map(|column| match column {
                ColumnData::Discrete(data) => {
                    let mut uniques = data.clone();
                    uniques.sort_unstable();
                    uniques.dedup();

                    ColumnInfo::Discrete {
                        unique_categories: uniques,
                    }
                }
                ColumnData::Continuous(data) => {
                    let filtered: Vec<_> = data.iter().cloned().filter(|v| v.is_finite()).collect();

                    let min = filtered
                        .iter()
                        .cloned()
                        .min_by(|a, b| a.total_cmp(&b))
                        .unwrap_or(f32::NAN);
                    let max = filtered
                        .iter()
                        .cloned()
                        .max_by(|a, b| a.total_cmp(&b))
                        .unwrap_or(f32::NAN);

                    ColumnInfo::Continuous { min, max }
                }
            })
            .collect();

        let train_infos: Vec<_> = columns
            .iter()
            .map(|data| match data {
                ColumnInfo::Discrete {
                    unique_categories, ..
                } => ColumnTrainInfo {
                    output_spans: vec![SpanInfo {
                        dim: unique_categories.len() as i64,
                        activation: "softmax",
                    }],
                },
                ColumnInfo::Continuous { .. } => ColumnTrainInfo {
                    output_spans: vec![SpanInfo {
                        dim: 1,
                        activation: "tanh",
                    }],
                },
            })
            .collect();

        Self {
            fit_infos: columns,
            train_infos,
            n_rows,
        }
    }

    pub fn transform(&self, column_index: usize, data: &ColumnData) -> Tensor {
        let column_info = &self.fit_infos[column_index];
        let n_rows = self.n_rows as i64;

        match (data, column_info) {
            (
                ColumnData::Discrete(data),
                ColumnInfo::Discrete {
                    unique_categories, ..
                },
            ) => {
                let n_uniques = unique_categories.len() as i64;

                let data_tensor = Tensor::of_slice(data);
                let uniques_tensor = Tensor::of_slice(&unique_categories);

                let data_x_uniques = data_tensor.broadcast_to(&[n_uniques, n_rows]);
                let uniques_x_data = uniques_tensor.broadcast_to(&[n_rows, n_uniques]);

                let hot_vectors = data_x_uniques.transpose(0, 1).eq_tensor(&uniques_x_data);
                let transformed = hot_vectors.totype(tch::Kind::Int8);

                transformed
            }
            (ColumnData::Continuous(data), ColumnInfo::Continuous { min, max, .. }) => {
                let range = max - min;
                let filtered = Tensor::of_slice(data);
                let normalized = (filtered - *min as f64) / range as f64;
                let transformed = normalized.reshape(&[data.len() as i64, 1]);

                transformed
            }
            _ => panic!("Invalid column data type"),
        }
    }

    pub fn inverse_transform(&self, column_index: usize, data: &Tensor) -> ColumnData {
        let n_rows = data.size()[0];

        match &self.fit_infos[column_index] {
            ColumnInfo::Discrete {
                unique_categories, ..
            } => {
                let mut out_data = Vec::with_capacity(n_rows as usize);
                let category_indices = data.argmax(Some(1), false);

                for idx in category_indices.iter::<i64>().unwrap() {
                    let inverse = unique_categories[idx as usize];
                    out_data.push(inverse);
                }

                ColumnData::Discrete(out_data)
            }
            ColumnInfo::Continuous { min, max, .. } => {
                let mut out_data = Vec::with_capacity(n_rows as usize);

                let range = max - min;
                let tensor_inverse = data * range as f64 + *min as f64;

                for i in 0..n_rows {
                    let value = tensor_inverse.double_value(&[i]) as f32;
                    out_data.push(value)
                }

                ColumnData::Continuous(out_data)
            }
        }
    }

    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    pub fn train_infos(&self) -> &[ColumnTrainInfo] {
        &self.train_infos
    }
}
