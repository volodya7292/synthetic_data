use crate::tvae::input::{ColumnData, ColumnDataRef};
use crate::utils;
use serde::{Deserialize, Deserializer, Serialize};
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

#[derive(Serialize, Deserialize)]
pub enum ColumnInfo {
    Discrete {
        unique_categories: Vec<i32>,
        pdf: Vec<usize>,
    },
    Continuous {
        min: f32,
        max: f32,
        pdf: Vec<usize>,
    },
}

impl ColumnInfo {
    pub fn calc_l1_distance(&self, data: &ColumnData) -> f32 {
        match (self, data) {
            (
                ColumnInfo::Discrete {
                    unique_categories,
                    pdf: real_pdf,
                },
                ColumnData::Discrete(data),
            ) => {
                let data_pdf = utils::calc_discrete_pdf(unique_categories, data);
                utils::l1_distance_between_pdfs(&data_pdf, real_pdf)
            }
            (
                ColumnInfo::Continuous {
                    min,
                    max,
                    pdf: real_pdf,
                },
                ColumnData::Continuous(data),
            ) => {
                let n_buckets = real_pdf.len();
                let data_pdf = utils::calc_continuous_pdf(*min, *max, data, n_buckets);
                utils::l1_distance_between_pdfs(&data_pdf, real_pdf)
            }
            _ => panic!("Invalid combination"),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct DataTransformer {
    column_infos: Vec<ColumnInfo>,
    #[serde(deserialize_with = "deserialize_correlation_matrix")]
    correlation_matrix: Vec<f32>,
}

fn deserialize_correlation_matrix<'de, D>(deserializer: D) -> Result<Vec<f32>, D::Error>
where
    D: Deserializer<'de>,
{
    let str_vec: Vec<serde_json::Value> = Deserialize::deserialize(deserializer)?;
    Ok(str_vec
        .iter()
        .map(|str| str.as_f64().unwrap_or(f64::NAN) as f32)
        .collect())
}

impl DataTransformer {
    pub fn prepare(columns: &[ColumnDataRef]) -> Self {
        if columns.is_empty() {
            return Self {
                column_infos: vec![],
                correlation_matrix: vec![],
            };
        }

        let n_rows = columns[0].len();
        if columns.iter().any(|v| v.len() != n_rows) {
            panic!("All columns must have the same number of rows");
        }

        let column_infos: Vec<_> = columns
            .iter()
            .map(|column| match column {
                ColumnDataRef::Discrete(data) => {
                    let mut uniques = data.to_vec();
                    uniques.sort_unstable();
                    uniques.dedup();

                    let pdf = utils::calc_discrete_pdf(&uniques, data);

                    ColumnInfo::Discrete {
                        unique_categories: uniques,
                        pdf,
                    }
                }
                ColumnDataRef::Continuous(data) => {
                    let filtered: Vec<_> = data.iter().cloned().filter(|v| v.is_finite()).collect();

                    let min = filtered
                        .iter()
                        .cloned()
                        .min_by(|a, b| a.total_cmp(b))
                        .unwrap_or(f32::NAN);
                    let max = filtered
                        .iter()
                        .cloned()
                        .max_by(|a, b| a.total_cmp(b))
                        .unwrap_or(f32::NAN);

                    let n_buckets = (data.len() as f64).sqrt().max(1.0) as usize;
                    let pdf = utils::calc_continuous_pdf(min, max, data, n_buckets);

                    ColumnInfo::Continuous { min, max, pdf }
                }
            })
            .collect();

        let correlation_matrix = utils::calc_correlation_matrix(columns);

        Self {
            column_infos,
            correlation_matrix,
        }
    }

    pub fn train_infos(&self) -> Vec<ColumnTrainInfo> {
        self.column_infos
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
            .collect()
    }

    pub fn column_infos(&self) -> &[ColumnInfo] {
        &self.column_infos
    }

    pub fn correlation_matrix(&self) -> &[f32] {
        &self.correlation_matrix
    }

    pub fn save(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap()
    }

    pub fn load(data: &serde_json::Value) -> Self {
        Self::deserialize(data).unwrap()
    }

    pub fn transform(&self, column_index: usize, data: &ColumnDataRef) -> Tensor {
        let column_info = &self.column_infos[column_index];
        let n_rows = data.len() as i64;

        match (data, column_info) {
            (
                ColumnDataRef::Discrete(data),
                ColumnInfo::Discrete {
                    unique_categories, ..
                },
            ) => {
                let n_uniques = unique_categories.len() as i64;

                let data_tensor = Tensor::from_slice(data);
                let uniques_tensor = Tensor::from_slice(unique_categories);

                let data_x_uniques = data_tensor.broadcast_to([n_uniques, n_rows]);
                let uniques_x_data = uniques_tensor.broadcast_to([n_rows, n_uniques]);

                let hot_vectors = data_x_uniques.transpose(0, 1).eq_tensor(&uniques_x_data);

                hot_vectors.totype(tch::Kind::Int8)
            }
            (ColumnDataRef::Continuous(data), ColumnInfo::Continuous { min, max, .. }) => {
                let range = max - min;
                let filtered = Tensor::from_slice(data);
                let normalized = (filtered - *min as f64) / range as f64;

                normalized.reshape([data.len() as i64, 1])
            }
            _ => panic!("Invalid column data type"),
        }
    }

    pub fn inverse_transform(&self, column_index: usize, data: &Tensor) -> ColumnData {
        let n_rows = data.size()[0];

        match &self.column_infos[column_index] {
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

                let data_norm = data.tanh();
                let range = max - min;
                let clamped = data_norm.clamp(0.0, 1.0);
                let tensor_inverse = clamped * range as f64 + *min as f64;

                for i in 0..n_rows {
                    let value = tensor_inverse.double_value(&[i]) as f32;
                    out_data.push(value)
                }

                ColumnData::Continuous(out_data)
            }
        }
    }
}