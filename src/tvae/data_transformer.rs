use crate::tvae::input::{ColumnData, ColumnDataRef};
use crate::utils::{self, Pdf};
use rand::Rng;
use serde::{Deserialize, Deserializer, Serialize};
use tch::Tensor;

pub struct SpanInfo {
    pub dim: i64,
    pub activation: &'static str,
}

pub struct ColumnTrainInfo {
    output_spans: Vec<SpanInfo>,
    balance_weights: Tensor,
}

impl ColumnTrainInfo {
    pub fn output_spans(&self) -> &[SpanInfo] {
        &self.output_spans
    }

    // Weights for rebalancing of under/over-represented categories
    pub fn balance_weights(&self) -> &Tensor {
        &self.balance_weights
    }

    pub fn total_dim(&self) -> i64 {
        self.output_spans.iter().map(|v| v.dim).sum::<i64>()
    }
}

#[derive(Serialize, Deserialize)]
pub enum ColumnInfo {
    Discrete {
        unique_categories: Vec<i32>,
        pdf: Pdf,
    },
    Continuous {
        min: f32,
        max: f32,
        pdf: Pdf,
    },
}

impl ColumnInfo {
    pub fn calc_data_pdf(&self, data: &ColumnData) -> Pdf {
        match (self, data) {
            (
                ColumnInfo::Discrete {
                    unique_categories, ..
                },
                ColumnData::Discrete(data),
            ) => utils::calc_discrete_pdf(unique_categories, data),
            (
                ColumnInfo::Continuous {
                    min,
                    max,
                    pdf: real_pdf,
                },
                ColumnData::Continuous(data),
            ) => {
                let n_buckets = real_pdf.buckets().len();
                utils::calc_continuous_pdf(*min, *max, data, n_buckets)
            }
            _ => panic!("Invalid combination"),
        }
    }

    pub fn calc_similarity(&self, data: &ColumnData) -> f32 {
        let data_pdf = self.calc_data_pdf(data);
        let target_pdf = self.target_pdf();
        data_pdf.similarity(target_pdf)
    }

    pub fn calc_hard_distance(&self, data: &ColumnData) -> f32 {
        let data_pdf = self.calc_data_pdf(data);
        let target_pdf = self.target_pdf();
        data_pdf.hard_distance(target_pdf)
    }

    pub fn calc_l1_distance(&self, data: &ColumnData) -> f32 {
        let data_pdf = self.calc_data_pdf(data);
        let target_pdf = self.target_pdf();
        data_pdf.l1_distance(target_pdf)
    }

    pub fn calc_kl(&self, data: &ColumnData) -> f32 {
        let data_pdf = self.calc_data_pdf(data);
        let target_pdf = self.target_pdf();
        Pdf::kl_div(target_pdf, &data_pdf)
    }

    pub fn target_pdf(&self) -> &Pdf {
        match self {
            ColumnInfo::Discrete { pdf, .. } => pdf,
            ColumnInfo::Continuous { pdf, .. } => pdf,
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

                    let n_buckets = (data.len() as f64).sqrt().clamp(1.0, 100.0) as usize;
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
                    unique_categories,
                    pdf,
                    ..
                } => ColumnTrainInfo {
                    output_spans: vec![SpanInfo {
                        dim: unique_categories.len() as i64,
                        activation: "softmax",
                    }],
                    balance_weights: Tensor::from_slice(&pdf.calc_balance_weights()),
                },
                ColumnInfo::Continuous { pdf, .. } => ColumnTrainInfo {
                    output_spans: vec![SpanInfo {
                        dim: pdf.buckets().len() as i64,
                        activation: "softmax",
                    }],
                    balance_weights: Tensor::from_slice(&pdf.calc_balance_weights()),
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
            (ColumnDataRef::Continuous(data), ColumnInfo::Continuous { min, max, pdf, .. }) => {
                let range = max - min;
                let normalized = (Tensor::from_slice(data) - *min as f64) / range as f64;
                let n_buckets = pdf.buckets().len() as i64;

                let data_tensor = (normalized * n_buckets)
                    .floor()
                    .clamp(0.0, n_buckets as f64 - 1.0)
                    .totype(tch::Kind::Int64);

                let uniques_tensor =
                    Tensor::from_slice(&(0..n_buckets as i32).collect::<Vec<i32>>());

                let data_x_uniques = data_tensor.broadcast_to([n_buckets, n_rows]);
                let uniques_x_data = uniques_tensor.broadcast_to([n_rows, n_buckets]);

                let hot_vectors = data_x_uniques.transpose(0, 1).eq_tensor(&uniques_x_data);

                hot_vectors.totype(tch::Kind::Int8)
            }
            _ => panic!("Invalid column data type"),
        }
    }

    pub fn inverse_transform(&self, column_index: usize, data: &Tensor) -> ColumnData {
        let n_rows = data.size()[0];
        let mut rng = rand::thread_rng();

        match &self.column_infos[column_index] {
            ColumnInfo::Discrete {
                unique_categories, ..
            } => {
                let mut out_data = Vec::with_capacity(n_rows as usize);
                let category_indices = data.argmax(1, false);

                for idx in category_indices.iter::<i64>().unwrap() {
                    let inverse = unique_categories[idx as usize];
                    out_data.push(inverse);
                }

                ColumnData::Discrete(out_data)
            }
            ColumnInfo::Continuous { min, max, pdf, .. } => {
                let num_buckets = pdf.buckets().len();
                let range = max - min;
                let bucket_width = range / num_buckets as f32;

                let out_data: Vec<_> = (0..n_rows)
                    .map(|row_idx| {
                        let row = data.get(row_idx);
                        let max_idx = row.argmax(0, false).int64_value(&[]);

                        let bucket_idx = max_idx;
                        let bucket_min = min + bucket_width * bucket_idx as f32;
                        let bucket_max = bucket_min + bucket_width;

                        rng.gen_range::<f32, _>(bucket_min..bucket_max)
                    })
                    .collect();

                ColumnData::Continuous(out_data)
            }
        }
    }

    pub fn inverse_transform_indexed(&self, column_index: usize, indices: &Tensor) -> ColumnData {
        let n_rows = indices.size()[0];
        let mut rng = rand::thread_rng();

        match &self.column_infos[column_index] {
            ColumnInfo::Discrete {
                unique_categories, ..
            } => {
                let mut out_data = Vec::with_capacity(n_rows as usize);
                let category_indices = indices;

                for idx in category_indices.iter::<i64>().unwrap() {
                    let inverse = unique_categories[idx as usize];
                    out_data.push(inverse);
                }

                ColumnData::Discrete(out_data)
            }
            ColumnInfo::Continuous { min, max, pdf, .. } => {
                let num_buckets = pdf.buckets().len();
                let range = max - min;
                let bucket_width = range / num_buckets as f32;

                let out_data: Vec<_> = (0..n_rows)
                    .map(|row_idx| {
                        let max_idx = indices.int64_value(&[row_idx]);

                        let bucket_idx = max_idx;
                        let bucket_min = min + bucket_width * bucket_idx as f32;
                        let bucket_max = bucket_min + bucket_width;

                        rng.gen_range::<f32, _>(bucket_min..bucket_max)
                    })
                    .collect();

                ColumnData::Continuous(out_data)
            }
        }
    }

    pub fn inverse_samples_to_indices(&self, full_data: &Tensor) -> Tensor {
        let mut start_idx = 0;
        let mut cols_out = vec![];

        for train_info in self.train_infos().iter() {
            let end_idx = start_idx + train_info.total_dim();

            let col_hots = full_data.slice(1, start_idx, end_idx, 1);
            let col_values = col_hots.argmax(1, false);

            cols_out.push(col_values);
            start_idx = end_idx;
        }

        Tensor::stack(&cols_out, 1)
    }

    pub fn full_hot_to_indices(&self, input: &Tensor, out: &mut [i64]) {
        let mut start_idx = 0;
        for (col_idx, train_info) in self.train_infos().iter().enumerate() {
            let end_idx = start_idx + train_info.total_dim();
            let generated_hot_value = input.slice(0, start_idx, end_idx, 1);
            let generated_value = generated_hot_value.argmax(0, false).int64_value(&[]);

            out[col_idx] = generated_value;
            start_idx = end_idx;
        }
    }

    pub fn avg_univariate_l1_dist(&self, columns: &[ColumnData]) -> f32 {
        let n = self.column_infos().len() as f32;
        self.column_infos()
            .iter()
            .zip(columns.iter())
            .map(|(a, b)| a.calc_l1_distance(b))
            .sum::<f32>()
            / n
    }

    pub fn avg_univariate_hard_dist(&self, columns: &[ColumnData]) -> f32 {
        let n = self.column_infos().len() as f32;
        self.column_infos()
            .iter()
            .zip(columns.iter())
            .map(|(a, b)| a.calc_hard_distance(b))
            .sum::<f32>()
            / n
    }

    /// Ccmputes D_KL (P || Q)
    pub fn univariate_kl_div(p: &Self, q: &[ColumnData]) -> f32 {
        p.column_infos()
            .iter()
            .zip(q.iter())
            .map(|(a, b)| a.calc_kl(b))
            .sum::<f32>()
    }

    pub fn pearson_dist(&self, columns: &[ColumnData]) -> f32 {
        let other_matrix =
            utils::calc_correlation_matrix(&columns.iter().map(|v| v.as_ref()).collect::<Vec<_>>());

        let v = 90;
        dbg!(other_matrix[v], self.correlation_matrix[v]);

        let sum = self
            .correlation_matrix
            .iter()
            .zip(other_matrix.iter())
            .map(|(a, b)| {
                if a.is_finite() && b.is_finite() {
                    ((a - b) / 2.0).abs()
                } else {
                    1.0
                }
            })
            .sum::<f32>();

        sum
    }
}
