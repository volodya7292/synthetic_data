use crate::tvae::input::ColumnDataRef;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tch::Tensor;

#[derive(Serialize, Deserialize, Clone)]
pub struct Pdf {
    buckets: Vec<usize>,
    sum: usize,
}

impl Pdf {
    pub fn new(buckets: Vec<usize>) -> Self {
        let sum = buckets.iter().sum::<usize>();
        Self { buckets, sum }
    }

    pub fn cached_sum(&self) -> usize {
        self.sum
    }

    pub fn buckets(&self) -> &[usize] {
        &self.buckets
    }

    pub fn l1_distance(&self, other: &Pdf) -> f32 {
        l1_distance_between_pdfs(&self.buckets, &other.buckets)
    }

    pub fn add(&mut self, bucket_idx: usize, count: usize) {
        self.buckets[bucket_idx] += count;
    }

    pub fn remove(&mut self, bucket_idx: usize, count: usize) -> Result<(), ()> {
        self.buckets[bucket_idx] = self.buckets[bucket_idx].checked_sub(count).ok_or(())?;
        Ok(())
    }
}

pub fn calc_change_influence(target: &Pdf, curr: &Pdf, add_idx: usize, remove_idx: usize) -> f32 {
    let curr_count_in_add = curr.buckets[add_idx];
    let curr_count_in_remove = curr.buckets[remove_idx];

    let target_count_in_add = target.buckets[add_idx];
    let target_count_in_remove = target.buckets[remove_idx];

    let importance = target.l1_distance(curr);
    let add_influence = (target_count_in_add as f32 - curr_count_in_add as f32) / target.cached_sum() as f32;
    let remove_incluence = (curr_count_in_remove as f32 - target_count_in_remove as f32) / target.cached_sum() as f32;
    (add_influence + remove_incluence) * importance
}

/// Returns map of counts for each unique element.
pub(crate) fn calc_discrete_pdf(uniques: &[i32], data: &[i32]) -> Pdf {
    let mut counts: HashMap<i32, usize> = uniques.iter().map(|v| (*v, 0)).collect();

    for v in data {
        *counts.get_mut(v).unwrap() += 1;
    }

    let mut counts: Vec<_> = counts.into_iter().collect();
    // Because we will return only counts, sort unique items
    // to have universal order when comparing pdfs with the same unique items
    counts.sort_unstable_by_key(|v| v.0);

    let values = counts.into_iter().map(|v| v.1).collect();
    Pdf::new(values)
}

/// Returns map of counts for each unique element.
pub(crate) fn calc_continuous_pdf(min: f32, max: f32, data: &[f32], n_buckets: usize) -> Pdf {
    let bucket_size = (max - min) / n_buckets as f32;
    let mut buckets = vec![0_usize; n_buckets];

    for v in data {
        let bucket_idx = ((v - min) / bucket_size).clamp(0.0, n_buckets as f32 - 1.0) as usize;
        buckets[bucket_idx] += 1;
    }

    Pdf::new(buckets)
}

/// The distance is in range [0, 1] where 0 means pdfs are equal and 1 means
/// maximum difference between pdfs.
pub(crate) fn l1_distance_between_pdfs(pdf1: &[usize], pdf2: &[usize]) -> f32 {
    assert_eq!(pdf1.len(), pdf2.len());

    let count1 = pdf1.iter().sum::<usize>() as f32;
    let count2 = pdf2.iter().sum::<usize>() as f32;

    let pdf1_norm: Vec<_> = pdf1.iter().map(|v| *v as f32 / count1).collect();
    let pdf2_norm: Vec<_> = pdf2.iter().map(|v| *v as f32 / count2).collect();

    let diff_sum = pdf1_norm
        .iter()
        .zip(&pdf2_norm)
        .fold(0.0, |accum, (v1, v2)| accum + (v1 - v2).abs());

    0.5 * diff_sum
}

/// Calculates correlation matrix for given columns. Returned data is in row-major order.
pub(crate) fn calc_correlation_matrix(data: &[ColumnDataRef]) -> Vec<f32> {
    let variables: Vec<_> = data
        .iter()
        .map(|col| match col {
            ColumnDataRef::Discrete(data) => Tensor::from_slice(data),
            ColumnDataRef::Continuous(data) => Tensor::from_slice(data),
        })
        .collect();

    let tensor = Tensor::stack(&variables, 0);
    let corr_mat = tensor.corrcoef();
    let flattened_mat = if data.len() > 1 {
        corr_mat.flatten(0, 1)
    } else {
        corr_mat.reshape([1])
    };

    let mut data = vec![0.0_f32; flattened_mat.size1().unwrap() as usize];
    for (out_v, in_v) in data.iter_mut().zip(flattened_mat.iter::<f64>().unwrap()) {
        *out_v = in_v as f32;
    }
    data
}
