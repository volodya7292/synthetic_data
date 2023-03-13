use std::collections::HashMap;

/// Returns map of counts for each unique element.
pub(crate) fn calc_discrete_pdf(uniques: &[i32], data: &[i32]) -> Vec<usize> {
    let mut counts: HashMap<i32, usize> = uniques.iter().map(|v| (*v, 0)).collect();

    for v in data {
        *counts.get_mut(v).unwrap() += 1;
    }

    let mut counts: Vec<_> = counts.into_iter().collect();
    // Because we will return only counts, sort unique items
    // to have universal order when comparing pdfs with the same unique items
    counts.sort_unstable_by_key(|v| v.0);

    counts.into_iter().map(|v| v.1).collect()
}

/// Returns map of counts for each unique element.
pub(crate) fn calc_continuous_pdf(min: f32, max: f32, data: &[f32]) -> Vec<usize> {
    let n_buckets = (data.len() as f32).sqrt().ceil().max(1.0) as usize;
    let bucket_size = (max - min) / n_buckets as f32;

    let mut buckets = vec![0_usize; n_buckets as usize];

    for v in data {
        let bucket_idx = (((v - min) / bucket_size) as usize).min(n_buckets - 1);
        buckets[bucket_idx] += 1;
    }

    buckets
}

/// The distance is in range [0, 1] where 0 means pdfs are equal and 1 means
/// maximum difference between pdfs.
pub(crate) fn l1_distance_between_pdfs(pdf1: &[usize], pdf2: &[usize]) -> f32 {
    assert_eq!(pdf1.len(), pdf2.len());

    let n_buckets = pdf1.len();
    let count1 = pdf1.iter().sum::<usize>() as f32;
    let count2 = pdf2.iter().sum::<usize>() as f32;

    let pdf1_norm: Vec<_> = pdf1.iter().map(|v| *v as f32 / count1 as f32).collect();
    let pdf2_norm: Vec<_> = pdf2.iter().map(|v| *v as f32 / count2 as f32).collect();

    let diff_sum = pdf1_norm
        .iter()
        .zip(&pdf2_norm)
        .fold(0.0, |accum, (v1, v2)| accum + (v1 - v2).abs());

    diff_sum / n_buckets as f32
}
