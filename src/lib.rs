#[cfg(test)]
mod tests;
mod tvae;

use crate::tvae::input::{ColumnData, ColumnDataRef};
use crate::tvae::TVAE;
use index_pool::IndexPool;
use lazy_static::lazy_static;
use std::collections::HashMap;
use std::ffi::c_void;
use std::slice;
use std::sync::{Arc, Mutex};

#[derive(Debug)]
#[repr(u32)]
pub enum RawColumnType {
    Continuous = 0,
    Discrete = 1,
}

#[derive(Debug)]
#[repr(C)]
pub struct RawColumnData {
    r#type: RawColumnType,
    /// # Pointer type:
    /// `Continuous`: `*const f32`,
    /// `Category`: `*const i32`,
    data: *const c_void,
}

/// Returns whether to stop the learning loop.
pub type FlowControlCallback = extern "C" fn(epoch: usize, loss: f64) -> bool;

#[derive(Debug, Clone)]
#[repr(C)]
pub struct TrainParams {
    batch_size: usize,
    flow_control_callback: FlowControlCallback,
}

pub type SynthNetHandle = usize;

lazy_static! {
    static ref NET_HANDLES: Mutex<IndexPool> = Default::default();
    static ref NET_STORAGE: Mutex<HashMap<SynthNetHandle, Arc<Mutex<TVAE>>>> = Default::default();
}

/// Creates and trains a new NN.
#[no_mangle]
pub unsafe extern "C" fn synth_net_fit(
    columns: *const RawColumnData,
    n_columns: usize,
    n_rows: usize,
    train_params: &TrainParams,
) -> SynthNetHandle {
    let handle = NET_HANDLES.lock().unwrap().new_id();

    let column_data: Vec<_> = (0..n_columns)
        .map(|i| {
            let raw_col_data = &*columns.add(i);

            match raw_col_data.r#type {
                RawColumnType::Continuous => ColumnDataRef::Continuous(slice::from_raw_parts(
                    raw_col_data.data as *const f32,
                    n_rows,
                )),
                RawColumnType::Discrete => ColumnDataRef::Discrete(slice::from_raw_parts(
                    raw_col_data.data as *const i32,
                    n_rows,
                )),
            }
        })
        .collect();

    let net = TVAE::fit(
        &column_data,
        train_params.batch_size,
        tch::Device::Cpu,
        |epoch, loss| (train_params.flow_control_callback)(epoch, loss),
    );
    NET_STORAGE
        .lock()
        .unwrap()
        .insert(handle, Arc::new(Mutex::new(net)));

    handle
}

/// Generates synthetic data using the specified NN.
/// `n_samples`: number of rows to sample.
///
/// # Safety:
/// The number of elements in `columns` array must be the same
/// as the size of `columns` passed to `synth_net_fit`.
#[no_mangle]
pub unsafe extern "C" fn synth_net_sample(
    net_handle: SynthNetHandle,
    columns: *const *mut c_void,
    n_samples: usize,
) {
    let net_storage = NET_STORAGE.lock().unwrap();
    let net = net_storage
        .get(&net_handle)
        .expect("synth_net_sample: net_handle must be valid")
        .lock()
        .unwrap();

    let n_columns = net.n_columns();
    let data = net.sample(n_samples);

    for (col_data, raw_col_data) in data.iter().zip(slice::from_raw_parts(columns, n_columns)) {
        match col_data {
            ColumnData::Discrete(data) => data
                .as_ptr()
                .copy_to_nonoverlapping(*raw_col_data as *mut i32, data.len()),
            ColumnData::Continuous(data) => data
                .as_ptr()
                .copy_to_nonoverlapping(*raw_col_data as *mut f32, data.len()),
        }
    }
}

/// Destroys the specified NN.
#[no_mangle]
pub unsafe extern "C" fn synth_net_destroy(handle: SynthNetHandle) {
    NET_HANDLES
        .lock()
        .unwrap()
        .return_id(handle)
        .expect("destroy_net: the network handle must be valid");
}
