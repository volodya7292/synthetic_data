pub mod tvae;
mod utils;

use crate::tvae::input::{ColumnDataRef, SampledColumnData};
use crate::tvae::TVAE;
use index_pool::IndexPool;
use lazy_static::lazy_static;
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::slice;
use std::sync::{Arc, Mutex};

pub type SynthNetHandle = usize;

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

#[derive(Debug)]
#[repr(C)]
pub struct SynthNetSnapshot {
    c_str: *const c_char,
}

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
///
/// `n_samples`: number of rows to sample.
/// `out_data`: a pointer to resulting column data in
/// column-major order (column1_data, column2_data, ...)
/// `out_realness`: a pointer to an array of realness metrics (in range \[0;1\]) for each column.
/// `out_correlation_realness`: overall column correlation realness
///
/// # Safety
/// The number of elements in `columns` array must be the same
/// as the size of `columns` passed to `synth_net_fit`.
#[no_mangle]
pub unsafe extern "C" fn synth_net_sample(
    net_handle: SynthNetHandle,
    out_data: *mut c_void,
    out_realness: *mut f32,
    out_correlation_realness: *mut f32,
    n_samples: usize,
) {
    let net_storage = NET_STORAGE.lock().unwrap();
    let net = net_storage
        .get(&net_handle)
        .expect("synth_net_sample: net_handle must be valid")
        .lock()
        .unwrap();

    let (data, corr_realness) = net.sample(n_samples);

    let mut curr_out_ptr = out_data;
    let mut curr_out_realness_ptr = out_realness;

    for col_data in data {
        match &col_data {
            SampledColumnData::Discrete(data, ..) => data
                .as_ptr()
                .copy_to_nonoverlapping(curr_out_ptr as *mut i32, data.len()),
            SampledColumnData::Continuous(data, ..) => data
                .as_ptr()
                .copy_to_nonoverlapping(curr_out_ptr as *mut f32, data.len()),
        }

        curr_out_realness_ptr.write(col_data.realness());

        curr_out_ptr = curr_out_ptr.add(col_data.element_size() * n_samples);
        curr_out_realness_ptr = curr_out_realness_ptr.add(1);
    }

    out_correlation_realness.write(corr_realness)
}

/// Saves network state into a snapshot.
#[no_mangle]
pub unsafe extern "C" fn synth_net_create_snapshot(net_handle: SynthNetHandle) -> SynthNetSnapshot {
    let net_storage = NET_STORAGE.lock().unwrap();
    let net = net_storage
        .get(&net_handle)
        .expect("synth_net_sample: net_handle must be valid")
        .lock()
        .unwrap();

    let val = net.save();
    let val_str = serde_json::to_string(&val).unwrap();
    let val_c_str = CString::new(val_str).unwrap();

    SynthNetSnapshot {
        c_str: val_c_str.into_raw(),
    }
}

/// Destroys the specified snapshot.
#[no_mangle]
pub unsafe extern "C" fn synth_net_create_from_snapshot(
    snapshot: &SynthNetSnapshot,
) -> SynthNetHandle {
    let handle = NET_HANDLES.lock().unwrap().new_id();

    let data_str = CStr::from_ptr(snapshot.c_str).to_str().unwrap();
    let net = TVAE::load(tch::Device::Cpu, serde_json::from_str(data_str).unwrap());

    NET_STORAGE
        .lock()
        .unwrap()
        .insert(handle, Arc::new(Mutex::new(net)));

    handle
}

/// Destroys the specified snapshot.
#[no_mangle]
pub unsafe extern "C" fn synth_net_snapshot_destroy(snapshot: &SynthNetSnapshot) {
    drop(CString::from_raw(snapshot.c_str as *mut _));
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
