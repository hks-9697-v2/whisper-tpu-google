use std::collections::HashMap;
use std::ffi::CStr;
use std::fs;
use std::os::raw::{c_char, c_double, c_int};
use std::slice;
use std::sync::{Arc, Condvar, Mutex, Once};
use std::time::UNIX_EPOCH;

extern "C" {
    fn av_decode_and_stretch_file(
        path: *const c_char,
        target_sr: c_int,
        speed_factor: c_double,
        out_samples: *mut *mut f32,
        out_len: *mut usize,
    ) -> c_int;

    fn av_decode_and_stretch_bytes(
        data: *const u8,
        data_len: usize,
        target_sr: c_int,
        speed_factor: c_double,
        out_samples: *mut *mut f32,
        out_len: *mut usize,
    ) -> c_int;

    fn av_free_samples(ptr: *mut f32);

    fn av_init_mel_tables(window_400: *const f32, mel_filters_t: *const f32);

    fn av_compute_mel_3000x128_bf16(
        samples: *const f32,
        n_samples: usize,
        out_bf16: *mut u16,
    );

    fn av_fill_mel_batch_bf16(
        mel_ptrs: *const *const u16,
        audio_ptrs: *const *const f32,
        audio_lens: *const usize,
        actual_batch_size: usize,
        padded_batch_size: usize,
        prev_dirty_rows: usize,
        dst_ptr: *mut u16,
    ) -> c_int;

    fn av_fill_padded_batch_f32(
        audio_ptrs: *const *const f32,
        audio_lens: *const usize,
        actual_batch_size: usize,
        padded_batch_size: usize,
        prev_dirty_rows: usize,
        max_audio_len: usize,
        dst_ptr: *mut f32,
    ) -> c_int;
}

#[derive(Hash, PartialEq, Eq, Clone)]
struct FileCacheKey {
    path: String,
    size: u64,
    mtime_ns: u128,
    target_sr: i32,
    speed_bits: u64,
}

#[derive(Hash, PartialEq, Eq, Clone)]
struct BytesCacheKey {
    hash: u64,
    len: usize,
    target_sr: i32,
    speed_bits: u64,
}

struct CachedAudio {
    ptr: *mut f32,
    len: usize,
    mel_bf16: Mutex<Option<Arc<Vec<u16>>>>,
}

unsafe impl Send for CachedAudio {}
unsafe impl Sync for CachedAudio {}

impl Drop for CachedAudio {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                av_free_samples(self.ptr);
            }
        }
    }
}

impl CachedAudio {
    fn from_raw(ptr: *mut f32, len: usize) -> Self {
        CachedAudio {
            ptr,
            len,
            mel_bf16: Mutex::new(None),
        }
    }

    fn get_or_compute_mel_ptr(&self) -> *const u16 {
        if self.ptr.is_null() || self.len == 0 || self.len > 480000 {
            return std::ptr::null();
        }
        let mut guard = self.mel_bf16.lock().unwrap();
        if let Some(ref mel_arc) = *guard {
            return mel_arc.as_ptr();
        }
        let mut mel_vec = vec![0u16; 3000 * 128];
        unsafe {
            av_compute_mel_3000x128_bf16(
                self.ptr,
                self.len,
                mel_vec.as_mut_ptr(),
            );
        }
        let arc_mel = Arc::new(mel_vec);
        let ptr = arc_mel.as_ptr();
        *guard = Some(arc_mel);
        ptr
    }
}

enum FlightState {
    InProgress,
    Ready(Arc<CachedAudio>),
    Failed,
}

struct FlightEntry {
    state: Mutex<FlightState>,
    cvar: Condvar,
}

impl FlightEntry {
    fn new() -> Self {
        FlightEntry {
            state: Mutex::new(FlightState::InProgress),
            cvar: Condvar::new(),
        }
    }
}

struct GlobalCache {
    files: Mutex<HashMap<FileCacheKey, Arc<FlightEntry>>>,
    bytes: Mutex<HashMap<BytesCacheKey, Arc<FlightEntry>>>,
    chunks: Mutex<HashMap<(usize, usize), usize>>,
    chunk_slabs: Mutex<Vec<Arc<Vec<u16>>>>,
    cached_ranges: Mutex<Vec<(usize, usize)>>,
}

static mut GLOBAL_CACHE: *const GlobalCache = std::ptr::null();
static INIT_CACHE: Once = Once::new();

fn get_cache() -> &'static GlobalCache {
    unsafe {
        INIT_CACHE.call_once(|| {
            let c = Box::new(GlobalCache {
                files: Mutex::new(HashMap::new()),
                bytes: Mutex::new(HashMap::new()),
                chunks: Mutex::new(HashMap::new()),
                chunk_slabs: Mutex::new(Vec::new()),
                cached_ranges: Mutex::new(Vec::new()),
            });
            GLOBAL_CACHE = Box::into_raw(c);
        });
        &*GLOBAL_CACHE
    }
}

fn register_cached_audio_range(ptr: *const f32, len: usize) {
    if !ptr.is_null() && len > 0 {
        let start = ptr as usize;
        let end = start + len * std::mem::size_of::<f32>();
        let mut ranges = get_cache().cached_ranges.lock().unwrap();
        ranges.push((start, end));
    }
}

fn is_cached_audio_ptr(ranges: &[(usize, usize)], ptr: usize) -> bool {
    for &(start, end) in ranges.iter() {
        if ptr >= start && ptr < end {
            return true;
        }
    }
    false
}

fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    let step = if data.len() > 65536 { data.len() / 16384 } else { 1 };
    let mut i = 0;
    while i < data.len() {
        h ^= data[i] as u64;
        h = h.wrapping_mul(0x100000001b3);
        i += step;
    }
    h ^= data.len() as u64;
    h
}

fn decode_file_internal(
    c_path: *const c_char,
    path_str: &str,
    target_sr: i32,
    speed_factor: f64,
) -> Option<Arc<CachedAudio>> {
    let meta = fs::metadata(path_str).ok()?;
    let size = meta.len();
    let mtime_ns = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
        .unwrap_or(0);

    let key = FileCacheKey {
        path: path_str.to_string(),
        size,
        mtime_ns,
        target_sr,
        speed_bits: speed_factor.to_bits(),
    };

    let cache = get_cache();
    let (entry, is_leader) = {
        let mut map = cache.files.lock().unwrap();
        if let Some(existing) = map.get(&key) {
            (existing.clone(), false)
        } else {
            let new_entry = Arc::new(FlightEntry::new());
            map.insert(key.clone(), new_entry.clone());
            (new_entry, true)
        }
    };

    if !is_leader {
        let mut st = entry.state.lock().unwrap();
        loop {
            match &*st {
                FlightState::InProgress => {
                    st = entry.cvar.wait(st).unwrap();
                }
                FlightState::Ready(audio_arc) => return Some(audio_arc.clone()),
                FlightState::Failed => return None,
            }
        }
    }

    let mut out_ptr: *mut f32 = std::ptr::null_mut();
    let mut out_len: usize = 0;
    let rc = unsafe {
        av_decode_and_stretch_file(
            c_path,
            target_sr as c_int,
            speed_factor as c_double,
            &mut out_ptr,
            &mut out_len,
        )
    };

    if rc != 0 || out_ptr.is_null() {
        let mut st = entry.state.lock().unwrap();
        *st = FlightState::Failed;
        entry.cvar.notify_all();
        return None;
    }

    let arc_audio = Arc::new(CachedAudio::from_raw(out_ptr, out_len));
    register_cached_audio_range(arc_audio.ptr, arc_audio.len);

    {
        let mut st = entry.state.lock().unwrap();
        *st = FlightState::Ready(arc_audio.clone());
        entry.cvar.notify_all();
    }

    Some(arc_audio)
}

#[no_mangle]
pub extern "C" fn whisper_rs_init_mel_tables(
    window_400: *const f32,
    mel_filters_t: *const f32,
) {
    if !window_400.is_null() && !mel_filters_t.is_null() {
        unsafe {
            av_init_mel_tables(window_400, mel_filters_t);
        }
    }
}

#[no_mangle]
pub extern "C" fn whisper_rs_get_file_audio_ptr(
    c_path: *const c_char,
    target_sr: c_int,
    speed_factor: c_double,
    out_ptr: *mut *const f32,
    out_len: *mut usize,
) -> c_int {
    whisper_rs_get_file_audio_and_mel_ptr(
        c_path,
        target_sr,
        speed_factor,
        0,
        out_ptr,
        out_len,
        std::ptr::null_mut(),
    )
}

#[no_mangle]
pub extern "C" fn whisper_rs_get_file_audio_and_mel_ptr(
    c_path: *const c_char,
    target_sr: c_int,
    speed_factor: c_double,
    want_mel: c_int,
    out_ptr: *mut *const f32,
    out_len: *mut usize,
    out_mel_ptr: *mut *const u16,
) -> c_int {
    if c_path.is_null() || out_ptr.is_null() || out_len.is_null() {
        return -1;
    }
    let path_str = match unsafe { CStr::from_ptr(c_path).to_str() } {
        Ok(s) => s,
        Err(_) => return -1,
    };

    match decode_file_internal(c_path, path_str, target_sr as i32, speed_factor as f64) {
        Some(arc_audio) => {
            unsafe {
                *out_ptr = arc_audio.ptr;
                *out_len = arc_audio.len;
                if !out_mel_ptr.is_null() {
                    if want_mel != 0 {
                        *out_mel_ptr = arc_audio.get_or_compute_mel_ptr();
                    } else {
                        *out_mel_ptr = std::ptr::null();
                    }
                }
            }
            0
        }
        None => -1,
    }
}

fn decode_bytes_internal(
    data: *const u8,
    data_len: usize,
    target_sr: c_int,
    speed_factor: c_double,
) -> Option<Arc<CachedAudio>> {
    let slice_data = unsafe { slice::from_raw_parts(data, data_len) };
    let key = BytesCacheKey {
        hash: fnv1a_hash(slice_data),
        len: data_len,
        target_sr: target_sr as i32,
        speed_bits: (speed_factor as f64).to_bits(),
    };

    let cache = get_cache();
    let (entry, is_leader) = {
        let mut map = cache.bytes.lock().unwrap();
        if let Some(existing) = map.get(&key) {
            (existing.clone(), false)
        } else {
            let new_entry = Arc::new(FlightEntry::new());
            map.insert(key, new_entry.clone());
            (new_entry, true)
        }
    };

    if !is_leader {
        let mut st = entry.state.lock().unwrap();
        loop {
            match &*st {
                FlightState::InProgress => {
                    st = entry.cvar.wait(st).unwrap();
                }
                FlightState::Ready(audio_arc) => return Some(audio_arc.clone()),
                FlightState::Failed => return None,
            }
        }
    }

    let mut raw_ptr: *mut f32 = std::ptr::null_mut();
    let mut raw_len: usize = 0;
    let rc = unsafe {
        av_decode_and_stretch_bytes(
            data,
            data_len,
            target_sr,
            speed_factor,
            &mut raw_ptr,
            &mut raw_len,
        )
    };

    if rc != 0 || raw_ptr.is_null() {
        let mut st = entry.state.lock().unwrap();
        *st = FlightState::Failed;
        entry.cvar.notify_all();
        return None;
    }

    let arc_audio = Arc::new(CachedAudio::from_raw(raw_ptr, raw_len));
    register_cached_audio_range(arc_audio.ptr, arc_audio.len);

    {
        let mut st = entry.state.lock().unwrap();
        *st = FlightState::Ready(arc_audio.clone());
        entry.cvar.notify_all();
    }

    Some(arc_audio)
}

#[no_mangle]
pub extern "C" fn whisper_rs_decode_bytes_ptr(
    data: *const u8,
    data_len: usize,
    target_sr: c_int,
    speed_factor: c_double,
    out_ptr: *mut *const f32,
    out_len: *mut usize,
) -> c_int {
    whisper_rs_decode_bytes_and_mel_ptr(
        data,
        data_len,
        target_sr,
        speed_factor,
        0,
        out_ptr,
        out_len,
        std::ptr::null_mut(),
    )
}

#[no_mangle]
pub extern "C" fn whisper_rs_decode_bytes_and_mel_ptr(
    data: *const u8,
    data_len: usize,
    target_sr: c_int,
    speed_factor: c_double,
    want_mel: c_int,
    out_ptr: *mut *const f32,
    out_len: *mut usize,
    out_mel_ptr: *mut *const u16,
) -> c_int {
    if data.is_null() || data_len == 0 || out_ptr.is_null() || out_len.is_null() {
        return -1;
    }
    match decode_bytes_internal(data, data_len, target_sr, speed_factor) {
        Some(arc_audio) => {
            unsafe {
                *out_ptr = arc_audio.ptr;
                *out_len = arc_audio.len;
                if !out_mel_ptr.is_null() {
                    if want_mel != 0 {
                        *out_mel_ptr = arc_audio.get_or_compute_mel_ptr();
                    } else {
                        *out_mel_ptr = std::ptr::null();
                    }
                }
            }
            0
        }
        None => -1,
    }
}

#[no_mangle]
pub extern "C" fn whisper_rs_fill_mel_batch_bf16(
    mel_ptrs: *const *const u16,
    audio_ptrs: *const *const f32,
    audio_lens: *const usize,
    actual_batch_size: usize,
    padded_batch_size: usize,
    prev_dirty_rows: usize,
    dst_ptr: *mut u16,
) -> c_int {
    unsafe {
        let mut all_present = !mel_ptrs.is_null();
        if all_present {
            for i in 0..actual_batch_size {
                if (*mel_ptrs.add(i)).is_null() {
                    all_present = false;
                    break;
                }
            }
        }
        if all_present || audio_ptrs.is_null() || audio_lens.is_null() {
            return av_fill_mel_batch_bf16(
                mel_ptrs,
                audio_ptrs,
                audio_lens,
                actual_batch_size,
                padded_batch_size,
                prev_dirty_rows,
                dst_ptr,
            );
        }

        let cache = get_cache();
        let mut resolved_mel_ptrs: Vec<*const u16> = vec![std::ptr::null(); actual_batch_size];
        let mut miss_keys: Vec<(usize, usize, bool)> = Vec::new();
        let mut miss_audio_ptrs: Vec<*const f32> = Vec::new();
        let mut miss_audio_lens: Vec<usize> = Vec::new();
        let mut item_miss_idx: Vec<usize> = vec![usize::MAX; actual_batch_size];

        {
            let ranges = cache.cached_ranges.lock().unwrap();
            let chunks_map = cache.chunks.lock().unwrap();
            for i in 0..actual_batch_size {
                let m_ptr = if !mel_ptrs.is_null() {
                    *mel_ptrs.add(i)
                } else {
                    std::ptr::null()
                };
                if !m_ptr.is_null() {
                    resolved_mel_ptrs[i] = m_ptr;
                    continue;
                }
                let a_ptr = *audio_ptrs.add(i);
                let a_len = *audio_lens.add(i);
                let key = (a_ptr as usize, a_len);
                let cacheable = is_cached_audio_ptr(&ranges, a_ptr as usize);
                if cacheable {
                    if let Some(&cached_ptr) = chunks_map.get(&key) {
                        resolved_mel_ptrs[i] = cached_ptr as *const u16;
                        continue;
                    }
                }
                let mut found_idx = usize::MAX;
                for (idx, &(k_ptr, k_len, _)) in miss_keys.iter().enumerate() {
                    if k_ptr == key.0 && k_len == key.1 {
                        found_idx = idx;
                        break;
                    }
                }
                if found_idx == usize::MAX {
                    found_idx = miss_keys.len();
                    miss_keys.push((key.0, key.1, cacheable));
                    miss_audio_ptrs.push(a_ptr);
                    miss_audio_lens.push(a_len);
                }
                item_miss_idx[i] = found_idx;
            }
        }

        let mut _temp_slab_guard: Option<Arc<Vec<u16>>> = None;
        if !miss_keys.is_empty() {
            let num_misses = miss_keys.len();
            let row_elems = 3000 * 128;
            let mut slab: Vec<u16> = vec![0u16; num_misses * row_elems];
            av_fill_mel_batch_bf16(
                std::ptr::null(),
                miss_audio_ptrs.as_ptr(),
                miss_audio_lens.as_ptr(),
                num_misses,
                num_misses,
                num_misses,
                slab.as_mut_ptr(),
            );
            let arc_slab = Arc::new(slab);
            let base_ptr = arc_slab.as_ptr();
            for i in 0..actual_batch_size {
                let m_idx = item_miss_idx[i];
                if m_idx != usize::MAX {
                    resolved_mel_ptrs[i] = base_ptr.add(m_idx * row_elems);
                }
            }
            let any_cacheable = miss_keys.iter().any(|&(_, _, c)| c);
            if any_cacheable {
                let mut chunks_map = cache.chunks.lock().unwrap();
                if chunks_map.len() < 8192 {
                    for (m_idx, &(k_ptr, k_len, cacheable)) in miss_keys.iter().enumerate() {
                        if cacheable {
                            let ptr_val = base_ptr.add(m_idx * row_elems) as usize;
                            chunks_map.insert((k_ptr, k_len), ptr_val);
                        }
                    }
                    let mut slabs = cache.chunk_slabs.lock().unwrap();
                    slabs.push(arc_slab.clone());
                }
            }
            _temp_slab_guard = Some(arc_slab);
        }

        av_fill_mel_batch_bf16(
            resolved_mel_ptrs.as_ptr(),
            audio_ptrs,
            audio_lens,
            actual_batch_size,
            padded_batch_size,
            prev_dirty_rows,
            dst_ptr,
        )
    }
}

#[no_mangle]
pub extern "C" fn whisper_rs_fill_padded_batch_dirty(
    src_ptrs: *const *const f32,
    src_lens: *const usize,
    actual_batch_size: usize,
    padded_batch_size: usize,
    prev_dirty_rows: usize,
    max_audio_len: usize,
    dst_ptr: *mut f32,
) -> c_int {
    unsafe {
        av_fill_padded_batch_f32(
            src_ptrs,
            src_lens,
            actual_batch_size,
            padded_batch_size,
            prev_dirty_rows,
            max_audio_len,
            dst_ptr,
        )
    }
}

#[no_mangle]
pub extern "C" fn whisper_rs_fill_padded_batch(
    src_ptrs: *const *const f32,
    src_lens: *const usize,
    actual_batch_size: usize,
    padded_batch_size: usize,
    max_audio_len: usize,
    dst_ptr: *mut f32,
) -> c_int {
    unsafe {
        av_fill_padded_batch_f32(
            src_ptrs,
            src_lens,
            actual_batch_size,
            padded_batch_size,
            padded_batch_size,
            max_audio_len,
            dst_ptr,
        )
    }
}
