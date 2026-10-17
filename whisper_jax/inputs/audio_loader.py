import ctypes
import os
import subprocess
import librosa
import jax.numpy as jnp
import numpy as np

_RUST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rust_audio")
_SO_PATH = os.path.join(_RUST_DIR, "libwhisper_audio_rs.so")
_LIB = None

_RING_SIZE = 4
_MEL_RING_BUFFERS = {}
_MEL_RING_IDX = {}
_AUDIO_RING_BUFFERS = {}
_AUDIO_RING_IDX = {}


def _ensure_built():
    c_src = os.path.join(_RUST_DIR, "av_shim.c")
    rs_src = os.path.join(_RUST_DIR, "src", "lib.rs")
    need_build = not os.path.exists(_SO_PATH)
    if not need_build:
        so_mtime = os.path.getmtime(_SO_PATH)
        if os.path.getmtime(c_src) > so_mtime or os.path.getmtime(rs_src) > so_mtime:
            need_build = True

    if need_build:
        obj_path = os.path.join(_RUST_DIR, "av_shim.o")
        a_path = os.path.join(_RUST_DIR, "libav_shim.a")
        subprocess.check_call(
            ["gcc", "-O3", "-fPIC", "-march=native", "-ffast-math", "-c", c_src, "-o", obj_path],
            cwd=_RUST_DIR,
        )
        subprocess.check_call(["ar", "rcs", a_path, obj_path], cwd=_RUST_DIR)
        subprocess.check_call(
            [
                "rustc",
                "-C",
                "opt-level=3",
                "--crate-type",
                "cdylib",
                rs_src,
                "-L",
                _RUST_DIR,
                "-l",
                "static=av_shim",
                "-l",
                "avfilter",
                "-l",
                "avformat",
                "-l",
                "avcodec",
                "-l",
                "avutil",
                "-l",
                "swresample",
                "-l",
                "pthread",
                "-l",
                "m",
                "-o",
                _SO_PATH,
            ],
            cwd=_RUST_DIR,
        )


def get_rust_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    _ensure_built()
    lib = ctypes.CDLL(_SO_PATH)

    lib.whisper_rs_init_mel_tables.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.whisper_rs_init_mel_tables.restype = None

    lib.whisper_rs_get_file_audio_ptr.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.whisper_rs_get_file_audio_ptr.restype = ctypes.c_int

    lib.whisper_rs_get_file_audio_and_mel_ptr.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.whisper_rs_get_file_audio_and_mel_ptr.restype = ctypes.c_int

    lib.whisper_rs_decode_bytes_ptr.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.whisper_rs_decode_bytes_ptr.restype = ctypes.c_int

    lib.whisper_rs_decode_bytes_and_mel_ptr.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.whisper_rs_decode_bytes_and_mel_ptr.restype = ctypes.c_int

    lib.whisper_rs_fill_mel_batch_bf16.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    lib.whisper_rs_fill_mel_batch_bf16.restype = ctypes.c_int

    lib.whisper_rs_fill_padded_batch.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    lib.whisper_rs_fill_padded_batch.restype = ctypes.c_int

    lib.whisper_rs_fill_padded_batch_dirty.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    lib.whisper_rs_fill_padded_batch_dirty.restype = ctypes.c_int

    win_400 = np.ascontiguousarray(np.hanning(400), dtype=np.float32)
    mel_t = np.ascontiguousarray(librosa.filters.mel(sr=16000, n_fft=400, n_mels=128).T, dtype=np.float32)
    lib.whisper_rs_init_mel_tables(
        win_400.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        mel_t.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    _LIB = lib
    return _LIB


def preallocate_batch_buffers_rs(batch_buckets, max_audio_length: int = 480000, use_mel_bf16: bool = True):
    for b in batch_buckets:
        b = int(b)
        if use_mel_bf16:
            if b not in _MEL_RING_BUFFERS:
                _MEL_RING_BUFFERS[b] = [
                    [np.zeros((b, 3000, 128), dtype=np.uint16).view(jnp.bfloat16), 0]
                    for _ in range(_RING_SIZE)
                ]
                _MEL_RING_IDX[b] = 0
        else:
            key = (b, int(max_audio_length))
            if key not in _AUDIO_RING_BUFFERS:
                _AUDIO_RING_BUFFERS[key] = [
                    [np.zeros((b, int(max_audio_length)), dtype=np.float32), 0]
                    for _ in range(_RING_SIZE)
                ]
                _AUDIO_RING_IDX[key] = 0


def decode_audio_file_rs(file_path: str, sampling_rate: int = 16000, speed_factor: float = 1.0) -> np.ndarray:
    """
    Decodes and time-stretches an audio file in native Rust + in-process libavfilter,
    returning a zero-copy NumPy float32 view backed by the Rust cache.
    """
    lib = get_rust_lib()
    out_ptr = ctypes.POINTER(ctypes.c_float)()
    out_len = ctypes.c_size_t(0)
    b_path = file_path.encode("utf-8") if isinstance(file_path, str) else file_path
    rc = lib.whisper_rs_get_file_audio_ptr(
        b_path,
        int(sampling_rate),
        float(speed_factor),
        ctypes.byref(out_ptr),
        ctypes.byref(out_len),
    )
    if rc != 0 or out_len.value == 0:
        return np.empty((0,), dtype=np.float32)
    arr = np.ctypeslib.as_array(out_ptr, shape=(out_len.value,))
    return arr


def decode_audio_file_with_mel_rs(file_path: str, sampling_rate: int = 16000, speed_factor: float = 1.0):
    """
    Decodes an audio file and (if single-chunk <= 480,000 samples) also computes/caches its
    (3000, 128) bfloat16 Mel-spectrogram in C/Rust.
    Returns (waveform_np, mel_ptr_int).
    """
    lib = get_rust_lib()
    out_ptr = ctypes.POINTER(ctypes.c_float)()
    out_len = ctypes.c_size_t(0)
    out_mel_ptr = ctypes.c_void_p(0)
    b_path = file_path.encode("utf-8") if isinstance(file_path, str) else file_path
    rc = lib.whisper_rs_get_file_audio_and_mel_ptr(
        b_path,
        int(sampling_rate),
        float(speed_factor),
        1,
        ctypes.byref(out_ptr),
        ctypes.byref(out_len),
        ctypes.byref(out_mel_ptr),
    )
    if rc != 0 or out_len.value == 0:
        return np.empty((0,), dtype=np.float32), 0
    arr = np.ctypeslib.as_array(out_ptr, shape=(out_len.value,))
    return arr, (out_mel_ptr.value or 0)


def decode_audio_bytes_rs(audio_bytes: bytes, sampling_rate: int = 16000, speed_factor: float = 1.0) -> np.ndarray:
    """
    Decodes and time-stretches in-memory audio bytes in native Rust + in-process libavfilter,
    returning a zero-copy NumPy float32 view backed by the Rust cache.
    """
    lib = get_rust_lib()
    out_ptr = ctypes.POINTER(ctypes.c_float)()
    out_len = ctypes.c_size_t(0)
    rc = lib.whisper_rs_decode_bytes_ptr(
        audio_bytes,
        len(audio_bytes),
        int(sampling_rate),
        float(speed_factor),
        ctypes.byref(out_ptr),
        ctypes.byref(out_len),
    )
    if rc != 0 or out_len.value == 0:
        return np.empty((0,), dtype=np.float32)
    arr = np.ctypeslib.as_array(out_ptr, shape=(out_len.value,))
    return arr


def decode_audio_bytes_with_mel_rs(audio_bytes: bytes, sampling_rate: int = 16000, speed_factor: float = 1.0):
    lib = get_rust_lib()
    out_ptr = ctypes.POINTER(ctypes.c_float)()
    out_len = ctypes.c_size_t(0)
    out_mel_ptr = ctypes.c_void_p(0)
    rc = lib.whisper_rs_decode_bytes_and_mel_ptr(
        audio_bytes,
        len(audio_bytes),
        int(sampling_rate),
        float(speed_factor),
        1,
        ctypes.byref(out_ptr),
        ctypes.byref(out_len),
        ctypes.byref(out_mel_ptr),
    )
    if rc != 0 or out_len.value == 0:
        return np.empty((0,), dtype=np.float32), 0
    arr = np.ctypeslib.as_array(out_ptr, shape=(out_len.value,))
    return arr, (out_mel_ptr.value or 0)


def fill_mel_batch_bf16_rs(mel_ptr_ints, audio_chunks, padded_batch_size: int) -> np.ndarray:
    """
    Assembles a batch of (padded_batch_size, 3000, 128) bfloat16 Mel-spectrograms into a
    pre-allocated ring buffer using the persistent C pthread pool.
    """
    lib = get_rust_lib()
    actual_size = len(mel_ptr_ints)
    ring = _MEL_RING_BUFFERS.get(padded_batch_size)
    if ring is None:
        preallocate_batch_buffers_rs([padded_batch_size], use_mel_bf16=True)
        ring = _MEL_RING_BUFFERS[padded_batch_size]
    idx = _MEL_RING_IDX[padded_batch_size]
    _MEL_RING_IDX[padded_batch_size] = (idx + 1) & (_RING_SIZE - 1)
    slot = ring[idx]
    out_buffer, prev_dirty_rows = slot[0], slot[1]

    PtrArray = ctypes.c_void_p * actual_size
    mel_ptrs = PtrArray(*mel_ptr_ints)

    if audio_chunks is not None:
        audio_ptrs = PtrArray(*(c.ctypes.data for c in audio_chunks))
        LenArray = ctypes.c_size_t * actual_size
        audio_lens = LenArray(*(c.shape[0] for c in audio_chunks))
    else:
        audio_ptrs = None
        audio_lens = None

    lib.whisper_rs_fill_mel_batch_bf16(
        mel_ptrs,
        audio_ptrs,
        audio_lens,
        actual_size,
        padded_batch_size,
        prev_dirty_rows,
        out_buffer.ctypes.data,
    )
    slot[1] = actual_size
    return out_buffer


def fill_padded_batch_rs(audio_chunks, padded_batch_size: int, max_audio_length: int, out_buffer: np.ndarray = None) -> np.ndarray:
    """
    Assembles and zero-pads a batch of audio chunks into a contiguous (padded_batch_size, max_audio_length)
    float32 array in parallel using the persistent C worker pool and pre-allocated ring buffers.
    """
    lib = get_rust_lib()
    actual_size = len(audio_chunks)
    if out_buffer is None:
        key = (padded_batch_size, max_audio_length)
        ring = _AUDIO_RING_BUFFERS.get(key)
        if ring is None:
            preallocate_batch_buffers_rs([padded_batch_size], max_audio_length, use_mel_bf16=False)
            ring = _AUDIO_RING_BUFFERS[key]
        idx = _AUDIO_RING_IDX[key]
        _AUDIO_RING_IDX[key] = (idx + 1) & (_RING_SIZE - 1)
        slot = ring[idx]
        out_buffer, prev_dirty_rows = slot[0], slot[1]
        slot[1] = actual_size
    else:
        prev_dirty_rows = padded_batch_size

    PtrArray = ctypes.c_void_p * actual_size
    LenArray = ctypes.c_size_t * actual_size
    ptrs = PtrArray(*(c.ctypes.data for c in audio_chunks))
    lens = LenArray(*(c.shape[0] for c in audio_chunks))

    lib.whisper_rs_fill_padded_batch_dirty(
        ptrs,
        lens,
        actual_size,
        padded_batch_size,
        prev_dirty_rows,
        max_audio_length,
        out_buffer.ctypes.data,
    )
    return out_buffer
