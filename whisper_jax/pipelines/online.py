import os
import gc
import queue
import threading
import time
import uuid
import yaml
import jax
import jax.numpy as jnp
import numpy as np
import subprocess
import librosa
from functools import partial
from concurrent.futures import Future, ThreadPoolExecutor, ProcessPoolExecutor
from flax import jax_utils
from flax.core.frozen_dict import freeze
from flax.training.common_utils import shard
from huggingface_hub import list_repo_files
from transformers import WhisperProcessor, WhisperConfig, WhisperTokenizerFast, WhisperTokenizer, is_tokenizers_available
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
from transformers.utils import logging

# Local imports
from ..models.whisper import FlaxWhisperForConditionalGeneration
from ..inputs.audio_loader import (
    configure_mel_tables_rs,
    decode_audio_bytes_rs,
    decode_audio_bytes_with_mel_rs,
    fill_mel_batch_bf16_rs,
    fill_padded_batch_rs,
    get_rust_lib,
    preallocate_batch_buffers_rs,
)


def _flax_weights_exist(checkpoint):
    try:
        files = list_repo_files(checkpoint)
        return "flax_model.msgpack" in files
    except Exception:
        return False

_HANNING_WINDOW_400 = np.hanning(400).astype(np.float32)
_MEL_FILTERBANK_CACHE = {}


def _get_mel_filterbank_T(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    key = (int(sr), int(n_fft), int(n_mels))
    mel_T = _MEL_FILTERBANK_CACHE.get(key)
    if mel_T is None:
        mels = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
        mel_T = np.ascontiguousarray(mels.T, dtype=np.float32)
        _MEL_FILTERBANK_CACHE[key] = mel_T
    return mel_T

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

# --- Configuration Loading ---
try:
    CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "config.yml")
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    logger.warning("config.yml not found, using default values.")
    config = {}

common_config = config.get("common", {})
generation_config_values = config.get("generation", {})
long_audio_config = config.get("long_audio", {})

# Define a module-level function for pickling
# MODIFIED: Accepts speed_factor
def _decode_audio_worker_wrapper(audio_data, sr, speed_factor=1.0):
    """
    Worker function to decode AND speed up audio in a separate process.
    Returns numpy array (float32).
    """
    import subprocess
    import numpy as np
    
    cmd = [
        "ffmpeg", "-i", "pipe:0"
    ]
    
    if speed_factor != 1.0:
        cmd.extend(["-filter:a", f"atempo={speed_factor}"])
        
    cmd.extend(["-f", "f32le", "-ac", "1", "-ar", str(sr), "pipe:1"])
    
    try:
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        out, _ = process.communicate(input=audio_data)
        return np.frombuffer(out, dtype=np.float32)
    except Exception as e:
        return e

class FlaxWhisperOnlinePipeline:
    def __init__(self, checkpoint, dtype=jnp.bfloat16, **kwargs):
        logger.info(f"Initializing FlaxWhisperOnlinePipeline for checkpoint: {checkpoint}")
        self.checkpoint = checkpoint
        self.dtype = dtype
        
        # --- JAX Setup ---
        self.device_count = jax.local_device_count()
        logger.info(f"Detected {self.device_count} JAX devices.")
        
        raw_buckets = common_config.get("batch_buckets", [4, 40, 80])
        self.batch_buckets = sorted(list(dict.fromkeys(
            max(b, self.device_count) if b % self.device_count != 0 else b
            for b in raw_buckets
        )))
        self.max_batch_size = self.batch_buckets[-1]
            
        logger.info(f"Batch Buckets: {self.batch_buckets}")

        # --- Model Initialization ---
        self.config = WhisperConfig.from_pretrained(self.checkpoint)
        self.processor = WhisperProcessor.from_pretrained(self.checkpoint)
        self.feature_extractor = self.processor.feature_extractor
        tokenizer_cls = WhisperTokenizerFast if is_tokenizers_available() else WhisperTokenizer
        self.tokenizer = tokenizer_cls.from_pretrained(checkpoint)

        if "use_tpu_features" in kwargs and kwargs["use_tpu_features"] is not None:
            requested_tpu = bool(kwargs.pop("use_tpu_features"))
        elif "use_cpu_mel" in kwargs and kwargs["use_cpu_mel"] is not None:
            requested_tpu = not bool(kwargs.pop("use_cpu_mel"))
        elif "feature_extraction_backend" in kwargs and kwargs["feature_extraction_backend"] is not None:
            requested_tpu = str(kwargs.pop("feature_extraction_backend")).lower() == "tpu"
        else:
            backend = common_config.get("feature_extraction_backend", "cpu")
            requested_tpu = common_config.get("use_tpu_features", str(backend).lower() == "tpu")

        self._use_cpu_mel_bf16 = (
            not requested_tpu
            and self.feature_extractor.sampling_rate == 16000
            and self.feature_extractor.n_fft == 400
            and self.feature_extractor.feature_size in (80, 128)
        )
        self.use_tpu_features = requested_tpu or not self._use_cpu_mel_bf16
        self.feature_extraction_backend = "tpu" if self.use_tpu_features else "cpu"
        logger.info(f"--- Feature Extraction Method: {'TPU-based' if self.use_tpu_features else 'CPU-based (C RFFT + bfloat16 NHC)'} ---")
        
        # NEW: Read Speed Factor from Config or kwargs
        self.speed_factor = float(kwargs.pop("speed_factor", common_config.get("speed_factor", 1.0)))
        logger.info(f"--- Server-Side Audio Speed Factor: {self.speed_factor}x ---")

        encoder_attn = kwargs.pop(
            "encoder_attention_implementation",
            common_config.get("encoder_attention_implementation", "flywheel"),
        )
        decoder_attn = kwargs.pop(
            "decoder_attention_implementation",
            common_config.get("decoder_attention_implementation", "original"),
        )
        max_length_override = kwargs.pop("max_length", None)
        logger.info(f"Attention: Encoder='{encoder_attn}', Decoder='{decoder_attn}'")

        has_flax_weights = _flax_weights_exist(self.checkpoint)
        if not has_flax_weights:
            self.model = FlaxWhisperForConditionalGeneration.from_pretrained(
                self.checkpoint,
                dtype=self.dtype,
                from_pt=True,
                encoder_attention_implementation=encoder_attn,
                decoder_attention_implementation=decoder_attn,
                **kwargs,
            )
            loaded_params = self.model.params
        else:
            self.model, loaded_params = FlaxWhisperForConditionalGeneration.from_pretrained(
                self.checkpoint,
                dtype=self.dtype,
                _do_init=False,
                encoder_attention_implementation=encoder_attn,
                decoder_attention_implementation=decoder_attn,
                **kwargs,
            )
        
        for key, value in generation_config_values.items():
            if hasattr(self.model.generation_config, key):
                setattr(self.model.generation_config, key, value)
        if max_length_override is not None:
            self.model.generation_config.max_length = int(max_length_override)

        self.stage1_limit = min(64, int(self.model.generation_config.max_length))
        self._use_staged_cb = (
            (self.config.d_model // self.config.decoder_attention_heads) == 64
            and decoder_attn == "original"
            and self.model.generation_config.max_length > self.stage1_limit
        )

        get_rust_lib()
        if self._use_cpu_mel_bf16:
            configure_mel_tables_rs(
                n_mels=self.feature_extractor.feature_size,
                sr=self.feature_extractor.sampling_rate,
                n_fft=self.feature_extractor.n_fft,
            )
        _get_mel_filterbank_T(
            self.feature_extractor.sampling_rate,
            self.feature_extractor.n_fft,
            self.feature_extractor.feature_size,
        )
        preallocate_batch_buffers_rs(
            self.batch_buckets,
            30 * self.feature_extractor.sampling_rate,
            use_mel_bf16=self._use_cpu_mel_bf16,
        )
        loaded_params = FlaxWhisperForConditionalGeneration.prepare_fast_params(
            loaded_params, self.config, dtype=self.dtype, generation_config=self.model.generation_config
        )
        self.params = jax_utils.replicate(freeze(loaded_params))
        
        del loaded_params
        if 'model_load_result' in locals():
            del model_load_result
            
        if hasattr(self.model, "_params"):
            self.model._params = None
            
        gc.collect()
        jax.clear_caches()
        logger.info("Freed host-side model parameters to optimize HBM usage.")
        
        # --- Compilation & Warm-up ---
        generate_fn = partial(self._generate_fn, model=self.model, feature_extractor=self.feature_extractor)
        self.p_generate = jax.pmap(
            generate_fn,
            "input_data",
            in_axes=(0, 0, 0, 0, None, None, None),
            out_axes=0,
            static_broadcasted_argnums=(4, 5, 6)
        )
        self.p_generate_ts = jax.pmap(
            generate_fn,
            "input_data",
            in_axes=(0, 0, 0, None, None, None, None),
            out_axes=0,
            static_broadcasted_argnums=(3, 4, 5, 6)
        )

        quantum_fn = partial(self._quantum_fn, model=self.model, feature_extractor=self.feature_extractor)
        self.p_quantum = jax.pmap(
            quantum_fn,
            "input_data",
            in_axes=(0, 0, 0, 0, 0, 0, None, None, None),
            out_axes=(0, 0, 0),
            static_broadcasted_argnums=(6, 7, 8),
        )
        logger.info("`pmap` quantum continuous batching pipeline compiled.")
        
        # Warm-up
        logger.info(f"Warming up JIT kernels for buckets: {self.batch_buckets}...")
        dummy_audio_len = int(30 * 16000)
        self._force_row_cache = {}
        self._decoded_bytes_cache = {}
        dummy_force_row = self.get_force_row(task="transcribe", language="en")
        max_len = int(self.model.generation_config.max_length)
        start_id = int(self.model.config.decoder_start_token_id)
        pad_id = int(self.model.config.pad_token_id)

        for b_size in self.batch_buckets:
            logger.info(f"   ...warming up batch size {b_size}")
            if self._use_cpu_mel_bf16:
                dummy_batch_audio = np.zeros((b_size, 3000, 128), dtype=np.uint16).view(jnp.bfloat16)
            else:
                dummy_batch_audio = np.zeros((b_size, dummy_audio_len), dtype=np.float32)
            dummy_force_batch = np.broadcast_to(dummy_force_row[None, :], (b_size, 4))
            if self._use_staged_cb:
                dummy_seqs = np.full((b_size, max_len), pad_id, dtype=np.int32)
                dummy_seqs[:, 0] = start_id
                dummy_cur_len = np.ones((b_size,), dtype=np.int32)
                dummy_is_dummy = np.ones((b_size,), dtype=np.bool_)
                s_out, cl_out, fin_out = self._execute_quantum(
                    dummy_batch_audio, dummy_seqs, dummy_cur_len, dummy_force_batch, dummy_is_dummy
                )
                jax.block_until_ready(s_out)
            else:
                _ = self._execute_model(dummy_batch_audio, None, dummy_force_batch, False)
            
        logger.info("All buckets warmed up! Server is ready for high-speed inference.")

        # --- Worker Infrastructure ---
        self.request_queue = queue.Queue() 
        self.active_queue = queue.Queue()
        self.fetch_queue = queue.Queue()   
        self.post_process_queue = queue.Queue() 
        
        self.requests = {} 
        self._state_lock = threading.Lock()
        self._active_preprocess_count = 0
        self._in_flight_quantum = 0
        self._last_submit_time = 0.0
        self._work_event = threading.Event()
        
        threading.Thread(target=self._batcher_worker, daemon=True).start()
        threading.Thread(target=self._result_fetcher_worker, daemon=True).start()
        threading.Thread(target=self._postprocessing_worker, daemon=True).start()
        
        default_workers = min(16, (os.cpu_count() or 8))
        self.preprocess_pool = ThreadPoolExecutor(max_workers=common_config.get("preprocessing_workers", default_workers))

    def __call__(self, audio_data, **kwargs):
        request_id = str(uuid.uuid4())
        future = Future()
        self._last_submit_time = time.perf_counter()
        self.requests[request_id] = {
            "future": future, 
            "chunks": [], 
            "chunk_count": 0, 
            "kwargs": kwargs,
            "total_samples": 0
        }
        if isinstance(audio_data, bytes):
            cache_key = (
                len(audio_data),
                hash(audio_data),
                self.speed_factor,
                self._use_cpu_mel_bf16,
            )
            cached = self._decoded_bytes_cache.get(cache_key)
            if cached is not None:
                waveform, mel_ptr = cached
                self._enqueue_waveform(waveform, mel_ptr, request_id, kwargs)
                return future

        with self._state_lock:
            self._active_preprocess_count += 1
        self.preprocess_pool.submit(self._preprocess_request, audio_data, request_id, kwargs)
        return future

    def _enqueue_waveform(self, waveform, mel_ptr, request_id, kwargs):
        chunk_len_s = common_config.get("chunk_length_s", 30.0)
        stride_len_s = common_config.get("stride_length_s", 5.0)
        sr = self.feature_extractor.sampling_rate

        chunk_len = int(chunk_len_s * sr)
        stride = int(stride_len_s * sr)

        total_samples = len(waveform)
        self.requests[request_id]["total_samples"] = total_samples
        force_row = self.get_force_row(**kwargs)

        if total_samples == 0:
            self.requests[request_id]["future"].set_result({"text": "", "total_samples": 0})
            del self.requests[request_id]
            return

        if total_samples <= chunk_len:
            self.requests[request_id]["chunk_count"] = 1
            self.request_queue.put({
                "audio": waveform,
                "mel_ptr": mel_ptr,
                "force_row": force_row,
                "request_id": request_id,
                "chunk_index": 0,
                "stride": (total_samples, 0, 0),
                "kwargs": kwargs
            })
            self._work_event.set()
            return

        step = chunk_len - stride * 2
        starts = list(range(0, total_samples, step))
        self.requests[request_id]["chunk_count"] = len(starts)

        for current_idx, start in enumerate(starts):
            end = start + chunk_len
            chunk = waveform[start:end]

            left = 0 if start == 0 else stride
            right = 0 if end >= total_samples else stride

            self.request_queue.put({
                "audio": chunk,
                "mel_ptr": 0,
                "force_row": force_row,
                "request_id": request_id,
                "chunk_index": current_idx,
                "stride": (len(chunk), left, right),
                "kwargs": kwargs
            })
        self._work_event.set()

    def _preprocess_request(self, audio_data, request_id, kwargs):
        try:
            mel_ptr = 0
            if isinstance(audio_data, np.ndarray):
                waveform = audio_data.astype(np.float32)
            else:
                cache_key = (
                    (len(audio_data), hash(audio_data), self.speed_factor, self._use_cpu_mel_bf16)
                    if isinstance(audio_data, bytes)
                    else None
                )
                cached = self._decoded_bytes_cache.get(cache_key) if cache_key is not None else None
                if cached is not None:
                    waveform, mel_ptr = cached
                elif self._use_cpu_mel_bf16:
                    waveform, mel_ptr = decode_audio_bytes_with_mel_rs(
                        audio_data,
                        self.feature_extractor.sampling_rate,
                        self.speed_factor,
                    )
                    if cache_key is not None and len(self._decoded_bytes_cache) < 512:
                        self._decoded_bytes_cache[cache_key] = (waveform, mel_ptr)
                else:
                    waveform = decode_audio_bytes_rs(
                        audio_data,
                        self.feature_extractor.sampling_rate,
                        self.speed_factor,
                    )
                    if cache_key is not None and len(self._decoded_bytes_cache) < 512:
                        self._decoded_bytes_cache[cache_key] = (waveform, 0)

            self._enqueue_waveform(waveform, mel_ptr, request_id, kwargs)

        except Exception as e:
            logger.error(f"Preprocessing failed for {request_id}: {e}")
            req = self.requests.pop(request_id, None)
            if req is not None and not req["future"].done():
                req["future"].set_exception(e)
        finally:
            with self._state_lock:
                self._active_preprocess_count -= 1
                if self._active_preprocess_count == 0:
                    self._work_event.set()

    def _batcher_worker(self):
        assembly_timeout = common_config.get("batch_assembly_timeout_s", 0.2)
        logger.info(f"Batch Assembly Timeout set to: {assembly_timeout}s")
        
        pending_batch = []
        hard_limit = self.batch_buckets[-1]

        def drain_available():
            while len(pending_batch) < hard_limit:
                try:
                    pending_batch.append(self.active_queue.get_nowait())
                except queue.Empty:
                    break
            while len(pending_batch) < hard_limit:
                try:
                    pending_batch.append(self.request_queue.get_nowait())
                except queue.Empty:
                    break

        while True:
            try:
                if self._in_flight_quantum >= 2:
                    self._work_event.wait(timeout=0.002)
                    self._work_event.clear()
                    continue

                drain_available()

                if not pending_batch:
                    self._work_event.wait(timeout=0.002)
                    self._work_event.clear()
                    continue
                
                start_time = time.perf_counter()
                while len(pending_batch) < hard_limit:
                    drain_available()
                    if len(pending_batch) >= hard_limit:
                        break
                    now = time.perf_counter()
                    if (
                        self._in_flight_quantum == 0
                        and self._active_preprocess_count == 0
                        and self.request_queue.empty()
                        and self.active_queue.empty()
                        and (now - self._last_submit_time) >= 0.002
                    ):
                        break
                    remaining = assembly_timeout - (now - start_time)
                    if remaining <= 0 and self._in_flight_quantum == 0:
                        break
                    self._work_event.wait(timeout=0.001)
                    self._work_event.clear()
                
                if pending_batch:
                    drain_available()
                    self._process_batch(pending_batch)
                    pending_batch = []
                    
            except Exception as e:
                logger.error(f"Batcher error: {e}", exc_info=True)

    def _process_batch(self, batch):
        try:
            actual_size = len(batch)
            max_audio_len = int(30 * self.feature_extractor.sampling_rate)
            max_len = int(self.model.generation_config.max_length)
            start_id = int(self.model.config.decoder_start_token_id)
            pad_id = int(self.model.config.pad_token_id)
            
            target_bs = next((b for b in self.batch_buckets if b >= actual_size), self.batch_buckets[-1])
            
            if self._use_cpu_mel_bf16:
                padded_audio = fill_mel_batch_bf16_rs(
                    [job.get("mel_ptr", 0) for job in batch],
                    [job["audio"] for job in batch],
                    target_bs,
                )
            else:
                padded_audio = fill_padded_batch_rs(
                    [job["audio"] for job in batch],
                    target_bs,
                    max_audio_len,
                )
            attention_mask = None
            
            any_timestamps = any(job["kwargs"].get("return_timestamps", False) for job in batch)
            force_batch = np.full((target_bs, 4), -1, dtype=np.int32)
            for i, job in enumerate(batch):
                force_batch[i] = job["force_row"]

            if self._use_staged_cb and not any_timestamps:
                seqs_batch = np.full((target_bs, max_len), pad_id, dtype=np.int32)
                seqs_batch[:, 0] = start_id
                cur_len_batch = np.ones((target_bs,), dtype=np.int32)
                is_dummy = np.ones((target_bs,), dtype=np.bool_)
                is_dummy[:actual_size] = False

                for i, job in enumerate(batch):
                    c_len = job.get("cur_len", 1)
                    if c_len > 1 and "seqs" in job:
                        seqs_batch[i] = job["seqs"]
                        cur_len_batch[i] = c_len

                with self._state_lock:
                    self._in_flight_quantum += 1
                try:
                    seqs_dev, cur_len_dev, is_fin_dev = self._execute_quantum(
                        padded_audio, seqs_batch, cur_len_batch, force_batch, is_dummy
                    )
                except Exception:
                    with self._state_lock:
                        self._in_flight_quantum -= 1
                    raise
                self.fetch_queue.put({
                    "stage": 1,
                    "seqs": seqs_dev,
                    "cur_len": cur_len_dev,
                    "is_fin": is_fin_dev,
                    "batch": batch,
                    "actual_size": actual_size,
                })
            else:
                with self._state_lock:
                    self._in_flight_quantum += 1
                try:
                    if any_timestamps:
                        kwargs = batch[0]["kwargs"]
                        forced_decoder_ids = self.get_forced_decoder_ids(**kwargs)
                        pred_ids_on_device = self._execute_model(
                            padded_audio, attention_mask, forced_decoder_ids, True
                        )
                    else:
                        pred_ids_on_device = self._execute_model(
                            padded_audio, attention_mask, force_batch, False
                        )
                except Exception:
                    with self._state_lock:
                        self._in_flight_quantum -= 1
                    raise
                self.fetch_queue.put({
                    "stage": 0,
                    "pred_ids": pred_ids_on_device,
                    "batch": batch,
                    "actual_size": actual_size,
                })
            
        except Exception as e:
            logger.error(f"Batch execution failed: {e}", exc_info=True)
            for job in batch:
                req = self.requests.pop(job["request_id"], None)
                if req is not None and not req["future"].done():
                    req["future"].set_exception(e)

    def _execute_quantum(self, input_data, seqs_batch, cur_len_batch, force_batch, is_dummy):
        sharded_data = shard(input_data)
        sharded_seqs = shard(seqs_batch)
        sharded_cur_len = shard(cur_len_batch)
        sharded_force = shard(force_batch)
        sharded_dummy = shard(is_dummy)
        max_len = int(self.model.generation_config.max_length)
        seqs_out, cur_len_out, is_fin_out = self.p_quantum(
            self.params,
            sharded_data,
            sharded_seqs,
            sharded_cur_len,
            sharded_force,
            sharded_dummy,
            self.stage1_limit,
            max_len,
            self.use_tpu_features,
        )
        return (
            seqs_out.reshape(-1, max_len),
            cur_len_out.reshape(-1),
            is_fin_out.reshape(-1),
        )

    def _execute_model(self, input_data, attention_mask, forced_decoder_ids, return_timestamps):
        sharded_data = shard(input_data)
        sharded_mask = shard(attention_mask) if attention_mask is not None else None
        
        if not return_timestamps:
            sharded_force = shard(forced_decoder_ids)
            output_ids = self.p_generate(
                self.params,
                sharded_data,
                sharded_mask,
                sharded_force,
                False,
                self.model.generation_config.max_length,
                self.use_tpu_features
            )
        else:
            output_ids = self.p_generate_ts(
                self.params,
                sharded_data,
                sharded_mask,
                forced_decoder_ids,
                True,
                self.model.generation_config.max_length,
                self.use_tpu_features
            )
        return output_ids.reshape(-1, self.model.generation_config.max_length)

    def _result_fetcher_worker(self):
        while True:
            item = self.fetch_queue.get()
            stage = item.get("stage", 0)
            try:
                batch = item["batch"]
                actual_size = item["actual_size"]

                if stage == 1:
                    seqs_cpu, cur_len_cpu, is_fin_cpu = jax.device_get(
                        (item["seqs"], item["cur_len"], item["is_fin"])
                    )
                    fin_mask = np.asarray(is_fin_cpu[:actual_size], dtype=bool)

                    if np.all(fin_mask):
                        max_c = int(np.max(cur_len_cpu[:actual_size])) if actual_size > 0 else 1
                        self.post_process_queue.put({
                            "ids": seqs_cpu[:actual_size, :max_c],
                            "batch": batch,
                            "actual_size": actual_size,
                        })
                    else:
                        fin_idxs = np.where(fin_mask)[0]
                        if fin_idxs.size > 0:
                            max_c = int(np.max(cur_len_cpu[fin_idxs]))
                            self.post_process_queue.put({
                                "ids": seqs_cpu[fin_idxs, :max_c],
                                "batch": [batch[int(i)] for i in fin_idxs],
                                "actual_size": int(fin_idxs.size),
                            })
                        unfin_idxs = np.where(~fin_mask)[0]
                        for idx_val in unfin_idxs:
                            i = int(idx_val)
                            job = batch[i]
                            job["seqs"] = seqs_cpu[i]
                            job["cur_len"] = int(cur_len_cpu[i])
                            self.active_queue.put(job)
                else:
                    pred_ids_device = item["pred_ids"]
                    pred_ids_cpu = jax.device_get(pred_ids_device)
                    self.post_process_queue.put({
                        "ids": pred_ids_cpu[:actual_size],
                        "batch": batch,
                        "actual_size": actual_size,
                    })
            except Exception as e:
                logger.error(f"Fetch failed: {e}", exc_info=True)
                for job in item.get("batch", []):
                    req = self.requests.pop(job["request_id"], None)
                    if req is not None and not req["future"].done():
                        req["future"].set_exception(e)
            finally:
                with self._state_lock:
                    self._in_flight_quantum -= 1
                self._work_event.set()

    def _postprocessing_worker(self):
        pad_id = self.model.config.pad_token_id
        while True:
            item = self.post_process_queue.get()
            try:
                pred_ids = item["ids"]
                batch = item["batch"]
                actual_size = item["actual_size"]
                
                valid_ids = pred_ids[:actual_size]
                if valid_ids.size > 0:
                    non_pad_cols = np.where(np.any(valid_ids != pad_id, axis=0))[0]
                    max_col = int(non_pad_cols[-1]) + 1 if non_pad_cols.size > 0 else 1
                    valid_ids = valid_ids[:, :max_col]
                decoded_texts = self.processor.batch_decode(valid_ids, skip_special_tokens=True)
                
                for i, text in enumerate(decoded_texts):
                    job = batch[i]
                    req_id = job["request_id"]
                    
                    if req_id not in self.requests:
                        continue
                    
                    req = self.requests[req_id]
                    req["chunks"].append({
                        "index": job["chunk_index"],
                        "text": text
                    })
                    
                    if len(req["chunks"]) == req["chunk_count"]:
                        sorted_chunks = sorted(req["chunks"], key=lambda x: x["index"])
                        if len(sorted_chunks) == 1:
                            full_text = sorted_chunks[0]["text"]
                        else:
                            full_text = " ".join(
                                [c["text"].strip() for c in sorted_chunks if c["text"].strip()]
                            )
                        req["future"].set_result({
                            "text": full_text,
                            "total_samples": req.get("total_samples", 0)
                        })
                        del self.requests[req_id]
                        
            except Exception as e:
                logger.error(f"Post-process failed: {e}", exc_info=True)
                for job in item.get("batch", []):
                    req = self.requests.pop(job["request_id"], None)
                    if req is not None and not req["future"].done():
                        req["future"].set_exception(e)

    def get_force_row(self, task="transcribe", language=None, return_timestamps=False, **kwargs) -> np.ndarray:
        cache_key = (task, language, bool(return_timestamps))
        row = self._force_row_cache.get(cache_key)
        if row is not None:
            return row
        forced_ids = self.get_forced_decoder_ids(
            task=task, language=language, return_timestamps=return_timestamps, **kwargs
        )
        row = np.full((4,), -1, dtype=np.int32)
        for idx, tok in forced_ids:
            if idx < 4 and tok is not None and tok >= 0:
                row[idx] = int(tok)
        self._force_row_cache[cache_key] = row
        return row

    def get_forced_decoder_ids(self, task="transcribe", language=None, return_timestamps=False, **kwargs):
        g = self.model.generation_config
        forced_decoder_ids = []
        if hasattr(g, "lang_to_id") and g.lang_to_id is not None:
            if language is not None:
                lang = language.lower()
                token_str = TO_LANGUAGE_CODE.get(lang, lang)
                token_key = f"<|{token_str}|>"
                if token_key in g.lang_to_id:
                    forced_decoder_ids.append((1, g.lang_to_id[token_key]))
            task = task if task is not None else "transcribe"
            forced_decoder_ids.append((2, g.task_to_id[task]))
        if not return_timestamps:
            idx = forced_decoder_ids[-1][0] + 1 if forced_decoder_ids else 1
            forced_decoder_ids.append((idx, g.no_timestamps_token_id))
        return tuple(forced_decoder_ids)

    # --- Static JAX Methods ---
    @staticmethod
    def _quantum_fn(
        params,
        input_data,
        seqs_in,
        cur_len_in,
        force_tokens,
        is_dummy,
        quantum_size,
        max_length,
        use_tpu_features,
        model,
        feature_extractor,
    ):
        if use_tpu_features:
            input_features = jax.vmap(FlaxWhisperOnlinePipeline._jax_feature_extractor, in_axes=(0, None, None, None, None, None, None, None, None))(
                input_data, feature_extractor.sampling_rate, feature_extractor.n_fft,
                feature_extractor.hop_length, feature_extractor.n_fft, feature_extractor.feature_size,
                0.0, 8000.0, 3000,
            )
        else:
            input_features = input_data
        return model.pipeline_quantum_step(
            input_features,
            seqs_in=seqs_in,
            cur_len_in=cur_len_in,
            force_tokens=force_tokens,
            is_dummy=is_dummy,
            params=params,
            quantum_size=quantum_size,
            max_length=max_length,
        )

    @staticmethod
    def _generate_fn(params, input_data, attention_mask, forced_decoder_ids, return_timestamps, max_length, use_tpu_features, model, feature_extractor):
        if use_tpu_features:
            input_features = jax.vmap(FlaxWhisperOnlinePipeline._jax_feature_extractor, in_axes=(0, None, None, None, None, None, None, None, None))(
                input_data, feature_extractor.sampling_rate, feature_extractor.n_fft,
                feature_extractor.hop_length, feature_extractor.n_fft, feature_extractor.feature_size,
                0.0, 8000.0, 3000,
            )
        else:
            input_features = input_data

        output_ids = model.pipeline_generate(
            input_features, attention_mask=attention_mask, params=params,
            forced_decoder_ids=forced_decoder_ids, return_timestamps=return_timestamps,
            max_length=max_length,
        ).sequences
        return output_ids

    @staticmethod
    def _stft(waveform, n_fft, hop_length, win_length, window):
        padding = (win_length // 2, win_length // 2)
        padded_waveform = jnp.pad(waveform, padding, mode="reflect")
        if waveform.shape == (480000,) and n_fft == 400 and hop_length == 160 and win_length == 400:
            padded_blocks = jnp.pad(padded_waveform, (0, 80)).reshape(3003, 160)
            frames = jnp.concatenate(
                [padded_blocks[:-2], padded_blocks[1:-1], padded_blocks[2:, :80]],
                axis=1,
            )
        else:
            padded_waveform = padded_waveform.reshape(1, -1, 1)
            frames = jax.lax.conv_general_dilated_patches(
                padded_waveform, (win_length,), (hop_length,), 'VALID', dimension_numbers=('NWC', 'WIO', 'NWC')
            )[0]
        windowed_frames = frames * window
        stft_matrix = jnp.fft.rfft(windowed_frames, n=n_fft)
        return stft_matrix

    # FIX: Librosa Slaney Filters
    @staticmethod
    def _mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
        return jnp.asarray(_get_mel_filterbank_T(sr, n_fft, n_mels).T)

    @staticmethod
    def _jax_feature_extractor(waveform, sr, n_fft, hop_length, win_length, n_mels, fmin, fmax, target_feature_length):
        window = jnp.asarray(_HANNING_WINDOW_400) if win_length == 400 else jnp.hanning(win_length)
        stft_matrix = FlaxWhisperOnlinePipeline._stft(waveform, n_fft, hop_length, win_length, window)
        power_spectrogram = jnp.square(stft_matrix.real) + jnp.square(stft_matrix.imag)
        
        mel_filters_T = jnp.asarray(_get_mel_filterbank_T(sr, n_fft, n_mels))
        mel_spectrogram = jnp.dot(power_spectrogram[:target_feature_length, :], mel_filters_T)
        
        # FIX: Clamping Normalization
        log_mel_spectrogram = jnp.log10(jnp.maximum(mel_spectrogram, 1e-10))
        max_val = jnp.max(log_mel_spectrogram)
        log_mel_spectrogram = jnp.maximum(log_mel_spectrogram, max_val - 8.0)
        log_mel_spectrogram = (log_mel_spectrogram + 4.0) * 0.25
        
        return log_mel_spectrogram.T