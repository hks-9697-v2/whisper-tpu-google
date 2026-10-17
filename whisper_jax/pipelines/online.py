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
        
        # NEW: Read Speed Factor from Config
        self.speed_factor = common_config.get("speed_factor", 1.0)
        logger.info(f"--- Server-Side Audio Speed Factor: {self.speed_factor}x ---")

        encoder_attn = kwargs.pop(
            "encoder_attention_implementation",
            common_config.get("encoder_attention_implementation", "flywheel"),
        )
        decoder_attn = kwargs.pop(
            "decoder_attention_implementation",
            common_config.get("decoder_attention_implementation", "original"),
        )
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
            in_axes=(0, 0, 0, None, None, None, None), 
            out_axes=0, 
            static_broadcasted_argnums=(3, 4, 5, 6)
        )
        logger.info("`pmap` pipeline compiled.")
        
        # Warm-up
        logger.info(f"Warming up JIT kernels for buckets: {self.batch_buckets}...")
        dummy_audio_len = int(30 * 16000)
        if getattr(self.model.generation_config, "lang_to_id", None) is not None:
            warmup_configs = [
                {"language": None, "task": "transcribe"},
                {"language": "en", "task": "transcribe"},
            ]
        else:
            warmup_configs = [
                {"language": None, "task": "transcribe"},
            ]

        for b_size in self.batch_buckets:
            for w_conf in warmup_configs:
                logger.info(f"   ...warming up batch size {b_size} | lang={w_conf['language']}")
                if self._use_cpu_mel_bf16:
                    dummy_batch_audio = np.zeros((b_size, 3000, 128), dtype=np.uint16).view(jnp.bfloat16)
                else:
                    dummy_batch_audio = np.zeros((b_size, dummy_audio_len), dtype=np.float32)
                dummy_mask = None
                forced_ids = self.get_forced_decoder_ids(task=w_conf["task"], language=w_conf["language"])
                _ = self._execute_model(dummy_batch_audio, dummy_mask, forced_ids, False)
            
        logger.info("All buckets warmed up! Server is ready for high-speed inference.")

        # --- Worker Infrastructure ---
        self.request_queue = queue.Queue() 
        self.fetch_queue = queue.Queue()   
        self.post_process_queue = queue.Queue() 
        
        self.requests = {} 
        
        threading.Thread(target=self._batcher_worker, daemon=True).start()
        threading.Thread(target=self._result_fetcher_worker, daemon=True).start()
        threading.Thread(target=self._postprocessing_worker, daemon=True).start()
        
        self.preprocess_pool = ThreadPoolExecutor(max_workers=common_config.get("preprocessing_workers", 64))

    def __call__(self, audio_data, **kwargs):
        request_id = str(uuid.uuid4())
        future = Future()
        self.requests[request_id] = {
            "future": future, 
            "chunks": [], 
            "chunk_count": 0, 
            "kwargs": kwargs,
            "total_samples": 0
        }
        self.preprocess_pool.submit(self._preprocess_request, audio_data, request_id, kwargs)
        return future

    def _preprocess_request(self, audio_data, request_id, kwargs):
        try:
            mel_ptr = 0
            if isinstance(audio_data, np.ndarray):
                waveform = audio_data.astype(np.float32)
            elif self._use_cpu_mel_bf16:
                waveform, mel_ptr = decode_audio_bytes_with_mel_rs(
                    audio_data,
                    self.feature_extractor.sampling_rate,
                    self.speed_factor,
                )
            else:
                waveform = decode_audio_bytes_rs(
                    audio_data,
                    self.feature_extractor.sampling_rate,
                    self.speed_factor,
                )

            chunk_len_s = common_config.get("chunk_length_s", 30.0)
            stride_len_s = common_config.get("stride_length_s", 5.0)
            sr = self.feature_extractor.sampling_rate
            
            chunk_len = int(chunk_len_s * sr)
            stride = int(stride_len_s * sr)
            
            total_samples = len(waveform)
            self.requests[request_id]["total_samples"] = total_samples
            
            if total_samples <= chunk_len:
                self.requests[request_id]["chunk_count"] = 1
                self.request_queue.put({
                    "audio": waveform,
                    "mel_ptr": mel_ptr,
                    "request_id": request_id,
                    "chunk_index": 0,
                    "stride": (total_samples, 0, 0),
                    "kwargs": kwargs
                })
                return

            step = chunk_len - stride * 2 
            
            current_idx = 0
            for start in range(0, total_samples, step):
                end = start + chunk_len
                chunk = waveform[start:end]
                
                left = 0 if start == 0 else stride
                right = 0 if end >= total_samples else stride
                
                self.request_queue.put({
                    "audio": chunk,
                    "mel_ptr": 0,
                    "request_id": request_id,
                    "chunk_index": current_idx,
                    "stride": (len(chunk), left, right),
                    "kwargs": kwargs
                })
                current_idx += 1
            
            self.requests[request_id]["chunk_count"] = current_idx

        except Exception as e:
            logger.error(f"Preprocessing failed for {request_id}: {e}")
            if request_id in self.requests:
                self.requests[request_id]["future"].set_exception(e)

    def _batcher_worker(self):
        assembly_timeout = common_config.get("batch_assembly_timeout_s", 0.2)
        logger.info(f"Batch Assembly Timeout set to: {assembly_timeout}s")
        
        pending_batch = []
        
        while True:
            try:
                if not pending_batch:
                    job = self.request_queue.get()
                    pending_batch.append(job)
                    logger.debug(f"Batcher: Started new batch. Queue size: {self.request_queue.qsize()}")
                
                hard_limit = self.batch_buckets[-1]
                
                start_time = time.time()
                while len(pending_batch) < hard_limit:
                    elapsed = time.time() - start_time
                    remaining = assembly_timeout - elapsed
                    if remaining <= 0:
                        logger.debug("Batcher: Assembly timeout reached.")
                        break
                    
                    try:
                        job = self.request_queue.get(timeout=remaining)
                        pending_batch.append(job)
                    except queue.Empty:
                        break
                
                if pending_batch:
                    batch_size = len(pending_batch)
                    assembly_time = time.time() - start_time
                    logger.info(f"Batcher: Processing batch of size {batch_size} (Buckets: {self.batch_buckets}). Assembly time: {assembly_time:.4f}s. Queue size: {self.request_queue.qsize()}")
                    self._process_batch(pending_batch)
                    pending_batch = []
                    
            except Exception as e:
                logger.error(f"Batcher error: {e}", exc_info=True)

    def _process_batch(self, batch):
        try:
            actual_size = len(batch)
            max_audio_len = int(30 * self.feature_extractor.sampling_rate)
            
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
            
            kwargs = batch[0]["kwargs"]
            forced_decoder_ids = self.get_forced_decoder_ids(**kwargs)
            return_timestamps = kwargs.get("return_timestamps", False)
            
            pred_ids_on_device = self._execute_model(
                padded_audio, attention_mask, forced_decoder_ids, return_timestamps
            )
            
            self.fetch_queue.put({
                "pred_ids": pred_ids_on_device,
                "batch": batch,
                "actual_size": actual_size
            })
            
        except Exception as e:
            logger.error(f"Batch execution failed: {e}", exc_info=True)
            for job in batch:
                if job["request_id"] in self.requests:
                    self.requests[job["request_id"]]["future"].set_exception(e)

    def _execute_model(self, input_data, attention_mask, forced_decoder_ids, return_timestamps):
        sharded_data = shard(input_data)
        sharded_mask = shard(attention_mask) if attention_mask is not None else None
        
        output_ids = self.p_generate(
            self.params, 
            sharded_data, 
            sharded_mask,
            forced_decoder_ids, 
            return_timestamps,
            self.model.generation_config.max_length,
            self.use_tpu_features
        )
        return output_ids.reshape(-1, self.model.generation_config.max_length)

    def _result_fetcher_worker(self):
        while True:
            item = self.fetch_queue.get()
            try:
                pred_ids_device = item["pred_ids"]
                batch = item["batch"]
                actual_size = item["actual_size"]
                
                pred_ids_cpu = jax.device_get(pred_ids_device)
                
                self.post_process_queue.put({
                    "ids": pred_ids_cpu,
                    "batch": batch,
                    "actual_size": actual_size
                })
            except Exception as e:
                logger.error(f"Fetch failed: {e}")

    def _postprocessing_worker(self):
        while True:
            item = self.post_process_queue.get()
            try:
                pred_ids = item["ids"]
                batch = item["batch"]
                actual_size = item["actual_size"]
                
                valid_ids = pred_ids[:actual_size]
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
                        full_text = " ".join([c["text"] for c in sorted_chunks])
                        req["future"].set_result({
                            "text": full_text,
                            "total_samples": req.get("total_samples", 0)
                        })
                        del self.requests[req_id]
                        
            except Exception as e:
                logger.error(f"Post-process failed: {e}", exc_info=True)

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