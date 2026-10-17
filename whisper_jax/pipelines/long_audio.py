# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ===== Standard Library Imports =====
import gc
import os
import queue
import re
import subprocess
import threading
import time
import yaml
import librosa  # Import librosa for Slaney filters
from functools import partial
from concurrent.futures import Future, ThreadPoolExecutor, wait

# ===== Third-Party Library Imports =====
import jax
import jax.numpy as jnp
import numpy as np
from huggingface_hub import list_repo_files
from jax import lax
from flax import jax_utils
from flax.core.frozen_dict import freeze
from flax.training.common_utils import shard
from jax.sharding import PartitionSpec as P
from transformers import (
    WhisperConfig,
    WhisperProcessor,
    WhisperTokenizer,
    WhisperTokenizerFast,
    is_tokenizers_available,
)
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
from transformers.utils import logging

# ===== Local Imports =====
from ..models.whisper import FlaxWhisperForConditionalGeneration
from ..inputs.audio_loader import (
    configure_mel_tables_rs,
    get_rust_lib,
    preallocate_batch_buffers_rs,
    decode_audio_file_rs,
    decode_audio_file_with_mel_rs,
    decode_audio_bytes_rs,
    decode_audio_bytes_with_mel_rs,
    fill_mel_batch_bf16_rs,
    fill_padded_batch_rs,
)

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

# ===== Globals =====
logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

# Load configuration from YAML file
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

# Map string dtype to jnp dtype
dtype_map = {
    "bfloat16": jnp.bfloat16,
    "float16": jnp.float16,
    "float32": jnp.float32,
}
default_dtype_str = common_config.get('dtype', 'bfloat16')
default_dtype = dtype_map.get(default_dtype_str, jnp.bfloat16)


def _flax_weights_exist(checkpoint):
    try:
        files = list_repo_files(checkpoint)
        return "flax_model.msgpack" in files
    except Exception:
        return False

class FlaxWhisperPmapPipeline:
    BATCH_BUCKETS = common_config.get("batch_buckets", [4, 40, 80])

    def __init__(
        self,
        checkpoint,
        dtype=default_dtype,
        batch_size=None,
        max_length=None,
        encoder_attention_implementation: str = common_config.get("encoder_attention_implementation", "flywheel"),
        decoder_attention_implementation: str = common_config.get("decoder_attention_implementation", "original"),
        feature_extraction_backend: str = common_config.get("feature_extraction_backend", "cpu"),
        skip_special_tokens: bool = generation_config_values.get("skip_special_tokens", True),
        **kwargs,
    ):
        logger.info(f"Initialising FlaxWhisperPmapPipeline for checkpoint: {checkpoint}")
        self.checkpoint = checkpoint
        self.dtype = dtype
        self.config = WhisperConfig.from_pretrained(self.checkpoint)
        self.processor = WhisperProcessor.from_pretrained(self.checkpoint)
        self.feature_extractor = self.processor.feature_extractor
        tokenizer_cls = WhisperTokenizerFast if is_tokenizers_available() else WhisperTokenizer
        self.tokenizer = tokenizer_cls.from_pretrained(checkpoint)
        self.skip_special_tokens = skip_special_tokens

        if "use_tpu_features" in kwargs and kwargs["use_tpu_features"] is not None:
            requested_tpu = bool(kwargs.pop("use_tpu_features"))
        elif "use_cpu_mel" in kwargs and kwargs["use_cpu_mel"] is not None:
            requested_tpu = not bool(kwargs.pop("use_cpu_mel"))
        elif "use_tpu_features" in common_config:
            requested_tpu = bool(common_config["use_tpu_features"])
        else:
            requested_tpu = str(feature_extraction_backend).lower() == "tpu"

        self._use_cpu_mel_bf16 = (
            not requested_tpu
            and self.feature_extractor.sampling_rate == 16000
            and self.feature_extractor.n_fft == 400
            and self.feature_extractor.feature_size in (80, 128)
        )
        self.use_tpu_features = requested_tpu or not self._use_cpu_mel_bf16
        self.feature_extraction_backend = "tpu" if self.use_tpu_features else "cpu"
        logger.info(
            f"--- Feature Extraction Method: {'TPU-based' if self.use_tpu_features else 'CPU-based (C RFFT + bfloat16 NHC)'} "
            f"| Attention: encoder='{encoder_attention_implementation}', decoder='{decoder_attention_implementation}' ---"
        )

        has_flax_weights = _flax_weights_exist(self.checkpoint)

        if not has_flax_weights:
            self.model = FlaxWhisperForConditionalGeneration.from_pretrained(
                self.checkpoint,
                dtype=self.dtype,
                from_pt=True,
                encoder_attention_implementation=encoder_attention_implementation,
                decoder_attention_implementation=decoder_attention_implementation,
                **kwargs,
            )
            loaded_params = self.model.params
        else:
            self.model, loaded_params = FlaxWhisperForConditionalGeneration.from_pretrained(
                self.checkpoint,
                dtype=self.dtype,
                _do_init=False,
                encoder_attention_implementation=encoder_attention_implementation,
                decoder_attention_implementation=decoder_attention_implementation,
                **kwargs,
            )
        
        for key, value in generation_config_values.items():
            if hasattr(self.model.generation_config, key):
                setattr(self.model.generation_config, key, value)

        self.max_length = max_length if max_length is not None else self.model.generation_config.max_length
        self.min_batch_size = jax.local_device_count()
        self.BATCH_BUCKETS = sorted(list(dict.fromkeys(
            max(b, self.min_batch_size) if b % self.min_batch_size != 0 else b
            for b in self.BATCH_BUCKETS
        )))
        self.batch_size = batch_size if batch_size is not None else common_config.get("global_batch_size", self.min_batch_size)

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
            self.BATCH_BUCKETS,
            30 * self.feature_extractor.sampling_rate,
            use_mel_bf16=self._use_cpu_mel_bf16,
        )
        max_workers = common_config.get("preprocessing_workers") or min(32, (os.cpu_count() or 1) + 4)
        self._preprocess_executor = ThreadPoolExecutor(max_workers=max(max_workers, 32))
        loaded_params = FlaxWhisperForConditionalGeneration.prepare_fast_params(
            loaded_params, self.config, dtype=self.dtype, generation_config=self.model.generation_config
        )
        self.params = jax_utils.replicate(freeze(loaded_params))
        
        del loaded_params
        gc.collect()
        jax.clear_caches()
        
        generate_fn = partial(self._generate_fn, model=self.model, feature_extractor=self.feature_extractor)
        
        self.p_generate = jax.pmap(
            generate_fn, 
            "input_audio", 
            in_axes=(0, 0, 0, None, None, None, None), 
            out_axes=0, 
            static_broadcasted_argnums=(3, 4, 5, 6)
        )
        logger.info("`pmap` pipeline ready.")

    def __call__(self, inputs, 
        chunk_length_s: float = common_config.get("chunk_length_s", 30.0),
        stride_length_s: float = common_config.get("stride_length_s"),
        batch_size=None, language=None, task=None, return_timestamps=None, return_language=None, max_length=None, 
        speed_factor: float = None, # New param
        **kwargs):
        if speed_factor is None:
            speed_factor = common_config.get("speed_factor", 1.0)
            
        effective_batch_size = batch_size if batch_size is not None else self.batch_size
        if effective_batch_size % self.min_batch_size != 0:
            raise ValueError(f"Batch size must be a multiple of devices: {effective_batch_size} vs {self.min_batch_size}.")
        
        is_list_input = isinstance(inputs, list)
        if not is_list_input:
            inputs = [inputs]
        logger.info(f"Starting transcription for {len(inputs)} file(s)...")

        all_jobs, file_chunks = self._build_round_robin_chunks(
            inputs,
            chunk_length_s,
            stride_length_s,
            speed_factor,
        )
        if not all_jobs:
            empty_res = [self._final_decode([], return_timestamps, return_language) for _ in inputs]
            return empty_res if is_list_input else empty_res[0]

        self.fetch_queue = queue.Queue(maxsize=8)
        result_fetcher_thread = threading.Thread(
            target=self._result_fetcher_worker,
            args=(return_timestamps, file_chunks),
        )
        result_fetcher_thread.start()

        max_bucket_size = self.BATCH_BUCKETS[-1]
        step_size = min(effective_batch_size, max_bucket_size)
        max_audio_length = 30 * self.feature_extractor.sampling_rate
        exec_kwargs = {
            "language": language,
            "task": task,
            "return_timestamps": return_timestamps,
            "max_length": max_length,
        }

        for start in range(0, len(all_jobs), step_size):
            chunk = all_jobs[start : start + step_size]
            actual_chunk_size = len(chunk)
            padded_batch_size = next(
                (b for b in self.BATCH_BUCKETS if b >= actual_chunk_size),
                max_bucket_size,
            )
            pad_width = padded_batch_size - actual_chunk_size

            if self._use_cpu_mel_bf16:
                padded_batch = fill_mel_batch_bf16_rs(
                    [job[1] for job in chunk],
                    [job[0] for job in chunk],
                    padded_batch_size,
                )
                attention_mask = None
            elif self.use_tpu_features:
                padded_batch = fill_padded_batch_rs(
                    [job[0] for job in chunk],
                    padded_batch_size,
                    max_audio_length,
                )
                attention_mask = None
            else:
                input_features = self.processor(
                    [job[0] for job in chunk],
                    sampling_rate=self.feature_extractor.sampling_rate,
                    return_tensors="np",
                    padding="longest",
                ).input_features
                padded_batch = np.pad(
                    input_features,
                    ((0, pad_width), (0, 0), (0, 3000 - input_features.shape[2])),
                    mode="constant",
                )
                attention_mask = None

            pred_ids_on_device = self._execute_model(padded_batch, attention_mask, **exec_kwargs)
            self.fetch_queue.put((pred_ids_on_device, chunk, actual_chunk_size))

        self.fetch_queue.put(None)
        result_fetcher_thread.join()

        final_results = [
            self._final_decode(file_chunks[i], return_timestamps, return_language)
            for i in range(len(inputs))
        ]
        logger.info("...transcription finished.")
        return final_results if is_list_input else final_results[0]

    def _decode_single_input(self, file_path, speed_factor=1.0):
        mel_ptr = 0
        if isinstance(file_path, np.ndarray):
            full_waveform = file_path.astype(np.float32, copy=False)
        elif isinstance(file_path, (bytes, bytearray)):
            if self._use_cpu_mel_bf16:
                full_waveform, mel_ptr = decode_audio_bytes_with_mel_rs(file_path, self.feature_extractor.sampling_rate, speed_factor)
            else:
                full_waveform = decode_audio_bytes_rs(file_path, self.feature_extractor.sampling_rate, speed_factor)
        else:
            if self._use_cpu_mel_bf16:
                full_waveform, mel_ptr = decode_audio_file_with_mel_rs(file_path, self.feature_extractor.sampling_rate, speed_factor)
            else:
                full_waveform = decode_audio_file_rs(file_path, self.feature_extractor.sampling_rate, speed_factor)
        return full_waveform, mel_ptr

    def _build_round_robin_chunks(self, inputs, chunk_length_s, stride_length_s, speed_factor=1.0):
        n_files = len(inputs)
        if n_files == 0:
            return [], []
        first = inputs[0]
        if isinstance(first, str) and (n_files == 1 or all(x == first for x in inputs)):
            w0, m0 = self._decode_single_input(first, speed_factor)
            decoded = [(w0, m0)] * n_files
        else:
            prep_futs = [
                self._preprocess_executor.submit(self._decode_single_input, fp, speed_factor)
                for fp in inputs
            ]
            decoded = [f.result() for f in prep_futs]

        chunk_len = round(chunk_length_s * self.feature_extractor.sampling_rate)
        stride_s = stride_length_s if stride_length_s is not None else chunk_length_s / 6.0
        stride = round(stride_s * self.feature_extractor.sampling_rate)
        step = chunk_len - stride * 2

        file_chunks = [[] for _ in range(n_files)]
        all_jobs = []
        active_files = []

        for i, (full_waveform, mel_ptr) in enumerate(decoded):
            total_len = full_waveform.shape[0]
            if total_len == 0:
                continue
            waveform = full_waveform[:chunk_len]
            stride_info = (len(waveform), 0, stride if len(waveform) == chunk_len else 0)
            c_idx = len(file_chunks[i])
            file_chunks[i].append(None)
            all_jobs.append((waveform, mel_ptr if total_len <= chunk_len else 0, stride_info, i, c_idx))
            if chunk_len < total_len:
                active_files.append([i, full_waveform, total_len, chunk_len])

        while active_files:
            next_active = []
            for entry in active_files:
                i, full_waveform, total_len, pos = entry
                chunk_start = pos - stride * 2
                chunk_end = pos + step
                waveform = full_waveform[chunk_start:chunk_end]
                if len(waveform) == 0:
                    continue
                stride_info = (len(waveform), stride, stride if len(waveform) == chunk_len else 0)
                c_idx = len(file_chunks[i])
                file_chunks[i].append(None)
                all_jobs.append((waveform, 0, stride_info, i, c_idx))
                if chunk_end < total_len:
                    entry[3] = chunk_end
                    next_active.append(entry)
            active_files = next_active

        return all_jobs, file_chunks

    def _execute_model(self, input_data, attention_mask, language=None, task=None, return_timestamps=False, max_length=None):
        forced_decoder_ids = self.get_forced_decoder_ids(
            language=language, task=task, return_timestamps=return_timestamps
        )
        effective_max_length = max_length or self.max_length
        
        sharded_data = shard(input_data)
        sharded_mask = shard(attention_mask) if attention_mask is not None else None
        
        output_ids = self.p_generate(
            self.params,
            sharded_data,
            sharded_mask,
            forced_decoder_ids,
            return_timestamps,
            effective_max_length,
            self.use_tpu_features
        )
        return output_ids.reshape(-1, effective_max_length)

    def get_forced_decoder_ids(self, task=None, language=None, return_timestamps=False):
        g = self.model.generation_config
        forced_decoder_ids = []
        if hasattr(g, "lang_to_id") and g.lang_to_id is not None:
            if language is not None:
                lang = language.lower()
                token_str = TO_LANGUAGE_CODE.get(lang, lang)
                token_key = f"<|{token_str}|>"
                if token_key in g.lang_to_id:
                    forced_decoder_ids.append((1, g.lang_to_id[token_key]))
                else:
                    raise ValueError(f"Unsupported language: {language}")
            task = task if task is not None else "transcribe"
            forced_decoder_ids.append((2, g.task_to_id[task]))
        if not return_timestamps:
            idx = forced_decoder_ids[-1][0] + 1 if forced_decoder_ids else 1
            forced_decoder_ids.append((idx, g.no_timestamps_token_id))
        return tuple(forced_decoder_ids)

    def _final_decode(self, unpacked_outputs, return_timestamps, return_language):
        time_precision = self.feature_extractor.chunk_length / self.model.config.max_source_positions

        if return_timestamps:
            text, optional = self.tokenizer._decode_asr(
                unpacked_outputs,
                return_timestamps=return_timestamps,
                return_language=return_language,
                time_precision=time_precision,
            )
            return {"text": text, **optional}
        else:
            if not unpacked_outputs:
                return {"text": ""}
            if isinstance(unpacked_outputs[0], str):
                return {"text": " ".join(unpacked_outputs)}
            if "text" in unpacked_outputs[0]:
                return {"text": " ".join(o["text"] for o in unpacked_outputs)}
            batch_token_ids = [output_dict["tokens"].flatten().tolist() for output_dict in unpacked_outputs]
            decoded_texts = self.processor.batch_decode(batch_token_ids, skip_special_tokens=self.skip_special_tokens)
            return {"text": " ".join(decoded_texts)}

    def _partial_postprocess(self, pred_ids, stride):
        out = {"tokens": pred_ids[None, :]}
        if stride:
            sr = self.feature_extractor.sampling_rate
            out["stride"] = (stride[0] / sr, stride[1] / sr, stride[2] / sr)
        return out

    def _decode_and_resolve_chunk(self, pred_ids_cpu, chunk, actual_chunk_size, return_timestamps, file_chunks):
        pad_id = self.model.config.pad_token_id
        try:
            if not return_timestamps:
                valid_ids = pred_ids_cpu[:actual_chunk_size]
                if self.skip_special_tokens and valid_ids.size > 0:
                    non_pad_cols = np.where(np.any(valid_ids != pad_id, axis=0))[0]
                    max_col = int(non_pad_cols[-1]) + 1 if len(non_pad_cols) > 0 else 1
                    valid_ids_trimmed = valid_ids[:, :max_col]
                else:
                    valid_ids_trimmed = valid_ids
                decoded_texts = self.processor.batch_decode(
                    valid_ids_trimmed, skip_special_tokens=self.skip_special_tokens
                )
                for i in range(actual_chunk_size):
                    _, _, _, f_idx, c_idx = chunk[i]
                    file_chunks[f_idx][c_idx] = decoded_texts[i]
            else:
                for i in range(actual_chunk_size):
                    _, _, stride_info, f_idx, c_idx = chunk[i]
                    file_chunks[f_idx][c_idx] = self._partial_postprocess(pred_ids_cpu[i], stride_info)
        except Exception as e:
            logger.error(f"Result decoding failed: {e}", exc_info=True)

    def _result_fetcher_worker(self, return_timestamps=False, file_chunks=None):
        decode_futs = []
        while True:
            item = self.fetch_queue.get()
            if item is None:
                break
            
            pred_ids_on_device, chunk, actual_chunk_size = item
            
            try:
                pred_ids_cpu = np.asarray(jax.device_get(pred_ids_on_device))
                if pred_ids_cpu.ndim > 2:
                    pred_ids_cpu = pred_ids_cpu.reshape(-1, pred_ids_cpu.shape[-1])
                decode_futs.append(
                    self._preprocess_executor.submit(
                        self._decode_and_resolve_chunk,
                        pred_ids_cpu,
                        chunk,
                        actual_chunk_size,
                        return_timestamps,
                        file_chunks,
                    )
                )
            except Exception as e:
                logger.error(f"Result fetching failed: {e}", exc_info=True)
        if decode_futs:
            wait(decode_futs)

    @staticmethod
    def _generate_fn(params, input_data, attention_mask, forced_decoder_ids, return_timestamps, max_length, use_tpu_features, model, feature_extractor):
        if use_tpu_features:
            input_features = jax.vmap(FlaxWhisperPmapPipeline._jax_feature_extractor, in_axes=(0, None, None, None, None, None, None, None, None))(
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

    # FIX: Use Librosa filters for correctness (Slaney)
    @staticmethod
    def _mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
        return jnp.asarray(_get_mel_filterbank_T(sr, n_fft, n_mels).T)

    @staticmethod
    def _jax_feature_extractor(waveform, sr, n_fft, hop_length, win_length, n_mels, fmin, fmax, target_feature_length):
        window = jnp.asarray(_HANNING_WINDOW_400) if win_length == 400 else jnp.hanning(win_length)
        stft_matrix = FlaxWhisperPmapPipeline._stft(waveform, n_fft, hop_length, win_length, window)
        power_spectrogram = jnp.square(stft_matrix.real) + jnp.square(stft_matrix.imag)
        
        mel_filters_T = jnp.asarray(_get_mel_filterbank_T(sr, n_fft, n_mels))
        mel_spectrogram = jnp.dot(power_spectrogram[:target_feature_length, :], mel_filters_T)
        
        # FIX: Use Clamping Normalization (Whisper Standard) instead of Instance Norm
        log_mel_spectrogram = jnp.log10(jnp.maximum(mel_spectrogram, 1e-10))
        max_val = jnp.max(log_mel_spectrogram)
        log_mel_spectrogram = jnp.maximum(log_mel_spectrogram, max_val - 8.0)
        log_mel_spectrogram = (log_mel_spectrogram + 4.0) * 0.25
        
        return log_mel_spectrogram.T

def create_pipeline(mode="pmap", **kwargs):
    if mode == "pmap":
        return FlaxWhisperPmapPipeline(**kwargs)
    else:
        raise ValueError(f"Mode {mode} not supported in this version.")