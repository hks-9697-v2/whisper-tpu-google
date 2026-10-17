# Architecture

![Architecture](docs/architecture.svg)

This document walks through the diagram above: how audio enters the repo, which pipeline handles it, what runs on the host CPU versus the TPU, and how the online pipeline does continuous batching.

## 1. Package Layout

```text
whisper_jax/
├── pipelines/      # short_audio.py, long_audio.py, long_audio_seek.py, online.py
├── models/         # whisper.py: FlaxWhisperForConditionalGeneration, fused encode/decode loops
├── kernels/        # Pallas TPU kernels: flywheel/, splash_attention/, ragged_paged_attention/
├── inputs/         # audio_loader.py (ctypes bindings, ring buffers) + rust_audio/ (FFmpeg + Mel in Rust/C)
├── layers/         # Flax building blocks
└── distributed/    # PjitPartitioner, InferenceState
server/main.py      # FastAPI server on top of the online pipeline
benchmarks/         # libri-short.py, libri-long.py, test_api*.py, verification scripts
config.yml          # Runtime settings shared by all pipelines
```

Legacy import paths (`whisper_jax.pipeline`, `whisper_jax.pipeline_long_audio`, `whisper_jax.pipeline_online`, `whisper_jax.modeling_flax_whisper`) are aliased in `whisper_jax/__init__.py`.

## 2. Pipelines

| Pipeline | Use case | How it works |
| :--- | :--- | :--- |
| `short_audio.py` | Offline batches of short clips | Decodes files in parallel, pads batches to a bucket size, runs `jax.pmap`. Clips over 30 s are split into chunks and the chunk texts are joined. |
| `long_audio.py` | Offline long-form audio | Decodes each file in full, splits it into overlapping 30 s chunks (`chunk_length_s`, `stride_length_s`), interleaves chunks from all files into shared batches, then merges each file's chunks back together. |
| `long_audio_seek.py` | Offline long-form audio | Gets each file's duration with `ffprobe`, then runs one `ffmpeg -ss` per chunk on a pool of worker threads. |
| `online.py` | Serving (used by `server/main.py`) | Asynchronous requests with continuous batching. See section 5. |

All pipelines pad batches to sizes from `batch_buckets` (default `[4, 40, 80]`; the online pipeline uses `[8, 40, 80]` on 8-chip v7x) and compile each bucket once at startup. That keeps the number of XLA programs small and avoids recompiling while requests are being served.

## 3. Host CPU: Decode, Features, Staging

Implemented in `whisper_jax/inputs/` with native Rust and C:

1. **Decode:** in-process FFmpeg (libav) turns audio bytes or files into 16 kHz float32. If `speed_factor` is not 1.0, an `atempo` filter speeds the audio up first.
2. **Log-Mel features** (`feature_extraction_backend: "cpu"`, the default): C RFFT and a Slaney Mel filterbank produce `bfloat16` features of shape `(B, 3000, 128)`. Models with 80 Mel bins are padded to 128 lanes and sliced back in the encoder. With `"tpu"`, the raw waveform is sent to the TPU and the STFT and Mel run inside the compiled JAX function.
3. **Batch staging:** rows are written into pre-allocated ring buffers for each bucket size, so a new batch does not allocate memory.

## 4. TPU: Encoder and Decoder

Implemented in `whisper_jax/models/whisper.py` and `whisper_jax/kernels/`. The batch is split across chips with `jax.pmap` (4 chips on v6e-4, 8 on v7x-8).

- **Encoder:** self-attention uses the FlyWheel Pallas flash attention kernel by default (`encoder_attention_implementation`), or Splash. Block sizes are picked per chip generation (v6e vs v7x).
- **Decoder (greedy, KV cache):**
  - *Offline pipelines:* a fused "Mega-Decoder" loop whose KV cache grows in stages, 64 → 128 → 512 tokens, so short outputs never pay for a full-length cache. Other decoder kernels can be selected with `decoder_attention_implementation` (`flywheel`, `flywheel_paged`, `rpa`).
  - *Online pipeline:* `pipeline_quantum_step` decodes in 64-token quanta (section 5).
- **Forced tokens:** language and task tokens are passed as a per-row `(B, 4)` `int32` array, not as compile-time constants. Requests with different languages or tasks can therefore share a batch without recompiling.

Token ids are then copied back to the host, decoded with the tokenizer, and chunk texts are joined.

## 5. Online Pipeline: Continuous Batching

`whisper_jax/pipelines/online.py`, served by `server/main.py`.

```text
HTTP request ──► preprocess thread pool ──► request_queue ─┐
                 (decode + Mel, split >30 s)                ├─► batcher ──► one quantum on TPU ──► result fetcher
                                            active_queue ◄──┘      ▲                                   │
                                                 ▲                 │                                   ├─► finished: text → Future
                                                 └─────────────────┴──── not finished ◄────────────────┘
```

1. **Ingest:** `pipeline(audio_bytes, language=..., task=...)` returns a `concurrent.futures.Future`. `server/main.py` awaits it with `asyncio.wrap_future`. `/transcribe_stream` collects a chunked upload and then follows the same path.
2. **Preprocess:** a thread pool (`preprocessing_workers`) decodes the audio and computes Mel features in Rust. Audio over `chunk_length_s` is split into chunks. Each chunk goes into `request_queue` with `cur_len = 1`.
3. **Batch:** the batcher fills a batch from `active_queue` first (unfinished requests), then from `request_queue`, and pads it to a bucket size. It waits up to `batch_assembly_timeout_s` for a fuller batch, and dispatches early when nothing else is pending.
4. **One quantum on TPU (`p_quantum`):** every row decodes up to 64 more tokens.
   - If every row is new, a fast path decodes tokens 0–64 with no extra setup.
   - If any row is a continuing request, the step first rebuilds each row's KV cache from its existing tokens in one parallel pass. It then decodes with per-row lengths, positions and causal masks.
5. **Fetch:** a fetcher thread copies `seqs`, `cur_len` and `is_finished` to the host. Finished rows are decoded to text and their Futures are resolved. Unfinished rows go back to `active_queue` with their tokens so far, and join the next batch alongside newly arrived requests.

Up to two quanta can be in flight at once, so the host can stage the next batch while the TPU runs. A partial batch waits for the in-flight quantum to finish, so returning requests and new arrivals go out together.

This gives you:

- Short requests return after their first quantum instead of waiting for the longest request in their batch.
- Long requests keep decoding in later quanta, sharing batches with new requests.
- Mixed languages and tasks share one batch with no recompilation.

Continuous batching is used when the decoder head dimension is 64, `decoder_attention_implementation` is `"original"`, and `max_length > 64`. Requests with `return_timestamps=True` use the regular single-call generate path.

## 6. Startup

- **Weights:** checkpoints without Flax weights are converted from PyTorch on load. Parameters are repacked into a fused layout (`prepare_fast_params`) and replicated across chips.
- **Warm-up:** every bucket size is compiled once with a dummy batch before the pipeline accepts requests. `server/main.py` reports `/ready` only after warm-up finishes. The server model can be set with `WHISPER_MODEL_ID` and the compile cache directory with `JAX_COMPILATION_CACHE_DIR`.

## 7. Configuration

`config.yml` (constructor arguments can override these):

| Key | Effect |
| :--- | :--- |
| `encoder_attention_implementation` | `flywheel` (default), `splash`, `original` |
| `decoder_attention_implementation` | `original` (Mega-Decoder, default), `flywheel`, `flywheel_paged`, `rpa` |
| `feature_extraction_backend` | `cpu` (default) or `tpu` |
| `batch_buckets` | Padded batch sizes compiled at startup |
| `chunk_length_s`, `stride_length_s` | Chunking for audio longer than 30 s |
| `speed_factor` | Optional audio speed-up before transcription |
| `batch_assembly_timeout_s` | How long the batcher waits to fill a batch |
| `generation.max_length` | Maximum tokens per chunk |
