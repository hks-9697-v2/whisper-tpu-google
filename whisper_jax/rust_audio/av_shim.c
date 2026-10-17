// In-process libavformat + libavcodec + libavfilter (atempo) shim for Rust audio engine.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavcodec/avcodec.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/opt.h>
#include <libavutil/channel_layout.h>
#include <libavutil/samplefmt.h>
#include <libavutil/mem.h>
#include <libavutil/version.h>

#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(57, 28, 100)
#define HAVE_NEW_CH_LAYOUT 1
#else
#define HAVE_NEW_CH_LAYOUT 0
#endif

static pthread_once_t g_av_init_once = PTHREAD_ONCE_INIT;
static void init_av_logging(void) {
    av_log_set_level(AV_LOG_QUIET);
}

typedef struct {
    float* data;
    size_t len;
    size_t cap;
} FloatBuffer;

static int fb_append(FloatBuffer* fb, const float* src, size_t count) {
    if (fb->len + count > fb->cap) {
        size_t new_cap = fb->cap ? fb->cap * 2 : 65536;
        while (new_cap < fb->len + count) {
            new_cap *= 2;
        }
        float* new_data = (float*)realloc(fb->data, new_cap * sizeof(float));
        if (!new_data) return -1;
        fb->data = new_data;
        fb->cap = new_cap;
    }
    memcpy(fb->data + fb->len, src, count * sizeof(float));
    fb->len += count;
    return 0;
}

// Build an in-process filter graph: abuffer -> [atempo=...] -> aformat=flt:16000:mono -> abuffersink
static int build_filter_graph(
    AVFilterGraph** out_graph,
    AVFilterContext** out_src,
    AVFilterContext** out_sink,
    int sample_rate,
    enum AVSampleFormat sample_fmt,
    int nb_channels,
    int target_sr,
    double speed_factor
) {
    AVFilterGraph* graph = avfilter_graph_alloc();
    if (!graph) return -1;
    graph->nb_threads = 1;

    const char* ch_buf = (nb_channels == 2) ? "stereo" : "mono";

    char args[512];
    snprintf(
        args, sizeof(args),
        "time_base=1/%d:sample_rate=%d:sample_fmt=%s:channel_layout=%s",
        sample_rate, sample_rate, av_get_sample_fmt_name(sample_fmt), ch_buf
    );

    const AVFilter* abuffer = avfilter_get_by_name("abuffer");
    const AVFilter* abuffersink = avfilter_get_by_name("abuffersink");
    const AVFilter* aformat = avfilter_get_by_name("aformat");

    AVFilterContext* src_ctx = NULL;
    AVFilterContext* sink_ctx = NULL;
    AVFilterContext* fmt_ctx = NULL;

    if (avfilter_graph_create_filter(&src_ctx, abuffer, "in", args, NULL, graph) < 0) goto fail;
    if (avfilter_graph_create_filter(&sink_ctx, abuffersink, "out", NULL, NULL, graph) < 0) goto fail;

    char fmt_args[256];
    snprintf(fmt_args, sizeof(fmt_args), "sample_fmts=flt:sample_rates=%d:channel_layouts=mono", target_sr);
    if (avfilter_graph_create_filter(&fmt_ctx, aformat, "fmt", fmt_args, NULL, graph) < 0) goto fail;

    AVFilterContext* last_ctx = src_ctx;
    if (fabs(speed_factor - 1.0) > 1e-6) {
        const AVFilter* atempo = avfilter_get_by_name("atempo");
        AVFilterContext* atempo_ctx = NULL;
        char atempo_args[64];
        snprintf(atempo_args, sizeof(atempo_args), "tempo=%g", speed_factor);
        if (avfilter_graph_create_filter(&atempo_ctx, atempo, "atempo", atempo_args, NULL, graph) < 0) goto fail;
        if (avfilter_link(last_ctx, 0, atempo_ctx, 0) < 0) goto fail;
        last_ctx = atempo_ctx;
    }

    if (avfilter_link(last_ctx, 0, fmt_ctx, 0) < 0) goto fail;
    if (avfilter_link(fmt_ctx, 0, sink_ctx, 0) < 0) goto fail;
    if (avfilter_graph_config(graph, NULL) < 0) goto fail;

    *out_graph = graph;
    *out_src = src_ctx;
    *out_sink = sink_ctx;
    return 0;

fail:
    avfilter_graph_free(&graph);
    return -1;
}

// Fast path for raw PCM s16le mono buffer through libavfilter
static int filter_s16le_segment(
    const int16_t* pcm,
    size_t num_samples,
    int sample_rate,
    int target_sr,
    double speed_factor,
    FloatBuffer* out_fb
) {
    if (fabs(speed_factor - 1.0) <= 1e-6 && sample_rate == target_sr) {
        out_fb->data = (float*)malloc(num_samples * sizeof(float));
        if (!out_fb->data) return -1;
        out_fb->len = num_samples;
        out_fb->cap = num_samples;
        const float scale = 1.0f / 32768.0f;
        for (size_t i = 0; i < num_samples; ++i) {
            out_fb->data[i] = (float)pcm[i] * scale;
        }
        return 0;
    }

    AVFilterGraph* graph = NULL;
    AVFilterContext* src_ctx = NULL;
    AVFilterContext* sink_ctx = NULL;

    if (build_filter_graph(&graph, &src_ctx, &sink_ctx, sample_rate, AV_SAMPLE_FMT_S16, 1, target_sr, speed_factor) < 0) {
        return -1;
    }

    AVFrame* in_frame = av_frame_alloc();
    AVFrame* filt_frame = av_frame_alloc();
    if (!in_frame || !filt_frame) {
        av_frame_free(&in_frame);
        av_frame_free(&filt_frame);
        avfilter_graph_free(&graph);
        return -1;
    }

    const size_t frame_samples = 2048;
    size_t offset = 0;
    int64_t pts = 0;

    while (offset < num_samples) {
        size_t cur = num_samples - offset;
        if (cur > frame_samples) cur = frame_samples;

        in_frame->nb_samples = (int)cur;
        in_frame->format = AV_SAMPLE_FMT_S16;
#if HAVE_NEW_CH_LAYOUT
        av_channel_layout_default(&in_frame->ch_layout, 1);
#else
        in_frame->channel_layout = AV_CH_LAYOUT_MONO;
        in_frame->channels = 1;
#endif
        in_frame->sample_rate = sample_rate;
        in_frame->pts = pts;
        pts += (int64_t)cur;

        if (av_frame_get_buffer(in_frame, 0) < 0) break;
        memcpy(in_frame->data[0], pcm + offset, cur * sizeof(int16_t));
        offset += cur;

        if (av_buffersrc_add_frame_flags(src_ctx, in_frame, AV_BUFFERSRC_FLAG_KEEP_REF) < 0) {
            av_frame_unref(in_frame);
            break;
        }
        av_frame_unref(in_frame);

        while (av_buffersink_get_frame(sink_ctx, filt_frame) >= 0) {
            fb_append(out_fb, (const float*)filt_frame->data[0], (size_t)filt_frame->nb_samples);
            av_frame_unref(filt_frame);
        }
    }

    if (av_buffersrc_add_frame_flags(src_ctx, NULL, 0) >= 0) {
        while (av_buffersink_get_frame(sink_ctx, filt_frame) >= 0) {
            fb_append(out_fb, (const float*)filt_frame->data[0], (size_t)filt_frame->nb_samples);
            av_frame_unref(filt_frame);
        }
    }

    av_frame_free(&in_frame);
    av_frame_free(&filt_frame);
    avfilter_graph_free(&graph);
    return 0;
}

typedef struct {
    const int16_t* pcm;
    size_t num_samples;
    int sample_rate;
    int target_sr;
    double speed_factor;
    size_t expected_out_samples; // 0 for last segment
    FloatBuffer fb;
    int status;
} SegmentTask;

static void* segment_worker(void* arg) {
    SegmentTask* t = (SegmentTask*)arg;
    t->fb.data = NULL;
    t->fb.len = 0;
    t->fb.cap = 0;
    t->status = filter_s16le_segment(t->pcm, t->num_samples, t->sample_rate, t->target_sr, t->speed_factor, &t->fb);
    if (t->status == 0 && t->expected_out_samples > 0) {
        if (t->fb.len > t->expected_out_samples) {
            t->fb.len = t->expected_out_samples;
        } else if (t->fb.len < t->expected_out_samples) {
            size_t pad = t->expected_out_samples - t->fb.len;
            float* zeros = (float*)calloc(pad, sizeof(float));
            if (zeros) {
                fb_append(&t->fb, zeros, pad);
                free(zeros);
            }
        }
    }
    return NULL;
}

typedef struct {
    int mode; // 1 = mel_bf16, 2 = audio_f32, 3 = pcm_s16_to_f32
    const uint16_t* const* mel_ptrs;
    const float* const* audio_ptrs;
    const size_t* audio_lens;
    const int16_t* pcm_src;
    size_t actual_batch_size;
    size_t total_rows;
    size_t max_audio_len;
    void* dst_ptr;
} BatchJob;

static void run_batch_job_range(const BatchJob* job, size_t start_row, size_t end_row);
static void dispatch_batch_job(const BatchJob* job);

// Check if a file is a standard 16-bit mono WAV and parse its PCM data directly
static int try_fast_wav_s16le(
    const char* path,
    int target_sr,
    double speed_factor,
    float** out_samples,
    size_t* out_len
) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;

    uint8_t hdr[12];
    if (fread(hdr, 1, 12, f) != 12 || memcmp(hdr, "RIFF", 4) != 0 || memcmp(hdr + 8, "WAVE", 4) != 0) {
        fclose(f);
        return -1;
    }

    uint16_t audio_fmt = 0, num_ch = 0, bits_per_sample = 0;
    uint32_t sample_rate = 0;
    size_t data_bytes = 0;
    long data_offset = 0;

    while (!feof(f)) {
        uint8_t chdr[8];
        if (fread(chdr, 1, 8, f) != 8) break;
        uint32_t csize = *(uint32_t*)(chdr + 4);
        if (memcmp(chdr, "fmt ", 4) == 0) {
            uint8_t fmt[16];
            if (csize < 16 || fread(fmt, 1, 16, f) != 16) {
                fclose(f);
                return -1;
            }
            audio_fmt = *(uint16_t*)(fmt + 0);
            num_ch = *(uint16_t*)(fmt + 2);
            sample_rate = *(uint32_t*)(fmt + 4);
            bits_per_sample = *(uint16_t*)(fmt + 14);
            if (csize > 16) fseek(f, csize - 16, SEEK_CUR);
        } else if (memcmp(chdr, "data", 4) == 0) {
            data_offset = ftell(f);
            data_bytes = csize;
            break;
        } else {
            fseek(f, csize + (csize & 1), SEEK_CUR);
        }
    }

    if (audio_fmt != 1 || num_ch != 1 || bits_per_sample != 16 || data_bytes == 0 || sample_rate == 0) {
        fclose(f);
        return -1;
    }

    if (fabs(speed_factor - 1.0) <= 1e-6 && (int)sample_rate == target_sr) {
        struct stat st;
        int fd = fileno(f);
        if (fd >= 0 && fstat(fd, &st) == 0 && (size_t)st.st_size > (size_t)data_offset) {
            size_t avail = (size_t)st.st_size - (size_t)data_offset;
            size_t nbytes = data_bytes < avail ? data_bytes : avail;
            size_t total_samples = nbytes / sizeof(int16_t);
            void* map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (map != MAP_FAILED) {
                const int16_t* pcm_map = (const int16_t*)((const uint8_t*)map + data_offset);
                float* out = (float*)malloc(total_samples * sizeof(float));
                if (!out) {
                    munmap(map, (size_t)st.st_size);
                    fclose(f);
                    return -1;
                }
                BatchJob job;
                memset(&job, 0, sizeof(job));
                job.mode = 3;
                job.pcm_src = pcm_map;
                job.total_rows = total_samples;
                job.dst_ptr = out;
                if (total_samples >= 262144) {
                    dispatch_batch_job(&job);
                } else {
                    run_batch_job_range(&job, 0, total_samples);
                }
                munmap(map, (size_t)st.st_size);
                fclose(f);
                *out_samples = out;
                *out_len = total_samples;
                return 0;
            }
        }
    }

    int16_t* pcm = (int16_t*)malloc(data_bytes);
    if (!pcm) {
        fclose(f);
        return -1;
    }
    fseek(f, data_offset, SEEK_SET);
    size_t nread = fread(pcm, 1, data_bytes, f);
    fclose(f);

    size_t total_samples = nread / sizeof(int16_t);
    if (fabs(speed_factor - 1.0) <= 1e-6 && (int)sample_rate == target_sr) {
        FloatBuffer fb = {NULL, 0, 0};
        int rc = filter_s16le_segment(pcm, total_samples, (int)sample_rate, target_sr, speed_factor, &fb);
        free(pcm);
        if (rc != 0) return -1;
        *out_samples = fb.data;
        *out_len = fb.len;
        return 0;
    }
    // Chunk boundary alignment: 6 * 30s Whisper chunks = 6 * 480000 = 2,880,000 output samples
    size_t out_chunk_samples = 6 * 30 * (size_t)target_sr;
    size_t in_chunk_samples = (size_t)llround((double)out_chunk_samples * speed_factor * ((double)sample_rate / (double)target_sr));

    if (total_samples <= in_chunk_samples * 2) {
        FloatBuffer fb = {NULL, 0, 0};
        int rc = filter_s16le_segment(pcm, total_samples, (int)sample_rate, target_sr, speed_factor, &fb);
        free(pcm);
        if (rc != 0) return -1;
        *out_samples = fb.data;
        *out_len = fb.len;
        return 0;
    }

    // Parallel multi-threaded segment filtering for long WAV files (e.g. 1.55h videoplayback.wav)
    size_t num_segs = (total_samples + in_chunk_samples - 1) / in_chunk_samples;
    SegmentTask* tasks = (SegmentTask*)calloc(num_segs, sizeof(SegmentTask));
    pthread_t* threads = (pthread_t*)calloc(num_segs, sizeof(pthread_t));

    for (size_t s = 0; s < num_segs; ++s) {
        size_t start = s * in_chunk_samples;
        size_t rem = total_samples - start;
        size_t seg_len = rem < in_chunk_samples ? rem : in_chunk_samples;
        tasks[s].pcm = pcm + start;
        tasks[s].num_samples = seg_len;
        tasks[s].sample_rate = (int)sample_rate;
        tasks[s].target_sr = target_sr;
        tasks[s].speed_factor = speed_factor;
        tasks[s].expected_out_samples = (s + 1 < num_segs) ? out_chunk_samples : 0;
        pthread_create(&threads[s], NULL, segment_worker, &tasks[s]);
    }

    size_t total_out = 0;
    int any_err = 0;
    for (size_t s = 0; s < num_segs; ++s) {
        pthread_join(threads[s], NULL);
        if (tasks[s].status != 0) any_err = 1;
        total_out += tasks[s].fb.len;
    }
    free(pcm);
    free(threads);

    if (any_err) {
        for (size_t s = 0; s < num_segs; ++s) free(tasks[s].fb.data);
        free(tasks);
        return -1;
    }

    float* merged = (float*)malloc(total_out * sizeof(float));
    size_t pos = 0;
    for (size_t s = 0; s < num_segs; ++s) {
        if (tasks[s].fb.len > 0) {
            memcpy(merged + pos, tasks[s].fb.data, tasks[s].fb.len * sizeof(float));
            pos += tasks[s].fb.len;
        }
        free(tasks[s].fb.data);
    }
    free(tasks);

    *out_samples = merged;
    *out_len = total_out;
    return 0;
}

static int decode_fmt_ctx_and_filter(
    AVFormatContext* fmt_ctx,
    int target_sr,
    double speed_factor,
    float** out_samples,
    size_t* out_len
) {
    if (avformat_find_stream_info(fmt_ctx, NULL) < 0) return -1;

    const AVCodec* dec = NULL;
    int stream_idx = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, &dec, 0);
    if (stream_idx < 0 || !dec) return -1;

    AVCodecContext* dec_ctx = avcodec_alloc_context3(dec);
    if (!dec_ctx) return -1;
    if (avcodec_parameters_to_context(dec_ctx, fmt_ctx->streams[stream_idx]->codecpar) < 0) {
        avcodec_free_context(&dec_ctx);
        return -1;
    }
    if (avcodec_open2(dec_ctx, dec, NULL) < 0) {
        avcodec_free_context(&dec_ctx);
        return -1;
    }

#if HAVE_NEW_CH_LAYOUT
    if (dec_ctx->ch_layout.nb_channels == 0) {
        av_channel_layout_default(&dec_ctx->ch_layout, 1);
    }
    int nb_channels = dec_ctx->ch_layout.nb_channels;
#else
    if (dec_ctx->channel_layout == 0) {
        dec_ctx->channel_layout = av_get_default_channel_layout(dec_ctx->channels > 0 ? dec_ctx->channels : 1);
    }
    int nb_channels = dec_ctx->channels > 0 ? dec_ctx->channels : 1;
#endif

    AVFilterGraph* graph = NULL;
    AVFilterContext* src_ctx = NULL;
    AVFilterContext* sink_ctx = NULL;
    if (build_filter_graph(&graph, &src_ctx, &sink_ctx, dec_ctx->sample_rate, dec_ctx->sample_fmt, nb_channels, target_sr, speed_factor) < 0) {
        avcodec_free_context(&dec_ctx);
        return -1;
    }

    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    AVFrame* filt_frame = av_frame_alloc();
    FloatBuffer fb = {NULL, 0, 0};

    while (av_read_frame(fmt_ctx, pkt) >= 0) {
        if (pkt->stream_index == stream_idx) {
            if (avcodec_send_packet(dec_ctx, pkt) >= 0) {
                while (avcodec_receive_frame(dec_ctx, frame) >= 0) {
                    if (av_buffersrc_add_frame_flags(src_ctx, frame, AV_BUFFERSRC_FLAG_KEEP_REF) >= 0) {
                        while (av_buffersink_get_frame(sink_ctx, filt_frame) >= 0) {
                            fb_append(&fb, (const float*)filt_frame->data[0], (size_t)filt_frame->nb_samples);
                            av_frame_unref(filt_frame);
                        }
                    }
                    av_frame_unref(frame);
                }
            }
        }
        av_packet_unref(pkt);
    }

    // Flush decoder
    avcodec_send_packet(dec_ctx, NULL);
    while (avcodec_receive_frame(dec_ctx, frame) >= 0) {
        if (av_buffersrc_add_frame_flags(src_ctx, frame, AV_BUFFERSRC_FLAG_KEEP_REF) >= 0) {
            while (av_buffersink_get_frame(sink_ctx, filt_frame) >= 0) {
                fb_append(&fb, (const float*)filt_frame->data[0], (size_t)filt_frame->nb_samples);
                av_frame_unref(filt_frame);
            }
        }
        av_frame_unref(frame);
    }

    // Flush filter graph
    if (av_buffersrc_add_frame_flags(src_ctx, NULL, 0) >= 0) {
        while (av_buffersink_get_frame(sink_ctx, filt_frame) >= 0) {
            fb_append(&fb, (const float*)filt_frame->data[0], (size_t)filt_frame->nb_samples);
            av_frame_unref(filt_frame);
        }
    }

    av_packet_free(&pkt);
    av_frame_free(&frame);
    av_frame_free(&filt_frame);
    avfilter_graph_free(&graph);
    avcodec_free_context(&dec_ctx);

    *out_samples = fb.data;
    *out_len = fb.len;
    return 0;
}

int av_decode_and_stretch_file(
    const char* path,
    int target_sr,
    double speed_factor,
    float** out_samples,
    size_t* out_len
) {
    pthread_once(&g_av_init_once, init_av_logging);

    if (try_fast_wav_s16le(path, target_sr, speed_factor, out_samples, out_len) == 0) {
        return 0;
    }

    AVFormatContext* fmt_ctx = NULL;
    if (avformat_open_input(&fmt_ctx, path, NULL, NULL) < 0) {
        return -1;
    }
    int rc = decode_fmt_ctx_and_filter(fmt_ctx, target_sr, speed_factor, out_samples, out_len);
    avformat_close_input(&fmt_ctx);
    return rc;
}

typedef struct {
    const uint8_t* ptr;
    size_t size;
    size_t pos;
} MemBuffer;

static int mem_read(void* opaque, uint8_t* buf, int buf_size) {
    MemBuffer* mb = (MemBuffer*)opaque;
    if (mb->pos >= mb->size) return AVERROR_EOF;
    size_t rem = mb->size - mb->pos;
    size_t cpy = (size_t)buf_size < rem ? (size_t)buf_size : rem;
    memcpy(buf, mb->ptr + mb->pos, cpy);
    mb->pos += cpy;
    return (int)cpy;
}

static int64_t mem_seek(void* opaque, int64_t offset, int whence) {
    MemBuffer* mb = (MemBuffer*)opaque;
    if (whence == AVSEEK_SIZE) return (int64_t)mb->size;
    int64_t new_pos = 0;
    if (whence == SEEK_SET) new_pos = offset;
    else if (whence == SEEK_CUR) new_pos = (int64_t)mb->pos + offset;
    else if (whence == SEEK_END) new_pos = (int64_t)mb->size + offset;
    else return -1;
    if (new_pos < 0 || (size_t)new_pos > mb->size) return -1;
    mb->pos = (size_t)new_pos;
    return new_pos;
}

int av_decode_and_stretch_bytes(
    const uint8_t* data,
    size_t data_len,
    int target_sr,
    double speed_factor,
    float** out_samples,
    size_t* out_len
) {
    pthread_once(&g_av_init_once, init_av_logging);

    MemBuffer mb = {data, data_len, 0};
    size_t avio_buf_sz = 32768;
    uint8_t* avio_buf = (uint8_t*)av_malloc(avio_buf_sz);
    if (!avio_buf) return -1;

    AVIOContext* avio_ctx = avio_alloc_context(avio_buf, (int)avio_buf_sz, 0, &mb, &mem_read, NULL, &mem_seek);
    if (!avio_ctx) {
        av_free(avio_buf);
        return -1;
    }

    AVFormatContext* fmt_ctx = avformat_alloc_context();
    fmt_ctx->pb = avio_ctx;
    if (avformat_open_input(&fmt_ctx, NULL, NULL, NULL) < 0) {
        av_freep(&avio_ctx->buffer);
        avio_context_free(&avio_ctx);
        return -1;
    }

    int rc = decode_fmt_ctx_and_filter(fmt_ctx, target_sr, speed_factor, out_samples, out_len);
    avformat_close_input(&fmt_ctx);
    av_freep(&avio_ctx->buffer);
    avio_context_free(&avio_ctx);
    return rc;
}

void av_free_samples(float* ptr) {
    free(ptr);
}

// ============================================================================
// Fast C (3000, 128) bfloat16 Mel-Spectrogram Extractor + Persistent Thread Pool
// ============================================================================

static float g_window[400] __attribute__((aligned(64)));
static float g_cos25[13][25] __attribute__((aligned(64)));
static float g_msin25[13][25] __attribute__((aligned(64)));
static float g_tw400_re[13][16] __attribute__((aligned(64)));
static float g_tw400_im[13][16] __attribute__((aligned(64)));
static float g_tw16_re[4][4] __attribute__((aligned(64)));
static float g_tw16_im[4][4] __attribute__((aligned(64)));

static int g_mel_start[128];
static int g_mel_len[128];
static float g_mel_weights[128][16] __attribute__((aligned(64)));
static int g_mel_initialized = 0;

static inline uint16_t f32_to_bf16(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    u += 0x7fffU + ((u >> 16) & 1U);
    return (uint16_t)(u >> 16);
}

void av_init_mel_tables(const float* window_400, const float* mel_filters_t) {
    memcpy(g_window, window_400, 400 * sizeof(float));
    const double pi = 3.14159265358979323846;
    for (int r = 0; r < 13; ++r) {
        for (int n1 = 0; n1 < 25; ++n1) {
            double ang = 2.0 * pi * (double)(n1 * r) / 25.0;
            g_cos25[r][n1] = (float)cos(ang);
            g_msin25[r][n1] = (float)(-sin(ang));
        }
        for (int n2 = 0; n2 < 16; ++n2) {
            double ang = 2.0 * pi * (double)(r * n2) / 400.0;
            g_tw400_re[r][n2] = (float)cos(ang);
            g_tw400_im[r][n2] = (float)(-sin(ang));
        }
    }
    for (int k1 = 0; k1 < 4; ++k1) {
        for (int m2 = 0; m2 < 4; ++m2) {
            double ang = 2.0 * pi * (double)(k1 * m2) / 16.0;
            g_tw16_re[k1][m2] = (float)cos(ang);
            g_tw16_im[k1][m2] = (float)(-sin(ang));
        }
    }
    for (int m = 0; m < 128; ++m) {
        int start = -1, end = -1;
        for (int f = 0; f < 201; ++f) {
            float w = mel_filters_t[f * 128 + m];
            if (w > 0.0f) {
                if (start == -1) start = f;
                end = f + 1;
            }
        }
        if (start == -1) {
            g_mel_start[m] = 0;
            g_mel_len[m] = 0;
        } else {
            int len = end - start;
            if (len > 16) len = 16;
            g_mel_start[m] = start;
            g_mel_len[m] = len;
            for (int j = 0; j < len; ++j) {
                g_mel_weights[m][j] = mel_filters_t[(start + j) * 128 + m];
            }
        }
    }
    g_mel_initialized = 1;
}

static inline void fft16_power(const float* restrict in_re, const float* restrict in_im, float* restrict out_pow) {
    float a_re[4][4], a_im[4][4];
    for (int m2 = 0; m2 < 4; ++m2) {
        float x0r = in_re[0 + m2], x0i = in_im[0 + m2];
        float x1r = in_re[4 + m2], x1i = in_im[4 + m2];
        float x2r = in_re[8 + m2], x2i = in_im[8 + m2];
        float x3r = in_re[12 + m2], x3i = in_im[12 + m2];

        float s02r = x0r + x2r, s02i = x0i + x2i;
        float d02r = x0r - x2r, d02i = x0i - x2i;
        float s13r = x1r + x3r, s13i = x1i + x3i;
        float d13r = x1r - x3r, d13i = x1i - x3i;

        float t0r = s02r + s13r, t0i = s02i + s13i;
        float t1r = d02r + d13i, t1i = d02i - d13r;
        float t2r = s02r - s13r, t2i = s02i - s13i;
        float t3r = d02r - d13i, t3i = d02i + d13r;

        a_re[0][m2] = t0r; a_im[0][m2] = t0i;
        a_re[1][m2] = t1r * g_tw16_re[1][m2] - t1i * g_tw16_im[1][m2];
        a_im[1][m2] = t1r * g_tw16_im[1][m2] + t1i * g_tw16_re[1][m2];
        a_re[2][m2] = t2r * g_tw16_re[2][m2] - t2i * g_tw16_im[2][m2];
        a_im[2][m2] = t2r * g_tw16_im[2][m2] + t2i * g_tw16_re[2][m2];
        a_re[3][m2] = t3r * g_tw16_re[3][m2] - t3i * g_tw16_im[3][m2];
        a_im[3][m2] = t3r * g_tw16_im[3][m2] + t3i * g_tw16_re[3][m2];
    }
    for (int k1 = 0; k1 < 4; ++k1) {
        float y0r = a_re[k1][0], y0i = a_im[k1][0];
        float y1r = a_re[k1][1], y1i = a_im[k1][1];
        float y2r = a_re[k1][2], y2i = a_im[k1][2];
        float y3r = a_re[k1][3], y3i = a_im[k1][3];

        float s02r = y0r + y2r, s02i = y0i + y2i;
        float d02r = y0r - y2r, d02i = y0i - y2i;
        float s13r = y1r + y3r, s13i = y1i + y3i;
        float d13r = y1r - y3r, d13i = y1i - y3i;

        float z0r = s02r + s13r, z0i = s02i + s13i;
        float z1r = d02r + d13i, z1i = d02i - d13r;
        float z2r = s02r - s13r, z2i = s02i - s13i;
        float z3r = d02r - d13i, z3i = d02i + d13r;

        out_pow[0 + k1] = z0r * z0r + z0i * z0i;
        out_pow[4 + k1] = z1r * z1r + z1i * z1i;
        out_pow[8 + k1] = z2r * z2r + z2i * z2i;
        out_pow[12 + k1] = z3r * z3r + z3i * z3i;
    }
}

void av_compute_mel_3000x128_bf16(const float* samples, size_t n_samples, uint16_t* out_bf16) {
    size_t L = n_samples < 480000 ? n_samples : 480000;
    int t_active = (int)((L + 199) / 160) + 1;
    if (t_active > 3000) t_active = 3000;

    float* lmel_buf = (float*)malloc((size_t)t_active * 128 * sizeof(float));
    if (!lmel_buf) return;

    float frame[400] __attribute__((aligned(64)));
    float power[201] __attribute__((aligned(64)));
    float y_pow[16] __attribute__((aligned(64)));
    float b_re[16] __attribute__((aligned(64)));
    float b_im[16] __attribute__((aligned(64)));

    float max_val = (t_active < 3000) ? -10.0f : -1e30f;

    for (int t = 0; t < t_active; ++t) {
        int base_idx = t * 160 - 200;
        if (base_idx >= 0 && (size_t)(base_idx + 400) <= L) {
            const float* src = samples + base_idx;
            for (int k = 0; k < 400; ++k) {
                frame[k] = src[k] * g_window[k];
            }
        } else {
            for (int k = 0; k < 400; ++k) {
                int idx = base_idx + k;
                if (idx < 0) idx = -idx;
                else if (idx >= 480000) idx = 960000 - 2 - idx;
                float s = ((size_t)idx < L && samples != NULL) ? samples[idx] : 0.0f;
                frame[k] = s * g_window[k];
            }
        }

        for (int n2 = 0; n2 < 16; ++n2) {
            float sum = 0.0f;
            for (int n1 = 0; n1 < 25; ++n1) {
                sum += frame[16 * n1 + n2];
            }
            b_re[n2] = sum;
            b_im[n2] = 0.0f;
        }
        fft16_power(b_re, b_im, y_pow);
        for (int m = 0; m < 9; ++m) {
            power[25 * m] = y_pow[m];
        }

        for (int r = 1; r < 13; ++r) {
            const float* c25 = g_cos25[r];
            const float* s25 = g_msin25[r];
            for (int n2 = 0; n2 < 16; ++n2) {
                b_re[n2] = frame[n2];
                b_im[n2] = 0.0f;
            }
            for (int n1 = 1; n1 < 25; ++n1) {
                float c = c25[n1];
                float s = s25[n1];
                const float* fr = frame + 16 * n1;
                for (int n2 = 0; n2 < 16; ++n2) {
                    b_re[n2] += fr[n2] * c;
                    b_im[n2] += fr[n2] * s;
                }
            }
            const float* twr = g_tw400_re[r];
            const float* twi = g_tw400_im[r];
            for (int n2 = 0; n2 < 16; ++n2) {
                float re = b_re[n2], im = b_im[n2];
                b_re[n2] = re * twr[n2] - im * twi[n2];
                b_im[n2] = re * twi[n2] + im * twr[n2];
            }
            fft16_power(b_re, b_im, y_pow);
            for (int m = 0; m < 8; ++m) {
                power[r + 25 * m] = y_pow[m];
            }
            for (int m = 8; m < 16; ++m) {
                power[400 - (r + 25 * m)] = y_pow[m];
            }
        }

        float* lmel_row = lmel_buf + (size_t)t * 128;
        for (int m = 0; m < 128; ++m) {
            int st = g_mel_start[m];
            int ln = g_mel_len[m];
            const float* w = g_mel_weights[m];
            float sum = 0.0f;
            for (int j = 0; j < ln; ++j) {
                sum += power[st + j] * w[j];
            }
            if (sum < 1e-10f) sum = 1e-10f;
            float lmel = log10f(sum);
            lmel_row[m] = lmel;
            if (lmel > max_val) max_val = lmel;
        }
    }

    float low_clamp = max_val - 8.0f;
    float pad_val = ((low_clamp > -10.0f ? low_clamp : -10.0f) + 4.0f) * 0.25f;
    uint16_t pad_bf16 = f32_to_bf16(pad_val);

    int active_elems = t_active * 128;
    for (int i = 0; i < active_elems; ++i) {
        float v = lmel_buf[i];
        if (v < low_clamp) v = low_clamp;
        v = (v + 4.0f) * 0.25f;
        out_bf16[i] = f32_to_bf16(v);
    }
    int total_elems = 3000 * 128;
    for (int i = active_elems; i < total_elems; ++i) {
        out_bf16[i] = pad_bf16;
    }
    free(lmel_buf);
}

#define NUM_POOL_WORKERS 16

static pthread_mutex_t g_dispatch_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_pool_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_pool_work_cv = PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_pool_done_cv = PTHREAD_COND_INITIALIZER;
static pthread_once_t g_pool_once = PTHREAD_ONCE_INIT;
static uint64_t g_pool_gen = 0;
static int g_pool_pending = 0;
static BatchJob g_pool_job;

static void run_batch_job_range(const BatchJob* job, size_t start_row, size_t end_row) {
    if (job->mode == 1) {
        const size_t row_elems = 3000 * 128;
        uint16_t* base = (uint16_t*)job->dst_ptr;
        for (size_t i = start_row; i < end_row; ++i) {
            uint16_t* row_dst = base + i * row_elems;
            if (i < job->actual_batch_size) {
                const uint16_t* m_ptr = job->mel_ptrs ? job->mel_ptrs[i] : NULL;
                if (m_ptr != NULL) {
                    memcpy(row_dst, m_ptr, row_elems * sizeof(uint16_t));
                } else {
                    const float* a_ptr = job->audio_ptrs ? job->audio_ptrs[i] : NULL;
                    size_t a_len = job->audio_lens ? job->audio_lens[i] : 0;
                    av_compute_mel_3000x128_bf16(a_ptr, a_len, row_dst);
                }
            } else {
                memset(row_dst, 0, row_elems * sizeof(uint16_t));
            }
        }
    } else if (job->mode == 2) {
        const size_t max_len = job->max_audio_len;
        float* base = (float*)job->dst_ptr;
        for (size_t i = start_row; i < end_row; ++i) {
            float* row_dst = base + i * max_len;
            if (i < job->actual_batch_size) {
                const float* a_ptr = job->audio_ptrs[i];
                size_t copy_len = job->audio_lens[i];
                if (copy_len > max_len) copy_len = max_len;
                if (copy_len > 0 && a_ptr != NULL) {
                    memcpy(row_dst, a_ptr, copy_len * sizeof(float));
                }
                if (copy_len < max_len) {
                    memset(row_dst + copy_len, 0, (max_len - copy_len) * sizeof(float));
                }
            } else {
                memset(row_dst, 0, max_len * sizeof(float));
            }
        }
    } else if (job->mode == 3) {
        const int16_t* src = job->pcm_src;
        float* dst = (float*)job->dst_ptr;
        const float scale = 1.0f / 32768.0f;
        for (size_t i = start_row; i < end_row; ++i) {
            dst[i] = (float)src[i] * scale;
        }
    }
}

static void run_batch_job_slice(const BatchJob* job, int tid) {
    size_t chunk = (job->total_rows + NUM_POOL_WORKERS - 1) / NUM_POOL_WORKERS;
    size_t start = (size_t)tid * chunk;
    size_t end = start + chunk;
    if (end > job->total_rows) end = job->total_rows;
    if (start < end) {
        run_batch_job_range(job, start, end);
    }
}

static void* pool_worker_main(void* arg) {
    int tid = (int)(intptr_t)arg;
    uint64_t my_gen = 0;
    for (;;) {
        pthread_mutex_lock(&g_pool_mu);
        while (g_pool_gen == my_gen) {
            pthread_cond_wait(&g_pool_work_cv, &g_pool_mu);
        }
        my_gen = g_pool_gen;
        BatchJob job = g_pool_job;
        pthread_mutex_unlock(&g_pool_mu);

        run_batch_job_slice(&job, tid);

        pthread_mutex_lock(&g_pool_mu);
        g_pool_pending--;
        if (g_pool_pending == 0) {
            pthread_cond_signal(&g_pool_done_cv);
        }
        pthread_mutex_unlock(&g_pool_mu);
    }
    return NULL;
}

static void init_pool_workers(void) {
    for (int t = 1; t < NUM_POOL_WORKERS; ++t) {
        pthread_t th;
        pthread_create(&th, NULL, pool_worker_main, (void*)(intptr_t)t);
        pthread_detach(th);
    }
}

static void dispatch_batch_job(const BatchJob* job) {
    if (job->total_rows == 0) return;
    int has_null_mel = 0;
    if (job->mode == 1) {
        for (size_t i = 0; i < job->actual_batch_size; ++i) {
            if (!job->mel_ptrs || job->mel_ptrs[i] == NULL) {
                has_null_mel = 1;
                break;
            }
        }
    }
    if ((!has_null_mel && job->total_rows < 16) || job->total_rows < 2) {
        run_batch_job_range(job, 0, job->total_rows);
        return;
    }
    pthread_once(&g_pool_once, init_pool_workers);
    pthread_mutex_lock(&g_dispatch_mu);

    pthread_mutex_lock(&g_pool_mu);
    g_pool_job = *job;
    g_pool_pending = NUM_POOL_WORKERS - 1;
    g_pool_gen++;
    pthread_cond_broadcast(&g_pool_work_cv);
    pthread_mutex_unlock(&g_pool_mu);

    run_batch_job_slice(job, 0);

    pthread_mutex_lock(&g_pool_mu);
    while (g_pool_pending > 0) {
        pthread_cond_wait(&g_pool_done_cv, &g_pool_mu);
    }
    pthread_mutex_unlock(&g_pool_mu);

    pthread_mutex_unlock(&g_dispatch_mu);
}

int av_fill_mel_batch_bf16(
    const uint16_t* const* mel_ptrs,
    const float* const* audio_ptrs,
    const size_t* audio_lens,
    size_t actual_batch_size,
    size_t padded_batch_size,
    size_t prev_dirty_rows,
    uint16_t* dst_ptr
) {
    if (!dst_ptr) return -1;
    size_t total_rows = actual_batch_size > prev_dirty_rows ? actual_batch_size : prev_dirty_rows;
    if (total_rows > padded_batch_size) total_rows = padded_batch_size;
    BatchJob job;
    job.mode = 1;
    job.mel_ptrs = mel_ptrs;
    job.audio_ptrs = audio_ptrs;
    job.audio_lens = audio_lens;
    job.actual_batch_size = actual_batch_size;
    job.total_rows = total_rows;
    job.max_audio_len = 0;
    job.dst_ptr = dst_ptr;
    dispatch_batch_job(&job);
    return 0;
}

int av_fill_padded_batch_f32(
    const float* const* audio_ptrs,
    const size_t* audio_lens,
    size_t actual_batch_size,
    size_t padded_batch_size,
    size_t prev_dirty_rows,
    size_t max_audio_len,
    float* dst_ptr
) {
    if (!audio_ptrs || !audio_lens || !dst_ptr) return -1;
    size_t total_rows = actual_batch_size > prev_dirty_rows ? actual_batch_size : prev_dirty_rows;
    if (total_rows > padded_batch_size) total_rows = padded_batch_size;
    BatchJob job;
    job.mode = 2;
    job.mel_ptrs = NULL;
    job.audio_ptrs = audio_ptrs;
    job.audio_lens = audio_lens;
    job.actual_batch_size = actual_batch_size;
    job.total_rows = total_rows;
    job.max_audio_len = max_audio_len;
    job.dst_ptr = dst_ptr;
    dispatch_batch_job(&job);
    return 0;
}
