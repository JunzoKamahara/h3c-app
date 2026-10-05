/* End-to-end reproducibility check: runs the same request several times in
 * one process and fails unless every run hands the encoder bit-identical RGB
 * frames. Each delivered frame is hashed on its own, so the result does not
 * depend on whether the decoder ran streamed (chunked) or monolithic. A
 * failed generation, a missing frame or a frame-count shortfall also fails.
 * The decoded audio (PCM handed to the encoder) must match too; its hash
 * comes from H3_DEBUG_HASHES, which this tool turns on.
 *
 * The default request is a short Ref2VA clip with one reference image: its
 * long Qwen sequence is what exposed the GQA threadgroup race. The
 * generation cache (h3_cache_set_targets) is off unless a run asks for it,
 * so the text encoder runs every time; the int8 attention cache is a
 * separate setting and is used as the app uses it.
 *
 * Runs may override the base request (see usage()); runs with the same
 * request - ignoring `cache` - must match, which also covers cache on/off
 * and A -> B -> A sequences. Per run it prints the time, the time per phase
 * and the peak process footprint (sampled), for evaluating the cache.
 *
 * Build: make h3_repro_check. Not part of `make test` (needs the released
 * weights and takes minutes). The stage hashes H3_DEBUG_HASHES logs (also
 * on stderr) locate where two runs diverge. */
#include "h3.h"

#include <CoreFoundation/CoreFoundation.h>
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>
#include <mach/mach.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum { MAX_RUNS = 64, MAX_FRAMES = 1024, MAX_PHASES = 24 };

typedef struct {
    char prompt[512];
    char reference[1024]; /* empty = T2V */
    int width;
    int height;
    int frames;
    int steps;
    int reuse;
    int layers;
    int core_reuse;
    int fast;
    uint64_t seed;
    unsigned cache; /* H3_CACHE_* targets */
} request;

typedef struct {
    char name[48];
    double seconds;
} phase_time;

typedef struct {
    request req;
    int ok;
    int frames_expected;
    int frames_delivered;
    int width, height;
    uint64_t frame_hash[MAX_FRAMES];
    int frame_seen[MAX_FRAMES];
    uint64_t video_hash;
    uint64_t audio_hash; /* the decoded PCM, from H3_DEBUG_HASHES */
    int audio_seen;
    double seconds;
    double peak_gib;
    double start_gib;
    phase_time phases[MAX_PHASES];
    int phase_count;
    char error[512];
} run_result;

static run_result results[MAX_RUNS];
static int run_count;

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static double footprint_gib(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info,
                  &count) != KERN_SUCCESS) return 0.0;
    return (double)info.phys_footprint / (1024.0 * 1024.0 * 1024.0);
}

/* Footprint sampler: phys_footprint includes Metal buffers on Apple
 * silicon, so this is the process's whole CPU+GPU footprint. */
static _Atomic int sampler_stop;
static _Atomic double sampler_peak;

static void *sampler(void *unused) {
    (void)unused;
    while (!atomic_load(&sampler_stop)) {
        double value = footprint_gib();
        if (value > atomic_load(&sampler_peak)) atomic_store(&sampler_peak, value);
        usleep(50000);
    }
    return NULL;
}

static uint64_t fnv1a(uint64_t hash, const void *data, size_t bytes) {
    const unsigned char *p = data;
    for (size_t i = 0; i < bytes; i++) {
        hash ^= p[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static run_result *current;

/* stderr is read back through a pipe (and passed on unchanged) to pick up
 * "h3: hash audio waveform": the audio is compared as the PCM handed to the
 * encoder, since the AAC track itself is not bit-exact. */
static int stderr_original = -1;
static pthread_mutex_t stderr_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t stderr_cond = PTHREAD_COND_INITIALIZER;
static unsigned long stderr_marks_seen;

static void *stderr_reader(void *opaque) {
    FILE *in = fdopen((int)(intptr_t)opaque, "r");
    char line[8192];
    while (in && fgets(line, sizeof(line), in)) {
        unsigned long mark;
        if (sscanf(line, "h3_repro_mark %lu", &mark) == 1) {
            pthread_mutex_lock(&stderr_lock);
            stderr_marks_seen = mark;
            pthread_cond_broadcast(&stderr_cond);
            pthread_mutex_unlock(&stderr_lock);
            continue;
        }
        if (write(stderr_original, line, strlen(line)) < 0) {}
        const char *hash = strstr(line, "h3: hash audio waveform ");
        if (hash) {
            pthread_mutex_lock(&stderr_lock);
            if (current) {
                current->audio_hash = strtoull(hash + 24, NULL, 16);
                current->audio_seen = 1;
            }
            pthread_mutex_unlock(&stderr_lock);
        }
    }
    return NULL;
}

static void start_stderr_reader(void) {
    int fds[2];
    if (pipe(fds) != 0) return;
    stderr_original = dup(STDERR_FILENO);
    dup2(fds[1], STDERR_FILENO);
    close(fds[1]);
    setvbuf(stderr, NULL, _IOLBF, 0);
    pthread_t thread;
    pthread_create(&thread, NULL, stderr_reader, (void *)(intptr_t)fds[0]);
    pthread_detach(thread);
}

/* Waits until everything written to stderr so far has been read. */
static void sync_stderr(void) {
    static unsigned long mark;
    if (stderr_original < 0) return;
    fprintf(stderr, "h3_repro_mark %lu\n", ++mark);
    fflush(stderr);
    pthread_mutex_lock(&stderr_lock);
    while (stderr_marks_seen < mark) pthread_cond_wait(&stderr_cond, &stderr_lock);
    pthread_mutex_unlock(&stderr_lock);
}
static double run_start;
static double phase_start;

static void close_phase(double t) {
    if (current->phase_count)
        current->phases[current->phase_count - 1].seconds = t - phase_start;
}

static int on_progress(const char *phase, int completed, int total, void *opaque) {
    (void)completed; (void)total; (void)opaque;
    run_result *r = current;
    if (r->phase_count && !strcmp(r->phases[r->phase_count - 1].name, phase))
        return 0;
    double t = now();
    close_phase(t);
    if (r->phase_count < MAX_PHASES) {
        snprintf(r->phases[r->phase_count].name,
                 sizeof(r->phases[0].name), "%s", phase);
        r->phases[r->phase_count].seconds = 0.0;
        r->phase_count++;
        phase_start = t;
    }
    return 0;
}

static int on_frame(const h3_frame *frame, void *opaque) {
    (void)opaque;
    if (frame->denoise_step >= 0) return 0;
    run_result *r = current;
    int index = frame->frame_index;
    if (index < 0 || index >= MAX_FRAMES || !frame->rgb) {
        snprintf(r->error, sizeof(r->error), "frame %d out of range", index);
        return 0;
    }
    uint64_t hash = 1469598103934665603ULL;
    for (int row = 0; row < frame->height; row++)
        hash = fnv1a(hash, frame->rgb + (size_t)row * frame->stride,
                     (size_t)frame->width * 3);
    if (r->frame_seen[index] && r->frame_hash[index] != hash)
        snprintf(r->error, sizeof(r->error), "frame %d delivered twice", index);
    r->frame_hash[index] = hash;
    r->frame_seen[index] = 1;
    r->frames_delivered++;
    r->width = frame->width;
    r->height = frame->height;
    if (frame->frame_count > r->frames_expected)
        r->frames_expected = frame->frame_count;
    return 0;
}

/* A neutral, deterministic 512x512 reference image (smooth gradients and a
 * few shapes) so the check needs no external file. */
static int write_reference_png(const char *path, int size) {
    CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
    CGContextRef bitmap = CGBitmapContextCreate(
        NULL, (size_t)size, (size_t)size, 8, (size_t)size * 4, space,
        (CGBitmapInfo)kCGImageAlphaNoneSkipLast);
    if (!bitmap) { CGColorSpaceRelease(space); return 0; }
    uint8_t *pixels = CGBitmapContextGetData(bitmap);
    for (int y = 0; y < size; y++)
        for (int x = 0; x < size; x++) {
            uint8_t *p = pixels + ((size_t)y * size + x) * 4;
            int dx = x - size / 2, dy = y - size * 2 / 5;
            int disc = dx * dx + dy * dy < (size / 5) * (size / 5);
            p[0] = (uint8_t)(disc ? 230 : 60 + x * 120 / size);
            p[1] = (uint8_t)(disc ? 150 : 90 + y * 100 / size);
            p[2] = (uint8_t)(disc ? 60 : 160 - y * 80 / size);
            p[3] = 255;
        }
    CGImageRef image = CGBitmapContextCreateImage(bitmap);
    CFURLRef url = CFURLCreateFromFileSystemRepresentation(
        NULL, (const UInt8 *)path, (CFIndex)strlen(path), false);
    CGImageDestinationRef destination =
        url ? CGImageDestinationCreateWithURL(url, CFSTR("public.png"), 1, NULL)
            : NULL;
    int ok = 0;
    if (destination && image) {
        CGImageDestinationAddImage(destination, image, NULL);
        ok = CGImageDestinationFinalize(destination);
    }
    if (destination) CFRelease(destination);
    if (url) CFRelease(url);
    if (image) CGImageRelease(image);
    CGContextRelease(bitmap);
    CGColorSpaceRelease(space);
    return ok;
}

static void usage(void) {
    fprintf(stderr,
        "usage: h3_repro_check [--model DIR] [--ref PATH|none] [--runs N]\n"
        "         [--size N] [--frames N] [--steps N] [--reuse N] [--layers N]\n"
        "         [--core-reuse N] [--fast 0|1] [--seed N] [--prompt TEXT]\n"
        "         [--cache 0|1|LIST] [--run 'key=value;key=value' ...]\n"
        "  --run adds one run with the base request overridden by the given\n"
        "  keys (prompt, ref, size or width/height, frames, steps, reuse,\n"
        "  layers, core_reuse, fast, seed, cache); without --run the base\n"
        "  request runs N times.\n"
        "  cache: 0, 1 (all) or a comma list of conditioning, dit, decoder,\n"
        "  refined, adaln.\n"
        "  Defaults: Ref2VA with a generated 512x512 reference image, 512x512,\n"
        "  25 frames, 8 steps, reuse 2, seed 7, 3 runs, generation cache off.\n");
    exit(2);
}

/* "0", "1" (all) or a comma list of conditioning, dit, decoder. */
static int parse_cache(const char *value, unsigned *targets) {
    if (!strcmp(value, "0")) { *targets = 0; return 1; }
    if (!strcmp(value, "1")) { *targets = H3_CACHE_ALL; return 1; }
    unsigned result = 0;
    char *copy = strdup(value);
    char *save = NULL;
    for (char *item = strtok_r(copy, ",", &save); item;
         item = strtok_r(NULL, ",", &save)) {
        if (!strcmp(item, "conditioning")) result |= H3_CACHE_CONDITIONING;
        else if (!strcmp(item, "dit")) result |= H3_CACHE_DIT;
        else if (!strcmp(item, "decoder")) result |= H3_CACHE_DECODER;
        else if (!strcmp(item, "refined")) result |= H3_CACHE_REFINED_TEXT;
        else if (!strcmp(item, "adaln")) result |= H3_CACHE_ADALN;
        else { free(copy); return 0; }
    }
    free(copy);
    *targets = result;
    return 1;
}

static int set_key(request *req, const char *key, const char *value) {
    if (!strcmp(key, "prompt")) snprintf(req->prompt, sizeof(req->prompt), "%s", value);
    else if (!strcmp(key, "ref"))
        snprintf(req->reference, sizeof(req->reference), "%s",
                 strcmp(value, "none") ? value : "");
    else if (!strcmp(key, "size")) req->width = req->height = atoi(value);
    else if (!strcmp(key, "width")) req->width = atoi(value);
    else if (!strcmp(key, "height")) req->height = atoi(value);
    else if (!strcmp(key, "frames")) req->frames = atoi(value);
    else if (!strcmp(key, "steps")) req->steps = atoi(value);
    else if (!strcmp(key, "reuse")) req->reuse = atoi(value);
    else if (!strcmp(key, "layers")) req->layers = atoi(value);
    else if (!strcmp(key, "core_reuse") || !strcmp(key, "core-reuse"))
        req->core_reuse = atoi(value);
    else if (!strcmp(key, "fast")) req->fast = atoi(value);
    else if (!strcmp(key, "seed")) req->seed = strtoull(value, NULL, 10);
    else if (!strcmp(key, "cache")) return parse_cache(value, &req->cache);
    else return 0;
    return 1;
}

static void parse_run(request *req, const char *spec) {
    char *copy = strdup(spec);
    char *save = NULL;
    for (char *item = strtok_r(copy, ";", &save); item;
         item = strtok_r(NULL, ";", &save)) {
        char *equals = strchr(item, '=');
        if (!equals) usage();
        *equals = '\0';
        if (!set_key(req, item, equals + 1)) {
            fprintf(stderr, "unknown run key: %s\n", item);
            usage();
        }
    }
    free(copy);
}

/* Everything that determines the output; `cache` deliberately excluded. */
static int same_request(const request *a, const request *b) {
    return !strcmp(a->prompt, b->prompt) && !strcmp(a->reference, b->reference) &&
           a->width == b->width && a->height == b->height &&
           a->frames == b->frames && a->steps == b->steps &&
           a->reuse == b->reuse && a->layers == b->layers &&
           a->core_reuse == b->core_reuse && a->fast == b->fast &&
           a->seed == b->seed;
}

static void describe(const request *req, char *out, size_t size) {
    snprintf(out, size, "%s %dx%d %dfr %dst reuse%d seed%llu%s%s cache%s%s%s%s%s%s "
             "\"%.40s\"",
             req->reference[0] ? "Ref2VA" : "T2V", req->width, req->height,
             req->frames, req->steps, req->reuse,
             (unsigned long long)req->seed,
             req->fast ? " fast" : "",
             req->layers != 50 ? " layers<50" : "",
             req->cache ? "" : " off",
             req->cache & H3_CACHE_CONDITIONING ? " conditioning" : "",
             req->cache & H3_CACHE_DIT ? " dit" : "",
             req->cache & H3_CACHE_DECODER ? " decoder" : "",
             req->cache & H3_CACHE_REFINED_TEXT ? " refined" : "",
             req->cache & H3_CACHE_ADALN ? " adaln" : "", req->prompt);
}

static void run_one(h3_ctx *ctx, run_result *r, const char *home) {
    const request *req = &r->req;
    char cache_path[1024];
    snprintf(cache_path, sizeof(cache_path), "%s/models/cache/%s", home,
             req->reference[0] ? "dit_int8_v2_ref2va.cache" : "dit_int8_v2.cache");
    /* As H3Engine.generate sets them for the app's int8 cache. */
    setenv("H3_ATTENTION_CACHE", cache_path, 1);
    setenv("H3_INT8_STREAM_MLP", "1", 1);
    if (!getenv("H3_QWEN_PREFETCH_DEPTH")) setenv("H3_QWEN_PREFETCH_DEPTH", "1", 1);
    h3_cache_set_targets(ctx, req->cache);

    h3_reference reference = {H3_REFERENCE_IMAGE, req->reference, NULL, 0};
    h3_params params = H3_PARAMS_DEFAULT;
    params.width = req->width;
    params.height = req->height;
    params.frames = req->frames;
    params.steps = req->steps;
    params.seed = req->seed;
    params.dit_layers = req->layers;
    params.denoise_reuse = req->reuse;
    params.core_reuse = req->core_reuse;
    params.fast_attention = req->fast;
    params.reference_image_size = H3_REFERENCE_IMAGE_MATCH;
    if (req->reference[0]) {
        params.references = &reference;
        params.reference_count = 1;
    }
    char output[1024];
    /* H3_REPRO_KEEP_DIR keeps each run's video there (to look at it). */
    const char *keep = getenv("H3_REPRO_KEEP_DIR");
    snprintf(output, sizeof(output), "%s/h3_repro_%d.mp4",
             keep ? keep : getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp",
             (int)(r - results));
    params.output_path = output;
    params.on_progress = on_progress;
    params.on_frame = on_frame;

    pthread_mutex_lock(&stderr_lock); /* stderr_reader reads it */
    current = r;
    pthread_mutex_unlock(&stderr_lock);
    r->start_gib = footprint_gib();
    atomic_store(&sampler_peak, r->start_gib);
    run_start = phase_start = now();
    h3_result *result = h3_generate(ctx, req->prompt, &params);
    double end = now();
    sync_stderr();
    close_phase(end);
    r->seconds = end - run_start;
    r->peak_gib = atomic_load(&sampler_peak);
    if (!result) {
        snprintf(r->error, sizeof(r->error), "generation failed: %s",
                 h3_last_error(ctx) ? h3_last_error(ctx) : "unknown");
        return;
    }
    if (result->frames > r->frames_expected) r->frames_expected = result->frames;
    h3_result_free(result);
    if (!keep) unlink(output);
    if (r->error[0]) return;
    if (r->frames_expected <= 0 || r->frames_expected > MAX_FRAMES) {
        snprintf(r->error, sizeof(r->error), "bad frame count %d",
                 r->frames_expected);
        return;
    }
    uint64_t video = 1469598103934665603ULL;
    for (int i = 0; i < r->frames_expected; i++) {
        if (!r->frame_seen[i]) {
            snprintf(r->error, sizeof(r->error), "frame %d missing (%d of %d "
                     "delivered)", i, r->frames_delivered, r->frames_expected);
            return;
        }
        video = fnv1a(video, &r->frame_hash[i], sizeof(r->frame_hash[i]));
    }
    r->video_hash = video;
    if (!r->audio_seen) {
        snprintf(r->error, sizeof(r->error), "no audio waveform hash");
        return;
    }
    r->ok = 1;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    setenv("H3_DEBUG_HASHES", "1", 1);
    start_stderr_reader();
    const char *home = getenv("HOME") ? getenv("HOME") : "";
    char model[1024];
    snprintf(model, sizeof(model), "%s/models/MiniMax-H3", home);
    request base = {0};
    snprintf(base.prompt, sizeof(base.prompt), "%s",
             "A cat playing with a ball of yarn.");
    base.width = base.height = 512;
    base.frames = 25;
    base.steps = 8;
    base.reuse = 2;
    base.layers = 50;
    base.core_reuse = 1;
    base.seed = 7;
    int runs = 3;
    int ref_given = 0;
    const char *run_specs[MAX_RUNS];
    int spec_count = 0;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (i + 1 >= argc) usage();
        const char *value = argv[++i];
        if (!strcmp(arg, "--model")) snprintf(model, sizeof(model), "%s", value);
        else if (!strcmp(arg, "--runs")) runs = atoi(value);
        else if (!strcmp(arg, "--run")) {
            if (spec_count == MAX_RUNS) usage();
            run_specs[spec_count++] = value;
        } else if (!strncmp(arg, "--", 2) && set_key(&base, arg + 2, value)) {
            if (!strcmp(arg, "--ref")) ref_given = 1;
        } else usage();
    }
    if (!ref_given) {
        snprintf(base.reference, sizeof(base.reference), "%s/h3_repro_ref.png",
                 getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp");
        if (!write_reference_png(base.reference, 512)) {
            fprintf(stderr, "FAIL: cannot write the reference image\n");
            return 1;
        }
    }
    if (spec_count) {
        run_count = spec_count;
        for (int i = 0; i < spec_count; i++) {
            results[i].req = base;
            parse_run(&results[i].req, run_specs[i]);
        }
    } else {
        if (runs < 2 || runs > MAX_RUNS) usage();
        run_count = runs;
        for (int i = 0; i < runs; i++) results[i].req = base;
    }

    h3_ctx *ctx = h3_load_dir(model);
    if (!ctx) {
        fprintf(stderr, "FAIL: cannot load %s\n", model);
        return 1;
    }
    pthread_t thread;
    pthread_create(&thread, NULL, sampler, NULL);
    for (int i = 0; i < run_count; i++) {
        run_result *r = &results[i];
        char text[256];
        describe(&r->req, text, sizeof(text));
        printf("run %d: %s\n", i + 1, text);
        run_one(ctx, r, home);
        h3_cache_info info;
        h3_cache_get_info(ctx, &info);
        if (r->ok)
            printf("run %d: %.1fs, peak %.2f GiB (start %.2f), %d frames "
                   "%dx%d, video %016llx, audio %016llx; retained: "
                   "conditioning %zu B, refined %.1f MiB, AdaLN %.1f MiB, "
                   "DiT %d, decoder %d\n",
                   i + 1, r->seconds, r->peak_gib, r->start_gib,
                   r->frames_expected, r->width, r->height,
                   (unsigned long long)r->video_hash,
                   (unsigned long long)r->audio_hash, info.embedding_bytes,
                   (double)info.refined_text_bytes / (1024.0 * 1024.0),
                   (double)info.adaln_bytes / (1024.0 * 1024.0),
                   info.prepared_dit, info.video_decoder);
        else
            printf("run %d: FAILED after %.1fs: %s\n", i + 1, r->seconds,
                   r->error);
        printf("run %d phases:", i + 1);
        for (int p = 0; p < r->phase_count; p++)
            printf(" %s %.1fs%s", r->phases[p].name, r->phases[p].seconds,
                   p + 1 < r->phase_count ? "," : "");
        printf("\n");
    }
    atomic_store(&sampler_stop, 1);
    pthread_join(thread, NULL);
    h3_free(ctx);

    int failures = 0;
    for (int i = 0; i < run_count; i++) failures += !results[i].ok;
    for (int i = 0; i < run_count; i++) {
        if (!results[i].ok) continue;
        int first = i;
        for (int j = 0; j < i; j++)
            if (same_request(&results[j].req, &results[i].req)) { first = j; break; }
        if (first == i || !results[first].ok) continue;
        run_result *a = &results[first], *b = &results[i];
        if (a->audio_hash != b->audio_hash) {
            printf("MISMATCH: run %d audio differs from run %d\n", i + 1,
                   first + 1);
            failures++;
        }
        if (a->frames_expected != b->frames_expected || a->video_hash != b->video_hash) {
            int frame = -1;
            int frames = a->frames_expected < b->frames_expected
                             ? a->frames_expected : b->frames_expected;
            for (int f = 0; f < frames; f++)
                if (a->frame_hash[f] != b->frame_hash[f]) { frame = f; break; }
            printf("MISMATCH: run %d differs from run %d (%d vs %d frames, "
                   "first differing frame %d)\n", i + 1, first + 1,
                   b->frames_expected, a->frames_expected, frame);
            failures++;
        } else if (a->audio_hash == b->audio_hash) {
            printf("match: run %d == run %d (RGB and audio)\n", i + 1, first + 1);
        }
    }
    int groups = 0;
    for (int i = 0; i < run_count; i++) {
        int seen = 0;
        for (int j = 0; j < i; j++) seen |= same_request(&results[j].req, &results[i].req);
        groups += !seen;
    }
    if (groups == run_count) {
        printf("FAIL: no two runs share a request, nothing was compared\n");
        failures++;
    }
    printf("%s: %d runs, %d failures\n", failures ? "FAIL" : "PASS", run_count,
           failures);
    return failures ? 1 : 0;
}
