/*
 * benchmark.c - the single runner for all kernels.
 *
 * For every kernel it: allocates aligned buffers, fills them with data, runs
 * the scalar and NEON versions many times, checks that the two agree, and
 * records the timings. At the end it prints a Markdown table and writes
 * results/results.csv.
 *
 * Memory note (Raspberry Pi 4B has only 2 GB RAM): each kernel allocates its
 * buffers, uses them, and frees them BEFORE the next kernel runs, so the peak
 * footprint is just one kernel's working set (~16 MB here). The sizes below are
 * #defined so you can shrink them if you want more headroom.
 *
 * Build:  see the top-level Makefile  ->  make run
 */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#include "../examples/kernels.h"

/* ---- problem sizes (tune down for less memory / faster runs) ------------ */
#define VADD_N      (1u << 20)   /* vector_add elements        (~16 MB total) */
#define VADD_REPS   200
#define COLL_N      (1u << 20)   /* collision circles          (~14 MB total) */
#define COLL_REPS   200
#define MM_DIM      256          /* matrix dimension (mult of 4) (~1 MB total) */
#define MM_REPS     20
#define FIR_N       (1u << 20)   /* FIR input samples           (~12 MB total) */
#define FIR_TAPS    16
#define FIR_REPS    50

/* ---- result bookkeeping ------------------------------------------------- */
typedef struct {
    char   name[24];
    char   size[28];
    double scalar_ms;
    double neon_ms;
    double speedup;
    int    correct;
} Result;

static Result results[8];
static int    n_results = 0;

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

static void *xalloc(size_t bytes)
{
    void *p = NULL;
    if (posix_memalign(&p, 16, bytes) != 0) {
        fprintf(stderr, "allocation of %zu bytes failed\n", bytes);
        exit(1);
    }
    return p;
}

static float frand(void) { return (float)rand() / (float)RAND_MAX; }

static void record(const char *name, const char *size,
                   double scalar_ms, double neon_ms, int correct)
{
    Result *r = &results[n_results++];
    snprintf(r->name, sizeof r->name, "%s", name);
    snprintf(r->size, sizeof r->size, "%s", size);
    r->scalar_ms = scalar_ms;
    r->neon_ms   = neon_ms;
    r->speedup   = (neon_ms > 0.0) ? scalar_ms / neon_ms : 0.0;
    r->correct   = correct;
}

/* ---- individual benchmarks --------------------------------------------- */

static void bench_vector_add(void)
{
    const size_t N = VADD_N;
    float *a = xalloc(N * sizeof *a);
    float *b = xalloc(N * sizeof *b);
    float *c = xalloc(N * sizeof *c);
    float *ref = xalloc(N * sizeof *ref);

    for (size_t i = 0; i < N; i++) { a[i] = frand(); b[i] = frand(); }

    vector_add_scalar(a, b, ref, N);

    double t = now_ms();
    for (int r = 0; r < VADD_REPS; r++) vector_add_scalar(a, b, c, N);
    double scalar_ms = (now_ms() - t) / VADD_REPS;

    t = now_ms();
    for (int r = 0; r < VADD_REPS; r++) vector_add_neon(a, b, c, N);
    double neon_ms = (now_ms() - t) / VADD_REPS;

    int ok = 1;
    for (size_t i = 0; i < N; i++)
        if (fabsf(c[i] - ref[i]) > 1e-4f) { ok = 0; break; }

    char sz[28];
    snprintf(sz, sizeof sz, "%zu elems", N);
    record("vector_add", sz, scalar_ms, neon_ms, ok);

    free(a); free(b); free(c); free(ref);
}

static void bench_collision(void)
{
    const size_t N = COLL_N;
    float *xs = xalloc(N * sizeof *xs);
    float *ys = xalloc(N * sizeof *ys);
    float *rr = xalloc(N * sizeof *rr);
    unsigned char *out = xalloc(N);
    unsigned char *ref = xalloc(N);

    for (size_t i = 0; i < N; i++) {
        xs[i] = frand() * 100.0f;
        ys[i] = frand() * 100.0f;
        rr[i] = frand() * 5.0f;
    }
    const float cx = 50.0f, cy = 50.0f, cr = 5.0f;

    collision_scalar(xs, ys, rr, N, cx, cy, cr, ref);

    double t = now_ms();
    for (int r = 0; r < COLL_REPS; r++) collision_scalar(xs, ys, rr, N, cx, cy, cr, out);
    double scalar_ms = (now_ms() - t) / COLL_REPS;

    t = now_ms();
    for (int r = 0; r < COLL_REPS; r++) collision_neon(xs, ys, rr, N, cx, cy, cr, out);
    double neon_ms = (now_ms() - t) / COLL_REPS;

    int ok = 1;
    for (size_t i = 0; i < N; i++)
        if (out[i] != ref[i]) { ok = 0; break; }

    char sz[28];
    snprintf(sz, sizeof sz, "%zu circles", N);
    record("collision", sz, scalar_ms, neon_ms, ok);

    free(xs); free(ys); free(rr); free(out); free(ref);
}

static void bench_matmul(void)
{
    const unsigned DIM = MM_DIM;
    const size_t cnt = (size_t)DIM * DIM;
    float *A = xalloc(cnt * sizeof *A);
    float *B = xalloc(cnt * sizeof *B);
    float *C = xalloc(cnt * sizeof *C);
    float *ref = xalloc(cnt * sizeof *ref);

    for (size_t i = 0; i < cnt; i++) { A[i] = frand(); B[i] = frand(); }

    matmul_scalar(A, B, ref, DIM);

    double t = now_ms();
    for (int r = 0; r < MM_REPS; r++) matmul_scalar(A, B, C, DIM);
    double scalar_ms = (now_ms() - t) / MM_REPS;

    t = now_ms();
    for (int r = 0; r < MM_REPS; r++) matmul_neon(A, B, C, DIM);
    double neon_ms = (now_ms() - t) / MM_REPS;

    /* Scalar and NEON accumulate in a different order, so allow a small
     * relative tolerance rather than an exact match. */
    int ok = 1;
    for (size_t i = 0; i < cnt; i++)
        if (fabsf(C[i] - ref[i]) > 1e-3f * fabsf(ref[i]) + 1e-3f) { ok = 0; break; }

    char sz[28];
    snprintf(sz, sizeof sz, "%ux%u", DIM, DIM);
    record("matrix_multiply", sz, scalar_ms, neon_ms, ok);

    free(A); free(B); free(C); free(ref);
}

static void bench_fir(void)
{
    const size_t TAPS = FIR_TAPS;
    const size_t N = FIR_N;
    const size_t NOUT = N - TAPS + 1;
    float *x = xalloc(N * sizeof *x);
    float *h = xalloc(TAPS * sizeof *h);
    float *y = xalloc(NOUT * sizeof *y);
    float *ref = xalloc(NOUT * sizeof *ref);

    for (size_t i = 0; i < N; i++) x[i] = frand();
    for (size_t k = 0; k < TAPS; k++) h[k] = frand();

    fir_scalar(x, h, ref, NOUT, TAPS);

    double t = now_ms();
    for (int r = 0; r < FIR_REPS; r++) fir_scalar(x, h, y, NOUT, TAPS);
    double scalar_ms = (now_ms() - t) / FIR_REPS;

    t = now_ms();
    for (int r = 0; r < FIR_REPS; r++) fir_neon(x, h, y, NOUT, TAPS);
    double neon_ms = (now_ms() - t) / FIR_REPS;

    int ok = 1;
    for (size_t i = 0; i < NOUT; i++)
        if (fabsf(y[i] - ref[i]) > 1e-3f * fabsf(ref[i]) + 1e-3f) { ok = 0; break; }

    char sz[28];
    snprintf(sz, sizeof sz, "%zu out / %zu taps", NOUT, TAPS);
    record("fir_filter", sz, scalar_ms, neon_ms, ok);

    free(x); free(h); free(y); free(ref);
}

/* ---- output ------------------------------------------------------------- */

static void print_table(void)
{
    printf("\n| Kernel | Size | Scalar (ms) | NEON (ms) | Speedup | Correct |\n");
    printf("|--------|------|-------------|-----------|---------|---------|\n");
    for (int i = 0; i < n_results; i++) {
        Result *r = &results[i];
        printf("| %-15s | %-18s | %11.3f | %9.3f | %6.2fx | %-3s |\n",
               r->name, r->size, r->scalar_ms, r->neon_ms, r->speedup,
               r->correct ? "yes" : "NO");
    }
}

static void write_csv(void)
{
    const char *path = "results/results.csv";
    FILE *f = fopen(path, "w");
    if (!f) {            /* results/ missing? fall back to current dir */
        path = "results.csv";
        f = fopen(path, "w");
    }
    if (!f) {
        fprintf(stderr, "could not open a CSV file for writing\n");
        return;
    }
    fprintf(f, "kernel,size,scalar_ms,neon_ms,speedup,correct\n");
    for (int i = 0; i < n_results; i++) {
        Result *r = &results[i];
        fprintf(f, "%s,%s,%.3f,%.3f,%.3f,%d\n",
                r->name, r->size, r->scalar_ms, r->neon_ms, r->speedup, r->correct);
    }
    fclose(f);
    printf("\nResults written to %s\n", path);
}

int main(void)
{
    srand(1234);   /* fixed seed -> reproducible inputs */

    printf("ARM NEON SIMD benchmark - scalar vs NEON\n");
    printf("Running all kernels...\n");

    bench_vector_add();
    bench_collision();
    bench_matmul();
    bench_fir();

    print_table();
    write_csv();
    return 0;
}
