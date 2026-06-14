# ARM NEON SIMD Explorations

> Squeezing extra speed out of Arm CPUs using the NEON coprocessor.

This started as a college project — *"Harvesting Existing Compute Power in ARM using the NEON
Coprocessor"* (with Eshaan and Srikar, under Dr. Sudheendra Kumar). I've since cleaned it up
into something you can actually clone and run.

The idea is simple. Normally a CPU instruction works on one number at a time. **NEON** is Arm's
SIMD unit — one instruction works on a whole vector at once (say four floats), using compute
that's already sitting in the chip but that plain C usually leaves idle. So I took four common
kernels, wrote each one twice — once in plain C, once in hand-written NEON intrinsics — and
benchmarked them side by side to see how much faster NEON really is.

## What's inside

```
docs/        the project decks — the writeup of how and why NEON works
examples/    the four kernels, each as a scalar and a NEON version
  collision.c        circle vs circle collision detection
  vector_add.c       c = a + b
  matrix_multiply.c  C = A * B, in 4x4 blocks
  fir_filter.c       a FIR (DSP) filter
bench/       benchmark.c — runs every kernel, checks the two agree, times them
results/     the numbers I measured
```

## Running it

It runs on any Linux box with an Arm NEON CPU (any 64-bit Arm core) — I ran it on a Raspberry
Pi 4B. NEON is built into 64-bit Arm, so plain `gcc` is all you need:

```sh
make run
```

That builds the benchmark, runs all four kernels, and writes `results/results.csv`.

## Results

Measured on a Raspberry Pi 4B (Cortex-A72, 64-bit Linux):

| Kernel          | Size                  | Scalar (ms) | NEON (ms) | Speedup |
|-----------------|-----------------------|-------------|-----------|---------|
| vector_add      | 1048576 elems         |       2.979 |     2.756 |   1.08x |
| collision       | 1048576 circles       |       4.701 |     3.441 |   1.37x |
| matrix_multiply | 256x256               |     115.377 |    10.085 |  11.44x |
| fir_filter      | 1048561 out / 16 taps |      29.763 |     7.940 |   3.75x |

The pattern that stood out: NEON helps most when there's lots of math per byte of data.
`matrix_multiply` flies (**11.44x**) because it's all multiply-accumulate. `vector_add` barely
moves (**1.08x**) because it's just one add per element — the CPU is waiting on memory, not
maths. The other two land in between.

## Docs

The two decks in [`docs/`](docs/) are really the heart of the project — the part where we
explain NEON properly:

- **[`docs/PPT 1.pdf`](docs/PPT%201.pdf)** — the main presentation: what SIMD is, the ways to
  use NEON, a step-by-step collision-detection example, and how to set it up on a Pi.
- **[`docs/Arm_neon_intrinsic_library_functions.pdf`](docs/Arm_neon_intrinsic_library_functions.pdf)**
  — a reference of the NEON intrinsic families (arithmetic, compare, load/store, shift, logical,
  and so on).

The code in [`examples/`](examples/) is just those ideas made runnable.

## References

- [Arm NEON intrinsics search](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
- [Arm C Language Extensions (ACLE)](https://developer.arm.com/Architectures/Arm%20C%20Language%20Extensions)
- [Optimizing C code with NEON intrinsics (102467)](https://developer.arm.com/documentation/102467/latest/)
- [NEON Programmer's Guide (DEN0018)](https://developer.arm.com/documentation/den0018/a/Introduction)
