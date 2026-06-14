# ARM NEON SIMD Explorations

> Harvesting the existing compute power in Arm processors with the NEON coprocessor.

**NEON** is Arm's Advanced SIMD (Single Instruction, Multiple Data) extension. Where a
normal instruction operates on one value at a time, a NEON instruction operates on a whole
vector of values packed into a 128-bit register at once — for example four 32-bit floats, or
sixteen 8-bit bytes. An ARMv8-A core (such as the Cortex-A72) has thirty-two of these
registers, so loops that do the same arithmetic over large arrays (image processing, DSP, physics,
linear algebra) can run several times faster for free — using compute the CPU already has but
that plain C leaves idle. This repo demonstrates that with four small kernels, each written
twice — once in plain C, once in hand-written NEON intrinsics — and a benchmark that times
the two side by side.

This started as a student project (*"Harvesting Existing Compute power in ARM using NEON
Coprocessor"*, mentor Dr. Sudheendra Kumar) and is now cleaned up into a small, runnable
showcase.

## What's in this repo

```
arm-neon-simd-explorations/
├── docs/                     # the project decks — a primary contribution (see Documentation)
│   ├── PPT 1.pdf             # NEON / SIMD overview + worked collision example
│   └── Arm_neon_intrinsic_library_functions.pdf   # catalogue of intrinsic categories
├── examples/                 # the kernels, each as scalar + NEON
│   ├── kernels.h
│   ├── collision.c           # circle/circle collision detection (compare + mask)
│   ├── vector_add.c          # c = a + b              (memory-bandwidth bound)
│   ├── matrix_multiply.c     # C = A * B, 4x4 blocked (compute bound, FMA)
│   └── fir_filter.c          # FIR convolution        (DSP multiply-accumulate)
├── bench/
│   └── benchmark.c           # runs every kernel, validates, times scalar vs NEON
├── results/                  # benchmark CSV lands here
├── Makefile
└── README.md
```

Each kernel comes in two forms with identical results: a `*_scalar` plain-C version and a
`*_neon` version using `arm_neon.h` intrinsics. The benchmark runs both, checks they agree,
and reports the speedup.

## Build & run

The benchmark runs on **any Linux machine with an Arm NEON-capable CPU** (any AArch64 / 64-bit
Arm core), where `arm_neon.h` and the NEON instructions are available out of the box. The
results below were measured on a Raspberry Pi 4B.

```sh
git clone <your-repo-url> arm-neon-simd-explorations
cd arm-neon-simd-explorations
make run
```

`make run` builds the benchmark and runs every kernel, printing a table and writing
`results/results.csv`. Under the hood it is just:

```sh
gcc -O2 -Wall -Wextra -std=c11 \
    bench/benchmark.c examples/*.c -o benchmark -lm
./benchmark
```

To build a single example on its own (the classic workflow from the project deck):

```sh
gcc -O2 yourfile.c -o yourfile -lm   # AArch64: NEON needs no extra flags
./yourfile
```

### Results

`make run` builds the benchmark, runs every kernel, and writes
[`results/results.csv`](results/results.csv). The numbers below were measured on a **Raspberry
Pi 4B (Cortex-A72, 64-bit Linux)**; yours will differ with hardware, kernel sizes, and compiler,
but the *shape* — NEON pulling furthest ahead on compute-bound work — holds.

| Kernel          | Size                  | Scalar (ms) | NEON (ms) | Speedup | Correct |
|-----------------|-----------------------|-------------|-----------|---------|---------|
| vector_add      | 1048576 elems         |       2.979 |     2.756 |   1.08x | yes     |
| collision       | 1048576 circles       |       4.701 |     3.441 |   1.37x | yes     |
| matrix_multiply | 256x256               |     115.377 |    10.085 |  11.44x | yes     |
| fir_filter      | 1048561 out / 16 taps |      29.763 |     7.940 |   3.75x | yes     |

`vector_add` gains the least (**1.08x**): it is limited by memory bandwidth — one add per
element, so the CPU spends its time moving data rather than computing. `matrix_multiply` gains
the most (**11.44x**): it is compute-bound and NEON's fused multiply-accumulate does four lanes
per instruction. `fir_filter` (**3.75x**) and `collision` (**1.37x**) fall in between, set by
how much arithmetic each does per byte of data it loads.

## Documentation

Alongside the benchmark, the two decks in [`docs/`](docs/) are a primary part of this project —
the written explanation of *how* and *why* NEON works that the code then puts into practice.
They stand on their own as a compact introduction to programming the NEON coprocessor:

- **[NEON / SIMD overview — `docs/PPT 1.pdf`](docs/PPT%201.pdf)** — *Harvesting Existing
  Compute Power in ARM using the NEON Coprocessor.* The main presentation: what SIMD is and the
  three execution models (SISD → vector-mode VFP → packed SIMD), vectorization, NEON compared
  with x86 MMX/SSE and Altivec, the four ways to use NEON (auto-vectorization, intrinsics,
  libraries, hand-written assembly), a step-by-step walk-through of the collision-detection
  example (scalar → basic → advanced de-interleaved intrinsics), the intrinsic naming
  convention and data types, feature macros, and tips for helping the compiler auto-vectorize.

- **[Intrinsic function reference — `docs/Arm_neon_intrinsic_library_functions.pdf`](docs/Arm_neon_intrinsic_library_functions.pdf)**
  — a categorized catalogue of the NEON intrinsic families: bit manipulation, compare, complex
  arithmetic, load/store, shift, logical, data-type conversions, move, vector and scalar
  arithmetic, vector manipulation, and transpose operations.

The kernels in [`examples/`](examples/) are the runnable counterpart to this material.

## Platform

- **Where it runs:** any Linux machine with an Arm NEON-capable CPU. On AArch64 (64-bit Arm)
  Advanced SIMD is part of the baseline ISA, so the intrinsics compile with a plain `gcc` — no
  `-mfpu` flag needed. (On a 32-bit Arm userland you would add `-mfpu=neon -mfloat-abi=hard`;
  see the commented line in the `Makefile`.)
- **Where we ran it:** Raspberry Pi 4B — Broadcom BCM2711, quad-core Cortex-A72 (ARMv8-A),
  64-bit, 2 GB RAM, running 64-bit Linux (Ubuntu Server 22.04).
- **Memory-conscious by design:** the benchmark allocates each kernel's buffers, uses them, and
  frees them before moving to the next kernel — peak footprint is about one kernel's working set
  (~16 MB), which kept it comfortable even on the Pi's 2 GB. All sizes are `#define`d at the top
  of `bench/benchmark.c` if you want to scale them.

## References

- [Arm NEON intrinsics search](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
- [Arm C Language Extensions (ACLE)](https://developer.arm.com/Architectures/Arm%20C%20Language%20Extensions)
- [Optimizing C code with NEON intrinsics (102467)](https://developer.arm.com/documentation/102467/latest/)
- [NEON Programmer's Guide (DEN0018)](https://developer.arm.com/documentation/den0018/a/Introduction)
