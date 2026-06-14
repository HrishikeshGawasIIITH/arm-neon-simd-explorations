# ARM NEON SIMD Explorations

> Harvesting the existing compute power in Arm processors with the NEON coprocessor.

**NEON** is Arm's Advanced SIMD (Single Instruction, Multiple Data) extension. Where a
normal instruction operates on one value at a time, a NEON instruction operates on a whole
vector of values packed into a 128-bit register at once — for example four 32-bit floats, or
sixteen 8-bit bytes. The Cortex-A72 in a Raspberry Pi 4B has thirty-two of these registers,
so loops that do the same arithmetic over large arrays (image processing, DSP, physics,
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
├── docs/                     # the original project decks (background reading)
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

This is meant to run **natively on the Raspberry Pi** (or any AArch64 Linux machine), where
`arm_neon.h` and the NEON instructions are available out of the box.

```sh
# on the Pi
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

### Example results

Run `make run` on your own Pi to populate this — actual numbers depend on your hardware,
kernel sizes, and compiler. The table below is **illustrative** of the shape of the results
(scalar vs NEON, with the speedup varying by how compute- vs memory-bound each kernel is):

| Kernel          | Size              | Scalar (ms) | NEON (ms) | Speedup | Correct |
|-----------------|-------------------|-------------|-----------|---------|---------|
| vector_add      | 1048576 elems     |        _…_  |    _…_    |  _~1–2x_ | yes     |
| collision       | 1048576 circles   |        _…_  |    _…_    |  _~3–4x_ | yes     |
| matrix_multiply | 256x256           |        _…_  |    _…_    |  _~4–6x_ | yes     |
| fir_filter      | 1048561 out / 16 taps |    _…_  |    _…_    |  _~3–4x_ | yes     |

`vector_add` gains the least because it is limited by memory bandwidth (one add per element);
`matrix_multiply` gains the most because it is compute-bound and NEON's fused
multiply-accumulate does four lanes at once.

## Platform

- **Target board:** Raspberry Pi 4B — Broadcom BCM2711, quad-core Cortex-A72 (ARMv8-A),
  64-bit, **2 GB RAM**, running 64-bit Linux (e.g. Ubuntu Server 22.04 / Raspberry Pi OS).
- **NEON on AArch64:** Advanced SIMD is part of the baseline 64-bit Arm ISA, so the intrinsics
  compile with a plain `gcc` — no `-mfpu` flag needed. (On a 32-bit Arm userland you would add
  `-mfpu=neon -mfloat-abi=hard`; see the commented line in the `Makefile`.)
- **Memory:** the Pi only has 2 GB, so the benchmark allocates each kernel's buffers, uses
  them, and frees them before moving to the next kernel — peak footprint is about one kernel's
  working set (~16 MB). All sizes are `#define`d at the top of `bench/benchmark.c` if you want
  to scale them down further.

## Background

For the full story — what SIMD is, the different ways to use NEON (auto-vectorisation,
intrinsics, libraries, assembly), the intrinsic naming convention and data types, and a
step-by-step walk-through of the collision-detection example — see the decks in [`docs/`](docs/).
