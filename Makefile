# Build the NEON benchmark on the Raspberry Pi (AArch64 / Cortex-A72).
#
#   make        - build the benchmark binary
#   make run    - build, then run all kernels and write results/results.csv
#   make clean  - remove build artifacts
#
# On 64-bit Arm (Raspberry Pi OS 64-bit / Ubuntu Server arm64) NEON is part of
# the baseline ISA, so no special flags are needed. If you build on a 32-bit
# Arm userland, uncomment the CFLAGS line below to enable NEON.

CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra -std=c11
# CFLAGS += -mfpu=neon -mfloat-abi=hard   # <- only needed on 32-bit Arm

SRC = bench/benchmark.c \
      examples/collision.c \
      examples/vector_add.c \
      examples/matrix_multiply.c \
      examples/fir_filter.c

BIN = benchmark

all: $(BIN)

$(BIN): $(SRC) examples/kernels.h
	$(CC) $(CFLAGS) $(SRC) -o $(BIN) -lm

run: $(BIN)
	mkdir -p results
	./$(BIN)

clean:
	rm -f $(BIN)

.PHONY: all run clean
