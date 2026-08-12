# Benchmarks

This directory is the home for maintained ZGC performance measurements. The
runnable cases live in `ops/`.

## Current snapshot

Lower latency is better. Higher throughput is better.

### Matmul shape scaling

| Shape (M×K×N) | Layout | Average latency | Throughput |
| --- | --- | ---: | ---: |
| 32×32×32 | contiguous | 3.920 us | 16.72 GFLOP/s |
| 64×64×64 | contiguous | 31.117 us | 16.85 GFLOP/s |
| 128×128×128 | contiguous | 373.730 us | 11.22 GFLOP/s |
| 32×128×64 | contiguous | 37.142 us | 14.12 GFLOP/s |

### Matmul layout cost (64×64×64)

| Layout | Average latency | Throughput | Relative to contiguous |
| --- | ---: | ---: | ---: |
| contiguous | 31.117 us | 16.85 GFLOP/s | 1.00× |
| strided lhs | 32.838 us | 15.97 GFLOP/s | 0.95× |
| strided rhs | 56.663 us | 9.25 GFLOP/s | 0.55× |
| strided output | 238.867 us | 2.19 GFLOP/s | 0.13× |

### Tensor operations

| Case | Average latency | Element throughput | Contiguous-relative throughput |
| --- | ---: | ---: | ---: |
| ReLU, contiguous 16K | 1.763 us | 9.291 Gelem/s | — |
| add, contiguous 128×128 | 3.927 us | 4.172 Gelem/s | 1.00× |
| add, transposed views 128×128 | 80.387 us | 203.8 Melem/s | 0.05× |
| add, broadcast bias 128×128 | 13.285 us | 1.233 Gelem/s | 0.30× |
| exp, contiguous 16K | 113.446 us | 144.4 Melem/s | — |
| sum axis, contiguous 256×256 | 8.188 us | 8.004 Gelem/s | 1.00× |
| sum axis, strided 256×256 | 99.006 us | 661.9 Melem/s | 0.08× |
| softmax axis, contiguous 256×256 | 923.519 us | 70.96 Melem/s | 1.00× |
| softmax axis, strided 256×256 | 1.317 ms | 49.78 Melem/s | 0.70× |

Recorded 2026-08-10 with Zig 0.16.0, `ReleaseFast`, x86_64 Darwin 24.6. These
are local development snapshots, not cross-machine comparisons. Each value is
the average of 10 timed runs using the case-specific defaults. ReLU was rerun
in isolation because its first full-suite sample was disturbed by system noise.

## Run

From the repository root:

```sh
zig build benchmark -Dop=relu -Doptimize=ReleaseFast
```

Without `-Dop`, the build runs the complete suite (`all`).

Run the comparison suite with:

```sh
zig build benchmark -Dop=all -Doptimize=ReleaseFast
```

Available selectors are:

| Group | `-Dop` values |
| --- | --- |
| Elementwise | `relu`, `add`, `add-strided`, `add-broadcast`, `exp` |
| Reduction | `sum`, `sum-strided`, `softmax`, `softmax-strided` |
| Matmul shapes | `matmul-32`, `matmul-64`, `matmul-128`, `matmul-rect` |
| Matmul layouts | `matmul-lhs-strided`, `matmul-rhs-strided`, `matmul-output-strided` |

The case-specific defaults can be overridden with `-Diterations`, `-Druns`,
and `-Dwarmup_iterations`. A zero iteration or warmup value selects the case
default. Always report the Zig version, target, optimization mode, CPU/OS, case
shape, and benchmark configuration alongside results.

The timer surrounds a batch of invocations, so its two reads are amortized over
the batch. Each case owns its input and output buffers, creates tensor views
outside the timed loop, warms the kernel first, and prevents the result from
being optimized away.

## Coverage

| Layer | Cases | Status |
| --- | --- | --- |
| Primitive | ReLU, add, exp | Measured, including strided and broadcast layouts |
| Primitive | matmul | Measured across four shapes and four 64³ layouts |
| Primitive | sum and softmax | Measured across contiguous and strided axes |
| Primitive | sub and remaining layout/elementwise ops | Planned |
| Fused/operator | dense + activation, attention projections, feed-forward | Planned |
| End to end | small MLP, transformer block; batch and sequence sweeps | Planned |
| Resources | workspace, allocations, binary size | Planned |

New operation cases belong in `benchmarks/ops/`. Add the case to the selection
in `main.zig`; each case exposes its name, default iteration counts, work unit,
work and byte counts per invocation, `init`, and `run`.

Do not compare current results directly with the archived zffnn CSV data. The
implementation, APIs, harness, and measurement semantics all changed.
