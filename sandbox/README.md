# Model sandbox
This is a sandbox environment used in development.
It was chosen to be commited for users that want more insight or control to the library.

Run the complete graph-building, memory-planning, loading, execution, and
output flow with diagnostics:

```sh
cd sandbox
zig build run
```

Build the lean binary used for code inspection:

```sh
zig build build-model -Doptimize=ReleaseFast
```

The resulting binary is `zig-out/bin/zgc-model`. Its stable
`zgc_run_model` symbol contains only model execution—no logging, timers, or
instrumentation. Disassemble it directly with:

```sh
zig build inspect-model -Doptimize=ReleaseFast
```

On macOS, debugger inspection can begin with:

```sh
lldb zig-out/bin/zgc-model
(lldb) breakpoint set --name zgc_run_model
(lldb) run
(lldb) disassemble --name zgc_run_model
```

Run the lean artifact without adding a timing or tracing harness:

```sh
zig build run-model -Doptimize=ReleaseFast
```

Benchmark a tensor operation through the public library API:

```sh
zig build benchmark-op -Dop=relu -Doptimize=ReleaseFast
```

Each operation benchmark owns its input and output storage and constructs views
once per run. The direct hardware clock surrounds a batch of operation
invocations, amortizing its two timer reads across the entire batch. Configure
the measurement with `-Diterations`, `-Druns`, and `-Dwarmup_iterations`.

Add another case under `src/benchmarks/ops/`, then extend `SelectedBenchmark`
in `src/benchmarks/ops.zig` with its `-Dop` name.
