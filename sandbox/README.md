# Model sandbox

The sandbox is a standalone Zig package that consumes ZGC through its
public package interface. 

It contains a 784→128→64→10 digit-classification network and
demonstrates a complete typed definition, definition-driven model generation,
external source storage, execution, diagnostics, and generated-code inspection
without relying on internal backend APIs.

Run commands from this directory unless noted otherwise.

## Interactive demo

```sh
zig build demo -Doptimize=ReleaseFast
```

The `demo` executable opens a 28×28 drawing canvas backed directly by the
model's runtime-bound input. Hold the left mouse button to draw a digit; the
dashboard updates all ten class probabilities as the canvas changes. Press
Enter to clear the canvas. The model parameters remain embedded read-only data.

## Diagnostic model run

```sh
zig build run
```

This prints the exact capacities discovered by the counting pass, the concrete
graph and output tree, the memory plan, a bounded memory preview before and
after execution, and the final output view.

The six parameter binaries live in `model_params/`. The
`model_params` module exposes them through `@embedFile`, and `src/model.zig`
assigns them with `zgc.Source.embed`. They are read-only data in the executable,
not members of the model's inline mutable memory. The 784-element input uses
`zgc.Source.bound` and borrows caller storage at runtime.

The weight files are serialized as `[output, input]`. Three zero-copy
transpose views adapt them to the graph matmul convention of `[input, output]`;
the views continue to alias the embedded bytes.

Consequently, the six parameters (437,544 bytes total) and the input receive no
memory-plan regions. The current unoptimized planner reserves 2,424 bytes only
for the network's nine intermediate/output tensors.

## Lean model artifact

Build the binary used for generated-code inspection:

```sh
zig build build-model -Doptimize=ReleaseFast
```

The resulting artifact is `zig-out/bin/zgc-model`. Its exported
`zgc_run_model` symbol contains model execution without logging or timing
instrumentation.

Run it directly:

```sh
zig build run-model -Doptimize=ReleaseFast
```

Disassemble only the stable execution symbol:

```sh
zig build inspect-model -Doptimize=ReleaseFast
```

On macOS, inspect it interactively with LLDB:

```text
lldb zig-out/bin/zgc-model
(lldb) breakpoint set --name zgc_run_model
(lldb) run
(lldb) disassemble --name zgc_run_model
```

## Model definition

The example in `src/model.zig` uses the same public workflow expected of a
consumer:

1. Instantiate `DefinitionBackend` with a source enum and front-end bounds.
2. Run the typed definition function once.
3. Finish the definition and call `definition.modelWith(...)` to select bound
   and embedded source storage.
4. Initialize the generated model, bind a runtime input, run, and retrieve an
   output view.

## Benchmarks

Benchmarks are maintained at the repository root rather than duplicated in the
sandbox. From the repository root, run:

```sh
zig build benchmark -Doptimize=ReleaseFast
```

See [benchmarks/README.md](../benchmarks/README.md) for individual selectors,
methodology, and current results.
