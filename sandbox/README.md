# Model sandbox

The sandbox is a standalone Zig package that consumes ZGC through its
public package interface. 

It acts as an example, demonstrating: 
a complete typed definition,
definition-driven model generation, 
source loading, 
execution, 
diagnostics, 
and generated-code inspection without relying on internal backend APIs.

Run commands from this directory unless noted otherwise.

## Diagnostic model run

```sh
zig build run
```

This prints the exact capacities discovered by the counting pass, the concrete
graph and output tree, the memory plan, memory before and after execution, and
the final output view.

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
3. Finish the definition and call `definition.model()`.
4. Initialize the generated model, load sources, run, and retrieve an output
   view.

## Benchmarks

Benchmarks are maintained at the repository root rather than duplicated in the
sandbox. From the repository root, run:

```sh
zig build benchmark -Doptimize=ReleaseFast
```

See [benchmarks/README.md](../benchmarks/README.md) for individual selectors,
methodology, and current results.
