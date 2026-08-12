# Zig Graph Compiler

ZGC is an early-stage, allocation-free inference graph compiler for Zig. A
model architecture is defined at compile time, lowered to a fixed execution
graph, assigned an inline memory plan, and emitted as a specialized Zig type.
The binary is the model: graph traversal, tensor ranks, shapes, dtypes, layouts,
and kernel selection are compile-time-known.

The project currently targets Zig 0.16.0 and is under active development.

## Current capabilities

- Typed, front-facing `DefinitionBackend` with enum-indexed sources.
- Automatic counting and graph-lowering passes through `definition.model()`.
- Exact graph capacities derived from user-configurable definition bounds.
- Inline aligned model memory with no heap allocation during execution.
- Contiguous, offset, broadcast, transposed, and generally strided tensor views.
- Multiple graph inputs, parameters, constants, and outputs.
- `f32`, `f16`, and `i8` tensor metadata and elementwise kernels where valid.
- ReLU, exp, add, sub, matmul, sum, softmax, and transpose operations.
- SIMD fast paths for contiguous kernels and generic strided traversal.
- Operation benchmarks and an external-consumer-style sandbox.

See [development state](docs/development-state.md) for precise limitations and
[architecture](docs/architecture.md) for the compilation pipeline.

## Defining a model

`DefinitionBackend` is the only public model-building backend. Source keys and
tensor values are concrete types; user code does not run separately against
counting and graph builders.

```zig
const std = @import("std");
const zgc = @import("zgc");

const Sources = enum(usize) { input, weights };
const Definition = zgc.DefinitionBackend(Sources, .{ .max_rank = 2 });

fn define(builder: *Definition) void {
    const input = builder.input(.input, .f32, &.{ 4, 8 });
    const weights = builder.parameter(.weights, .f32, &.{ 8, 16 });
    builder.output(builder.relu(builder.matmul(input, weights)));
}

const definition = blk: {
    var builder = Definition.init();
    define(&builder);
    break :blk builder.finish();
};

const MyModel = definition.model();

pub fn main() void {
    var model = MyModel.init();

    const input_values: [4 * 8]f32 = @splat(1);
    const weight_values: [8 * 16]f32 = @splat(0.25);
    const input_bytes: [@sizeOf(@TypeOf(input_values))]u8 = @bitCast(input_values);
    const weight_bytes: [@sizeOf(@TypeOf(weight_values))]u8 = @bitCast(weight_values);

    model.Source(Sources.input, &input_bytes);
    model.Source(Sources.weights, &weight_bytes);
    model.run();

    const output = model.outputView(0);
    std.debug.print("shape={any} data={any}\n", .{ output.shape, output.storage });
}
```

Definition limits have defaults and may be overridden at compile time. These
are front-end bounds, not final allocation sizes. The counting pass derives the
exact node, tensor, reference, output, source, and rank capacities before graph
construction and memory planning.

## Build and test

```sh
zig build test
zig build check
```

The package exports a module named `zgc`. The [sandbox](sandbox/README.md) shows
how a separate Zig package consumes it through `b.dependency("zgc", ...)`.

## Benchmarks

Run the complete maintained suite in `ReleaseFast`:

```sh
zig build benchmark -Doptimize=ReleaseFast
```

Select an individual case with `-Dop`, for example:

```sh
zig build benchmark -Dop=matmul-rhs-strided -Doptimize=ReleaseFast
```

See the [benchmark dashboard](benchmarks/README.md) for selectors, methodology,
and the latest local snapshot.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/zgc/backends/` | Definition, exact counting, graph lowering, and pipeline orchestration |
| `src/zgc/kernels/` | Elementwise, reduction, contraction, layout, and special kernels |
| `src/zgc/` | Graph, tensor/view, operation, storage, and executable-model machinery |
| `src/extensions/` | Older higher-level matrix/network extensions; not the primary graph API |
| `tests/` | Compile-time graph, runtime model, validation, view, and kernel coverage |
| `benchmarks/` | Maintained operation benchmark harness and results |
| `sandbox/` | Standalone consumer, diagnostics, and generated-code inspection |
| `docs/` | Architecture and current development status |

## Documentation

- [Documentation index](docs/README.md)
- [Architecture and compilation pipeline](docs/architecture.md)
- [Current development state](docs/development-state.md)
- [Sandbox and binary inspection](sandbox/README.md)
- [Benchmarks](benchmarks/README.md)
- [Upgrade notes](upgrades_notes.md)
