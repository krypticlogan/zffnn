# Development state

ZGC is functional but pre-release. The compile-time definition-to-model path,
runtime execution, tests, sandbox, and operation benchmark suite are working on
Zig 0.16.0. The API should still be expected to change.

## Need to implement
1. reshape / permute / slice
2. mul / div
3. max / mean reductions
4. batched matmul
5. concatenate
6. convolution

## Implemented

### Definition and compilation

- A concrete, typed `DefinitionBackend` is the public model-builder.
- Models are defined once and compiled with `definition.model()`.
- Internal counting derives exact capacities within configurable definition
  bounds.
- Graph lowering preserves stable tensor IDs, source indices, operation order,
  multiple outputs, shapes, dtypes, and layouts.
- Invalid ranks, shapes, axes, dtypes, input counts, and broadcasting are
  rejected at compile time where the relevant metadata is static.

### Tensors and layouts

- Owned tensor metadata and mutable/read-only runtime views.
- Rank-zero through bounded-rank shapes.
- Contiguous layouts, offsets, arbitrary strides, negative strides, transpose
  aliases, axis slices, and broadcast views.
- `f32`, `f16`, and `i8` dtypes with scalar/vector mappings and accumulation
  helpers.

### Operations

| Operation | Current support |
| --- | --- |
| ReLU | Contiguous SIMD and generic strided traversal; float and signed integer |
| Exp | Floating-point tensors; contiguous SIMD and strided traversal |
| Add/sub | Matching dtypes, NumPy-style trailing-axis broadcasting, strided traversal |
| Matmul | Rank-2 tensors with contiguous and strided inputs/outputs |
| Sum | Single-axis reduction, including strided axes and rank-zero results |
| Softmax | Stable single-axis floating-point implementation, including strided axes |
| Transpose | Aliasing graph view; no runtime copy or kernel |

### Model and storage

- One inline, aligned memory allocation per model instance.
- Compile-time memory plan shared by runtime instances.
- Typed source/input copying, borrowed runtime inputs, embedded read-only
  parameters/constants, sequential graph execution, and typed output views.
- Views alias their root tensor's storage without adding another allocation.
- Writer-based capacity, graph, tree, memory-plan, and bounded model-memory
  inspection through `zgc.Inspect`.

### Tooling

- Unit and end-to-end tests through `zig build test`.
- Compile-only check through `zig build check`.
- Maintained root benchmark suite with shape and layout comparisons.
- Reusable model-specific inspection CLI and generated-model runner modules.
- Standalone sandbox consumer with an interactive model application and model
  artifact tooling.
- Embedding generator executable remains part of the root build.

## Important limitations

- Shapes and extents are currently compile-time fixed; bounded runtime extents
  are a design goal, not an implemented feature.
- Memory planning does not yet perform lifetime analysis or reuse regions.
- Runtime-bound inputs currently report a missing binding when their view is
  first resolved during execution rather than through a separate run preflight.
- There is no graph optimization, fusion, constant folding, or dead-node
  elimination pass yet.
- Matmul is a direct specialized kernel, not a tuned BLAS replacement.
- No training, automatic differentiation, dynamic control flow, or device/GPU
  backend exists.
- External parameter packs and memory-mapped parameter bindings are not yet
  implemented; parameters can currently be owned or compile-time embedded.
- `src/extensions/` provides standalone matrix and feed-forward network
  utilities through `zgc.Extensions`; model graphs use the separate
  `DefinitionBackend` API.
- Public naming and module boundaries remain subject to change before a stable
  release.

## Test coverage

The test suite currently exercises:

- typed definition construction and exact capacity counting;
- source indexing, graph materialization, and multiple ranks/outputs;
- shape, dtype, axis, rank, and broadcasting validation;
- aligned memory planning and source loading;
- model execution and typed output access;
- contiguous, offset, broadcast, transposed, and negative-stride views;
- ReLU, exp, add, sub, matmul, sum, softmax, and transpose behavior;
- SIMD tails and strided fallbacks;
- end-to-end execution across graph-produced aliasing views.
