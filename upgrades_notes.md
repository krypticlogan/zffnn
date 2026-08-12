# Upgrade Notes
---
This document outlines the upgrade path for the library to support new architectures, broader applications and exist in a more complete state.

## Philosophy and intentions:
- **Architecture is known at compile time.**
- **Capacity is known at compile time.**
- **Runtime extents may vary within compile-time bounds.**
- **Memory layout is planned before execution.**
- **Shape errors should be compile-time errors whenever possible.**
- **Invalid and poorly defined processes should be compile-time errors whenever possible**
- **The execution graph is generated, not interpreted.**
- **Heap allocation is optional, not required.**
- **The binary *is* the model**

## Architectural changes:
  - Execution graph -- The execution graph will be defined by the programmer 
  and built by the compiler at compile-time.

  - Abstractions
    - `Tensor` (tensors are a high-level abstraction; they can represent a lot of different things)
      - `Storage` (for containing data -- owned memory)
      - `Dtype `(Moving away from f32 to support arbitrary data types)
      - `Layout` (Preferred layout/view -- `View` may be a factory internally for these layouts)

    - `View` (wraps storage, and provides and interface to 'see' the data, allowing for different layouts)
      - `Shape` (tuple type for defining dimensional shape)
      - `Stride` (tuple type for defining dimensional stride)

    - `Graph Builder` (generally for internal use, construction of tensor-based models) -- create a definition builder to consume
    - `Operators` (compute operations, composition/control operations, view operations)
    - `Models` (predefined generators for particular types of models)

  Model Pipeline
  ---
    Builder /
      ↓
    Validation / lowering _
      ↓
    Graph /
      ↓
    Memory and parameter plan /
      ↓
    Executable Model type /
      ↓
    Model instance /
      ↓
    run() / forward() /

 

  ### Next steps: 
    1. Add validation and better errors. in progress -- must do final pass (eliminate debug asserts, add better error messages, and add more validation)
    2. Implement kernels/operations -- in progress -- must do final pass
    3. Typed definition builder and definition-consuming counting/graph passes. complete
    5. Introduce storage reuse, fusion, and specialized kernels.
    6. Integrate parameter embeddings/linking into graph builder.
  
  ## Optimization goals: (3 levels)
  Though it's split into three distinct parts, the overall goal remains the same. Don't do extra work, and generate as effecient binaries as feasible.

  This is worth considering in design, but does not need to be fine-tuned until later on.
  
  ### 1. Kernel 
  - Cache sizes
  - Data layout
  - Quantization
  ### 2. Operator
  - Fused Ops
  ### 3. Graph 
  - Eliminate intermediate copies wherever possible
  - Specialize paths
  - Reuse memory
  - Separate data from views
  - Allow fine-tuning/customization for models that may sacrifice binary size for speed, and vice versa


  ## QOL (Visibility & Testing)
  Testing should be as simple and well covered as possible, and visibility into data structure and internals should be high.

  Matrices, graphs, and other internals should have some debug representation.

  ## Potential future goals:
    - Solid scientific computing
    - Backtrace the graph for optimization purposes (autodiff)

### Issues
  - Model.Source accepts any enum type, not
    specifically the model’s source enum. Another
    enum with the same numeric value can load a
    source accidentally. It also passes raw slices
    directly to @memcpy without a clear byte-length
    diagnostic. See src/zgc/model.zig:42.

  - Definition tensor values are publicly
    constructible records containing an ID, dtype,
    and shape. A fabricated value—or a value from
    another builder of the same type—can bypass
    graph provenance and metadata consistency. See
    src/zgc/backends/definition.zig:14.

  - Memory planning assigns permanent storage to
    every owning tensor. It is correct but does no
    lifetime reuse, so longer graphs will consume
    substantially more inline memory than
    necessary. See src/zgc/storage.zig:13.

  - Validation tests cover predicate behavior well,
    but there is little direct coverage of public-
    API misuse, wrong source sizes/types, capacity
    failures, or expected compile-error
    diagnostics.

  - The simultaneous export of the new graph API
    and legacy Extensions surface could make
    package positioning unclear until the older API
    is migrated or explicitly marked compatibility-
    only.