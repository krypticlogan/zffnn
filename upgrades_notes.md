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

    - `Graph Builder` (generally for internal use, construction of tensor-based models)
    - `Operators` (compute operations, composition/control operations, view operations)
    - `Models` (predefined generators for particular types of models)

  Model Pipeline
  ---
    Builder /
       ↓
    Graph /
       ↓
    Validation / lowering _
       ↓
    Memory and parameter plan /
       ↓
    Executable Model type /
       ↓
    Model instance /
       ↓
    run() / forward() /
    

  ### Next steps: Memory and parameter plan
    1. Add SourceKind and source records. done
    2. Add dtype byte-size/alignment helpers. done
    3. Create a basic storage plan with one region per tensor. done
    4. Bind inputs, parameters, and constants to those regions. done
    5. Execute nodes in graph order. done
    6. Return views of every graph output. done
    7. Add validation and better errors.
    8. Introduce storage reuse, fusion, and specialized kernels.
    9. 
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
   