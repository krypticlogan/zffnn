# Architecture

ZGC specializes a statically defined tensor program into an executable Zig
model type. Definition bounds make compile-time storage possible; a counting
pass then removes unused capacity before the graph and model are materialized.

```text
typed model function
        │
        ▼
DefinitionBackend ──► completed definition
                            │
                            ▼
                     CountingBackend
                     exact capacities
                            │
                            ▼
                       GraphBackend
                     concrete graph
                            │
                            ▼
                       MemoryPlan
                  aligned tensor regions
                            │
                            ▼
                     executable Model type
```

Calling `definition.model()` performs every stage after definition. Counting
and graph backends are internal implementation details and are not exported as
public model-building APIs.

## Definition

`DefinitionBackend(SourceKey, limits)` is the typed front end. `SourceKey` must
be an enum, giving every input, parameter, or constant a stable compile-time
index. Its operation methods consume and return one concrete tensor-value type
whose metadata contains an ID, dtype, and bounded shape.

The definition records:

- source tensors and their source kinds;
- compute and view operations;
- flattened input references;
- inferred dtype and shape metadata;
- graph outputs.

Shape and dtype validation occurs while operations are added. `finish()` returns
the completed immutable definition value.

Definition limits currently cover maximum rank, nodes, tensors, input
references, and outputs. Defaults support small models; larger definitions can
override individual fields. Exceeding a bound is a compile error.

## Counting and graph lowering

The counting backend reads the completed definition and derives exact graph
capacities. In particular, the graph's rank capacity is the largest rank
actually used, rather than the definition's rank bound. Source storage uses
direct enum indexing, so its capacity is the highest referenced source index
plus one.

The graph backend then lowers tensors in definition order. Compute results own
contiguous storage. View operations, currently transpose, preserve their source
storage tensor and produce an aliasing layout with adjusted shape and strides.

The concrete graph stores fixed arrays of nodes, tensor metadata, flattened
input references, outputs, and sources. Node order is execution order.

## Memory planning and model generation

`MemoryPlan` assigns one aligned byte region to each storage-owning tensor.
Aliasing views point at their root storage tensor's region. The generated model
contains one inline byte array sized and aligned by that plan. Model-owned
sources and compute results receive regions in this array; embedded parameters,
embedded constants, and runtime-bound inputs remain external to it.

The current planner does not reuse regions when tensor lifetimes do not overlap,
so independent compute results receive distinct storage. Execution itself does
not allocate.

The model API currently provides:

- `init()` to zero-initialize model memory;
- `copyInput(key, values)` to copy a typed runtime input into owned storage;
- `copySource(key, values)` to initialize any model-owned source;
- `bindInput(key, values)` to borrow a typed runtime input without copying;
- `run()` to execute compute nodes in graph order;
- `outputView(index)` to retrieve a typed read-only view;
- `debugPrintMemory(limit)` plus compile-time graph and memory-plan metadata.

View nodes do not execute kernels. Their result layouts are resolved during
graph construction, and downstream compute kernels receive views into the
aliased storage.

`definition.modelWith(...)` selects non-default storage by source-enum tag.
`zgc.Source.embed(bytes)` places a parameter or constant in read-only program
data, while `zgc.Source.bound` makes an input borrow storage supplied to each
model instance. Dtype is enforced by the typed copy/bind APIs, and element or
byte counts are checked before a source is accepted.

## Kernel dispatch

Each compute node resolves typed input and output views and dispatches through
`Op.Compute.execute`. Kernels are grouped by family:

| Family | Implemented operations |
| --- | --- |
| Elementwise | ReLU, exp, add, sub; broadcasting for binary operations |
| Contraction | Rank-2 matmul |
| Reduction | Sum over one axis |
| Special | Softmax over one axis |
| Layout | Compile-time transpose inference |

Contiguous elementwise and selected reduction/contraction paths use SIMD.
Generic view traversal handles offsets and positive or negative strides where
the relevant kernel supports them.
