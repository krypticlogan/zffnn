# Design constraints

ZGC specializes a model from compile-time graph metadata while keeping runtime
data ownership explicit. The following constraints define the current model
construction and execution flow.

## Compile-time model structure

- Architecture, tensor ranks, shapes, dtypes, layouts, and operation order are
  known at compile time.
- Definition bounds limit front-end construction. The counting pass derives the
  exact capacities used by graph lowering and memory planning.
- Shape and operation compatibility errors are reported during compilation when
  their inputs are statically known.
- The generated model type contains a fixed graph and memory plan; execution
  does not interpret or allocate graph nodes.

## Source ownership

Every graph source has one storage policy selected by `definition.modelWith`:

- Owned sources receive an aligned region in model memory and are populated
  through `copyInput` or `copySource`.
- Bound inputs borrow a caller-owned slice through `bindInput`. The slice must
  remain valid and unchanged for the duration of `run()`.
- Embedded parameters and constants use read-only program data supplied through
  `Source.embed`, commonly with `@embedFile`.

Unspecified policies are owned. Bound and embedded sources do not consume space
in the model's mutable memory plan.

## Views and execution

- Operations receive read-only input views and write only to their designated
  output storage.
- Layout-changing graph operations such as transpose create aliases rather than
  copying tensor data.
- Views preserve shape, offset, and stride metadata and continue to reference
  the root tensor's storage.
- `run()` executes the fixed operation list sequentially. Runtime input values
  may change between runs without rebuilding the model type.

## Memory

- A model instance owns one inline, aligned mutable byte array.
- The compile-time memory plan assigns regions to owned source and result
  tensors; aliases do not receive separate regions.
- The planner assigns permanent regions and does not reuse storage based on
  tensor lifetimes.
- Heap allocation is not required for model initialization or execution.

See [architecture](architecture.md) for the compilation pipeline and
[development state](development-state.md) for supported operations and current
limitations.
