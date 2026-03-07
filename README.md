# Zig FFNN

## User-defined NN Example for pretrained weights
```
const definition: []const struct { usize, Activation} = &.{ 
    .{feature_ct, .none}, 
    .{10, .relu}, 
    .{15, .sigmoid}, 
    .{2, .softmax}
};
const Net = NN(definition, batch_size);
var nn = Net.load_from_bin(path_to_binaries);

const preds = nn.forward(input); // for inference
```
## Description
ZFFNN is a statically defined feedforward neural network engine.

Instead of creating networks dynamically at runtime like most ML libs, this project uses Zig comptime to create a user-defined network at compile time. This enables:
- Compile-time dimension validation

- Zero heap allocations

- Fully static memory layout

- Deterministic binaries

Matrices and network internals are resolved to shape-encoded types and verified for dimensional correctness at compile time.

Matrices are stored row-major and use Zig's @Vector type internally to facilitate SIMD operations where applicable.

*Poorly defined models (shape mismatch, incorrect use of activations/ops, etc.) will **never** compile.*

## Philosophy
As opposed to established frameworks (PyTorch, TensorFlow), which prioritize being dynamic and flexibile, sometimes a static and predictable system is preferred. 
zffn embraces the static and predictable architecture, where the model is entirely static at runtime. Changing the network's definitio requires recompilation, allowing for the compiler to verify correctness before execution.

This lib is *not* a PyTorch replacement, by omitting dynamic control and runtime allocation we gain determinism and simplicity.
With this, it excels at creating tiny binaries for pretrained, deterministic neural networks.

While training will be supported, the more natural use case is for inference on pretrained models in a constrained environment.

ZFFNN compiles to a static binary for any user-defined network, so it is well suited for microcontrollers, single-board computers (Raspberry Pi), and other environments where space and predictability are imperative.

It is also able to load arbitrary network definitions from pretrained models and create itself at compile time, given a binary that defines weights and bias (typically from PyTorch).

This provides the ability to do inference on a system where speed, space, and correctness are vital.

## Features:
- Dense NN of arbitrary size built at compile time
- Static memory (zero heap allocations)
- Inference through forward pass
- SIMD via Zig @Vector 
- All operations on data (matmul, add, sub, etc.) are checked at compile to ensure they are well defined.
- Training via Stochastic Gradient Descent (coming soon)

- Currently supported activations are:
    - ReLU (Leaky ReLU to be added)
    - Sigmoid
    - Softmax
- Loss functions may be extended, but must be well defined for the shape, and will be checked at compile time.

## Future work:
- Backprop and training support
- Builtin loss functions
- Sparse layer support
- More activation functions
- Greater SIMD optimization
- Stripped inference-only build (smaller binaries)

