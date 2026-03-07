# Zig FFNN

## User-defined NN Example
```
const defintion: []const struct { usize, MatOp} = &.{ 
    .{feature_ct, .relu}, 
    .{10, .relu}, 
    .{15, .sigmoid}, 
    .{2, .softmax}
};
const Net = NN(definition, batch_size);
var nn = Net.new();

nn.train(training_data, 1000); // SGD training
nn.forward(input); // for inference
```
## Description
ZFFNN is a statically defined dense neural network engine.

Instead of creating networks dynamically at runtime like most NN libs, this project uses Zig comptime to create a user defined FFNN at compile time. This enables:
- Compile-time dimension validation

- Zero heap allocations

- Fully static memory layout

- Deterministic binary output

Matrices and network internals are resolved to shape-encoded types and verified at compile time.

Matrices are stored row-major and use Zig's @Vector type internally to facilitate SIMD operations where applicable.

*Poorly defined models (shape mismatch, incorrect use of activations/ops, etc.) will **never** compile.*

## Philosophy
As opposed to established frameworks (PyTorch, TensorFlow), which rely on being dynamic and flexibility, sometimes you need a static and predicticable load. zffnn explores the opposite architecture, where the model is entirely static at runtime, meaning that a new definition of a network will need to be compiled and verified again from scratch.

This lib is strictly not a PyTorch replacement, as it greatly restricts the flexibility in training due to having no runtime allocations or dynamic representation.
However, it excels at creating tiny binaries for pretrained, deterministic neural networks.

While training will be supported, the more natural use case is for inference on pretrained models in a constrained environment.

ZFFNN compiles to a static binary for any defined user network, so it can fit and run well on microcontrollers, single-board computers (Raspberry Pi), and anywhere else where space and predictability are imperative.

Loading pretrained model definitons from training libs like PyTorch is in progress.
<!-- It is also able to load arbitrary network definitions from pretrained models and create itself at compile time, given a binary that defines weights and bias (typically from PyTorch). (in progress) -->

This gives it the ability to do inference on a system where speed, space, and correctness are vital.

## Features:
- Dense NN of arbitrary size built at compile time
- Static memory (zero heap allocations)
- Inference through forward pass
- SIMD via Zig @Vector 
- All operations (matmul, add, sub, etc.) are checked at compile to ensure they are well defined.
- Training via Stochastic Gradient Descent (coming soon)

- Currently supported activations are:
    - ReLU (Leaky ReLU to be added)
    - Sigmoid
    - Softmax
    - others to be added
- Loss functions may be extended, but must be well defined for the shape, and will be checked at compile time.

## Future work:
- Backprop and support for loss functions for training
- Sparse layer support
- Greater optimization through SIMD
- Stripped build mode for inference only (even smaller binary when not training)
- More loss and activation functions
