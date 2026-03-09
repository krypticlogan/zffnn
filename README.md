# Zig FFNN

## How to build and use
Download and save zffnn as a dependency of the project
```
zig fetch --save git+https://github.com/krypticlogan/zffnn
```

In your build.zig file, load the module as a dependency.
```zig
const nn_dep = b.dependency("zffnn", .{
        .target = target,
        .optimize = optimize,
    });
const zffnn = nn_dep.module("zffnn");
```

Then later on, when creating your target, add the module as an import.
```zig
const exe = b.addExecutable(.{
    .name = "demo",
    .root_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zffnn", .module = zffnn }, // import the module here
        },
        .link_libc = true,
    }),
});
```

## User-defined NN Example for pretrained weights
```zig
const zf = @import("zffnn");
const Activation = zf.Activation;
const NN = zf.NN;

const definition: []const struct { usize, Activation } = &.{ 
    .{feature_ct, .none}, 
    .{10, .relu}, 
    .{15, .sigmoid}, 
    .{2, .softmax}
};
const Net = NN(definition, batch_size);
var nn = comptime Net.load_from_bin(path_to_binaries);

const preds = nn.forward(input); // for inference # everything prior to this is comptime
preds.show();
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

*Poorly defined models (shape mismatch, incorrect use of activations/ops, etc.) will never compile.*

## Philosophy
Unlike PyTorch or TensorFlow, which prioritize being dynamic and flexibile, sometimes a static and predictable system is preferred. 

This engine embraces the static and predictable architecture, where the model is entirely stable at runtime. Changing the network's definition requires recompilation, allowing for the compiler to verify correctness before execution.

This lib is not a replacement for those libraries mentioned previously, but by omitting dynamic control and runtime allocation we gain determinism and simplicity, and retain.
With this, it excels at creating tiny binaries for pretrained, deterministic neural networks and excel in speed through compile time optimizations.

While training will be supported, the natural use case is for inference on pretrained models in a constrained environment.

The network itself compiles to a static binary for any user-defined network, so it is well suited for microcontrollers, single-board computers, and other environments where space and predictability are imperative.

To load a model you trained in PyTorch, load binaries for the weights and biases and they'll be built into the network.
.
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

