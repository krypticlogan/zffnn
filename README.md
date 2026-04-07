# ZFFNN — Compile-Time Feedforward Neural Networks in Zig

A statically-defined feedforward neural network library built using Zig's comptime system.

Networks are fully constructed at compile time, with:
- compile-time shape validation
- zero heap allocation
- static memory layout
- deterministic binaries

This library is primarily designed for **inference on pretrained models**, particularly in constrained or embedded environments. 

A demo is provided for example usage.


#### *Note that this library is currently a work in progress and not fully tested. As such, it is not recommended for use in critical software. Use with caution and if you run into any issues during usage, make me aware and I'll do my best to get it fixed.*

---

## Installation

```bash
zig fetch --save git+https://github.com/krypticlogan/zffnn
```
Add to your build.zig:
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

    }),
});
```
Lastly, import the module in your src exe:
```zig
const zf = @import("zffnn");
```

### Defining a Network

Networks are defined at compile time using a shape + activation specification:
```zig
const Activation = zf.Activation;
const NN = zf.NN;

const definition: []const struct { usize, Activation } = &.{
    .{784, .none},     // input layer (no activation)
    .{128, .relu},     // hidden
    .{10, .softmax},   // output
};

const Net = NN(definition, batch_size);
```
# Usage
## Minimal Flow for inference using pretrained weights
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
const Net = NN(definition, batch_size); // Generated NN type
var nn = comptime Net.load_from_embeds(); // this line requires setup in build.zig 

const preds = nn.forward(input); // inference (everything prior to this is comptime)
preds.show();
```
### Embedding Parameters
Using weights and biases binaries within the library requires some setup.

Parameters must be embedded at compile time to ensure validity and static inference, so a helper artifact is provided to facilitate passing binaries from a local directory to the model.
```zig
{ // Run the embed helper to generate the embeds.zig file
    const params_dir = b.path("model_params");
    const embed_file_name = "embeds.zig";
    
    const zffn_embed_gen = nn_dep.artifact("embed_helper");
    const run_gen = b.addRunArtifact(zffn_embed_gen);
    run_gen.addArg("3"); // number of trainable layers (all but the input), 3 here
    run_gen.addArg(embed_file_name);
    run_gen.addDirectoryArg(params_dir); // directory containing model parameters
    
    const out_dir = run_gen.addOutputDirectoryArg("zffnn_embeds"); // output directory for the generated embeds.zig file

    const embed_mod = b.createModule(.{
        .root_source_file = out_dir.path(b, embed_file_name),
        .target = target,
        .optimize = optimize,
    });
    // add the embed module to the zffnn import, the name "embed_params" must be used
    zffnn.addImport("embed_params", embed_mod);
}
```

Inside your exe, you can call:
```zig
var nn = comptime Net.load_from_embeds();
```
as shown before, and there goes your model, ready to recieve input.

---

# Internals
## Constraints

```definition.len``` >= 2

First layer must use ```.none ``` activation

Each layer defines:

- number of nodes

- activation function

Supported activations:

- ```.relu```

- ```.sigmoid```

- ```.softmax```

## Shape Conventions

This library uses column-major activations internally and batch-major inputs externally.

This makes mathematical operations easier for the engine, but allows users to keep 'normal' representation of the data.

### Input
```zig 
Mat(batch_size, input_size)
```

Where:

One row = one sample

One column = one feature

### Internal Representation

All internal activations are stored as:

```zig
Mat(layer_size, batch_size)
```

So the input is transposed on entry.

### Output
```zig
Mat(output_size, batch_size)
```
---

## Forward Pass
```zig
var nn = Net.new();
const output = nn.forward(input);
```

Input must match: *(batch_size, input_size)*

Output shape is: *(output_size, batch_size)*

## Matrix API

### Core matrix type: ```Mat(rows, cols)```

#### Operations

```add```, ```sub``` full match OR per-row broadcast *(n, m) + (n, 1)*

```mul``` standard matrix multiplication

```t()``` transpose

```exp```, ```sum```, ```max``` per row ops 

```relu()```, ```sigmoid()```, ```softmax()``` activations

All operations are:

- shape-checked at compile time

- allocation-free

### Memory Model
- No heap allocation

- All tensors are stack or static

- Network structure is part of the type

### Compiled binary contains:

- model weights

- full network layout

## What This Library Is (and Isn’t)
### Designed For

- Pretrained inference

- Embedded systems

- Deterministic execution

- Small binary deployments

i.e
- Embedded deterministic NPC logic in games
- Inference on embedded systems (Raspberry Pi with sensors)
- Any constrained environment where determinism and memory are paramount.

### *Not* Designed For

- Dynamic model construction

- Runtime shape changes

- GPU acceleration

- Large-scale training workloads

### Key Differences from PyTorch / TensorFlow

| Feature | ZFFNN | PyTorch / TF |
|----|----|----|
Graph construction | Compile-time | Runtime
Memory | Static	| Dynamic
Shape errors | Compile-time	| Runtime
Model loading | Embedded in binary | Runtime IO
Flexibility	| Low | High
Determinism | Built-in | Varies

### Limitations

- Dense layers only

- No backpropagation (planned)

- No loss functions (planned)

- Limited activation set

- Compile times increase with model size

- No hardware-specific kernel optimization (yet)

### Roadmap

- Backpropagation + training

- Additional activations

- Loss functions

- Sparse layers

- SIMD improvements

- Optional inference-only stripped builds

### Summary

ZFFNN treats neural networks as compile-time constructs, not runtime objects.

This enables:

- stronger guarantees

- simpler execution model

- predictable performance

- minimal runtime overhead

At the cost of:

- flexibility

- compile-time complexity
