# ZFFNN — Compile-Time Feedforward Neural Networks in Zig

ZFFNN is a statically-defined feedforward neural network library built using Zig's comptime system.

Networks are fully constructed at compile time, with:
- compile-time shape validation
- zero heap allocation
- static memory layout
- deterministic binaries

This library is primarily designed for **inference on pretrained models**, particularly in constrained or embedded environments. 

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
const Net = NN(definition, batch_size);
var nn = comptime Net.load_from_embeds(); // this line requires setup in build.zig 

const preds = nn.forward(input); // inference (everything prior to this is comptime)
preds.show();
```
### Embedding Parameters
Using weights and biases binaries within the library requires some setup.

Parameters must be embedded at compile time to ensure validity and static inference, so this helper is provided to facilitate passing binaries from a directory to your model.
```zig
pub fn addEmbeddedParams(
    b: *std.Build,
    target_mod: *std.Build.Module,
    /// - The files should be named 'w{layer_index}.bin' and 'b{layer_index}.bin' for each layer.
    dir: std.Build.LazyPath,
    /// Number of trainable layers.
    layer_count: usize,
) void {
    const wf = b.addWriteFiles();

    var zig_src: std.ArrayList(u8) = .empty;
    defer zig_src.deinit(b.allocator);
    
    zig_src.writer(b.allocator).writeAll("pub const weights = [_][]const u8{\n") catch unreachable;
    for (1..layer_count + 1) |i| {
        const w_src = dir.path(b, b.fmt("w{d}.bin", .{i}));
        _ = wf.addCopyFile(w_src, b.fmt("w{d}.bin", .{i}));
        zig_src.writer(b.allocator).print("    @embedFile(\"w{d}.bin\"),\n", .{i}) catch unreachable;
    }
    zig_src.writer(b.allocator).writeAll("};\n\n") catch unreachable;

    zig_src.writer(b.allocator).writeAll("pub const biases = [_][]const u8{\n") catch unreachable;
    for (1..layer_count + 1) |i| {
        const b_src = dir.path(b, b.fmt("b{d}.bin", .{i}));
        _ = wf.addCopyFile(b_src, b.fmt("b{d}.bin", .{i}));
        zig_src.writer(b.allocator).print("    @embedFile(\"b{d}.bin\"),\n", .{i}) catch unreachable;
    }
    zig_src.writer(b.allocator).writeAll("};\n") catch unreachable;

    const embeds = wf.add("zffnn_embeds.zig", zig_src.items);
    target_mod.addAnonymousImport("embed_params", .{
        .root_source_file = embeds,
    });
}
```

Use this inside your build.zig file as so after creating your dependency module.
```zig
addEmbeddedParams(
    build, 
    zffnn_mod, 
    b.path("model_params"), 
    trainable_layer_ct
);
```

Inside your exe, you can call:
```zig
var nn = comptime Net.load_from_embeds();
```
as shown before, and there goes your model, ready to recieve input.

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
### Forward Pass
```zig
var nn = Net.new();
const output = nn.forward(input);
```

Input must match: *(batch_size, input_size)*

Output shape is: *(output_size, batch_size)*

## Parameter Loading (Pretrained Models)

ZFFNN supports compile-time embedding of weights and biases.

### File Format

Each layer (starting from 1) must have, so excluding layer 0 (input layer):
```
w1.bin, b1.bin
w2.bin, b2.bin
...
```
Where:
w{i}.bin = flattened row-major weights
b{i}.bin = flattened biases

### Shape Requirements

For layer *i*:

Weights: (layer_size, prev_layer_size)

Bias: (layer_size, 1)

**If shapes do not match, compilation fails.**

### Loading

Set the parameter directory via build options, then:

```zig
var nn = comptime Net.load_from_bin();
```
All parameters are embedded into the binary via ```@embedFile```.

## Matrix API

### Core matrix type: ```Mat(rows, cols)```

#### Operations

```add```, ```sub```

full match OR per-row broadcast *(n, m) + (n, 1)*

```mul```

standard matrix multiplication

```t()```

transpose

```exp```, ```sum```, ```max```

activations:

```relu()```

```sigmoid()```

```softmax()``` (numerically stabilized)

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

