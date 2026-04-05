const std = @import("std");
const Mat = @import("matrix.zig").Mat;
const Activation = @import("activations.zig").Activation;

pub const layer_kind = union(enum) { input, hidden, output };

pub fn Layer(kind: layer_kind, activation: Activation, comptime len: usize, comptime parent_len: usize, batch_size: usize) type {
    return struct {
        weights: Mat(len, parent_len) = undefined,
        z: Mat(len, batch_size) = undefined,
        a: Mat(len, batch_size) = undefined,
        bias: Mat(len, 1) = undefined,
        activation: Activation = undefined,
        kind: layer_kind,

        /// - pass a random type to this function and the weights and biases will be initalized to deterministic (per rand seed) random floats
        /// - random floats are normalized between -1 and 1
        /// - otherwise, the internals will be 0 initalized
        pub fn init(layer: *@This()) void {
            layer.activation = activation;
            layer.kind = kind;
            switch (layer.kind) {
                .hidden, .output => {
                    layer.bias.init(0);
                    layer.weights.init(0);
                    layer.z.init(0);
                    layer.a.init(0);
                },
                .input => {
                    layer.z.init(0);
                    layer.a.init(0);
                },
            }
        }

        pub fn random_fill_wb(self: *@This(), prng: *std.Random.Xoshiro256) void {
            if (self.kind == .input) @panic("Input layer has no weights or biases to fill");
            self.bias.random_fill(prng.random());
            self.weights.random_fill(prng.random());
        }

        pub fn forward(self: *@This(), layer_input: Mat(parent_len, batch_size)) void {
            if (self.kind == .input) @panic("Do not pass over the input layer. Complete all normalization and preprocessing prior to forward feed");
            self.z = self.weights.mul(layer_input).add(self.bias);
            self.a = self.activation.apply(self.z);
        }
    };
}
