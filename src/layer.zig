const std = @import("std");
const Mat = @import("matrix.zig").Mat;
const Activation = @import("activations.zig").Activation;

// pub const layer_kind = union(enum) { input, hidden, output };

pub const Role = union(enum) {
    input,
    hidden,
    output,
    // softmax/output layers?
};

const BatchStandard = enum {
    one, // this is for models of batch size 1, they're super fast and get a special path
    multi, // this is the general path for moderately sized models with batched outputs
};

const medium_model_threshold = 32 * 1000; // b
const large_model_threshold = 512 * 1000; // b
const SizeStandard = enum {
    small, // this is for small models
    medium, // this is for medium models
    large, // this is for large models
};

const ShapeStandard = enum {
  wide,
  skinny,
  square,
};

pub fn Layer(kind: Role, activation: Activation, comptime len: usize, comptime parent_len: usize, batch_size: usize) type {
    return struct {
        // out: len
        // in: parent_len
        // internal width: batch_size
        // 
        // first, we do some classification of the layer's shape and size to determine the best path
        const active_bytes = @sizeOf(f32) * (parent_len * len + batch_size * (parent_len + len)); // sizeof dtype ( out * in + batch_size * (out + in) );
        const size_standard: SizeStandard = if (active_bytes >= large_model_threshold) .large else if (active_bytes >= medium_model_threshold) .medium else .small;
        const batch_standard: BatchStandard = if (batch_size == 1) .one else .multi;
        const shape_standard: ShapeStandard = if (len > parent_len) .wide else if (len < parent_len) .skinny else .square;
        

        weights: Mat(len, parent_len) = undefined,
        a: Mat(len, batch_size) = undefined,
        bias: Mat(len, 1) = undefined,
        activation: Activation = undefined,
        kind: Role,

        /// - pass a random type to this function and the weights and biases will be initalized to deterministic (per rand seed) random floats
        /// - random floats are normalized between -1 and 1
        /// - otherwise, the internals will be 0 initalized
        pub fn init(layer: *@This()) void {
            layer.activation = activation;
            layer.kind = kind;
            switch (layer.kind) {
                .hidden, .output => {
                    // const desc = std.fmt.comptimePrint("Layer.init: {d}x{d} batch={d} kind={any} layer_sz_bytes={any}b regime={any} batch_standard={any}\n", .{parent_len, len, batch_size, layer.kind, active_bytes, size_standard, batch_standard});
                    // @compileLog(desc);
                    layer.bias.init(0);
                    layer.weights.init(0);
                    layer.a.init(0);
                },
                .input => {
                    layer.a.init(0);
                },
            }
        }

        pub fn random_fill_wb(self: *@This(), prng: *std.Random.Xoshiro256) void {
            if (self.kind == .input) @panic("Input layer has no weights or biases to fill");
            self.bias.random_fill(prng.random());
            self.weights.random_fill(prng.random());
        }

        pub fn forward(self: *@This(), layer_input: *const Mat(parent_len, batch_size)) void {
            if (self.kind == .input) @panic("Do not pass over the input layer. Complete all normalization and preprocessing prior to forward feed");
            if (batch_standard == .one) {
                // std.debug.print("fast", .{});
                self.a = self.weights.mul(layer_input, false).add(self.bias);
                self.activation.apply(&self.a, false);
            } else {
                switch (size_standard) {
                    .small => {
                        self.a = self.weights.mul(layer_input, true).add(self.bias);
                        self.activation.apply(&self.a, true);
                    },
                    .medium => {
                        self.weights.mul_into(layer_input, &self.a, true, false);
                        self.a.i_add(self.bias);
                        self.activation.apply(&self.a, true);
                    },
                    .large => {
                        // std.debug.print("large", .{});
                        self.weights.mul_into(layer_input, &self.a, true, true);
                        self.a.i_add(self.bias);
                        self.activation.apply(&self.a, true);
                    },
                }
            }
        }
    };
}
