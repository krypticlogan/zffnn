const std = @import("std");
const print = std.debug.print;

const Activation = @import("activations.zig").Activation;
const Mat = @import("matrix.zig").Mat;
const Layer = @import("layer.zig").Layer;
const layer_kind = @import("layer.zig").layer_kind;

/// Generates a new network with the shape specified
/// - The shape should be of a array type
/// - The length of the tuple denotes the depth (number of layers) of the network
/// - Each entry of the tuple denotes how many neurons per layer
pub fn NN(comptime def: []const struct { usize, Activation }, comptime batch_size: usize) type {
    @setEvalBranchQuota(2_000_000_000);
    const depth = def.len;
    comptime var layers: [depth]type = undefined;
    // Create nodes and weights
    // var child_len: usize = 0;
    var parent_len: usize = 0;
    inline for (def, 0..) |layer_def, layer| {
        const nodes_len = layer_def[0];
        if (layer > 0) parent_len = def[layer - 1][0];

        // if (layer < depth - 1) child_len = def[layer + 1][0]
        // else child_len = 0;

        const kind: layer_kind = blk: {
            if (layer == 0) break :blk .input;
            if (layer + 1 != depth) break :blk .hidden;
            break :blk .output;
        };

        layers[layer] = Layer(kind, layer_def[1], nodes_len, parent_len, batch_size);
    }
    const LayersTuple = std.meta.Tuple(&layers);

    return struct {
        const This = @This();
        // var rand_gen: usize = 0;

        layers: LayersTuple = undefined,
        num_nodes: usize = undefined,
        // renderer: Renderer(shape),
        // allocator: std.mem.Allocator,

        /// Build a new predefined NN with the definition provided
        pub fn new() This {
            @setEvalBranchQuota(2_000_000_000);
            const self: This = comptime blk: {
                var tmp: This = undefined;
                tmp.num_nodes = 0;

                for (0.., def) |layer, layer_def| {
                    tmp.num_nodes += layer_def[0];
                    tmp.layers[layer].init();
                }
                break :blk tmp;
            };
            return self;
        }

        pub fn random_init(self: *This, seed: usize) void {
            var prng = std.Random.Xoshiro256.init(seed);
            inline for (1..def.len) |layer| {
                self.layers[layer].random_fill_wb(&prng);
            }
        }

        /// Pass the directory to your pretrained weights and biases and import them to the model here
        /// - Files should be labeled as "w1.bin, b1.bin, w2.bin, ... and so on"
        /// Weights and biases begin at 1 because the 'zeroth' layer is the input layer and does not possess weights or biases
        pub fn load_from_embeds() This {
            const embeds = @import("embed_params");

            var self: This = new();
            for (1..def.len) |i| {
                const w = std.mem.bytesAsSlice(f32, embeds.weights[i - 1]);
                const b = std.mem.bytesAsSlice(f32, embeds.biases[i - 1]);
                // TODO: downcast to f16 for storage
                self.layer_from_bin(i, w, b);
            }
            return self;
        }

        fn layer_from_bin(self: *This, layer: usize, w_bin: []align(1) const f32, b_bin: []align(1) const f32) void {
            var weights = &self.layers[layer].weights;
            var bias = &self.layers[layer].bias;
            if (w_bin.len != weights.rows() * weights.cols()) @compileError("Provided weights do not match the expected shape");
            if (b_bin.len != bias.rows() * bias.cols()) @compileError("Provided biases do not match the expected shape");
            for (0..weights.rows()) |i| {
                const offset = i * weights.cols();
                weights.data[i] = @as(@Vector(weights.cols(), f32), w_bin[offset .. offset + weights.cols()].*);
            }
            for (0..bias.rows()) |i| {
                const offset = i * bias.cols();
                bias.data[i] = @as(@Vector(bias.cols(), f32), b_bin[offset .. offset + bias.cols()].*);
            }
        }

        pub fn show(self: *@This()) void {
            inline for (0.., self.layers) |i, layer| {
                print("Layer {d} | ", .{i});
                print("Nodes: {any}\n", .{layer.a.rows()});
                print("Layer {d} has {d} nodes and {d} connections\n", .{ i, layer.a.rows(), layer.weights.cols() * layer.a.rows() });
                print("\n", .{});
            }
        }

        // pub fn train(nn: *@This(), iterations: usize) void {
        //     const bar_size = 35;
        //     var bar: [bar_size + 1]u8 = undefined;
        //     bar[bar_size] = '|';
        //     var iter: usize = 0;
        //     while (iter < iterations + 1) : (iter+=1) {
        //         // loading bar
        //         const dist: usize = bar_size * iter / iterations;
        //         for (0..bar_size) |i| {
        //             bar[i] = if (i < dist) '*' else '-';
        //         }
        //         print("\riter {d}: {s}", .{iter, bar});
        //         nn.forward();
        //         // backprop
        //     }
        //     print("\n", .{});
        // }

        pub fn forward(self: *@This(), input: Mat(batch_size, def[0][0])) Mat(def[depth - 1][0], batch_size) {
            self.layers[0].a = input.t();
            inline for (1..depth) |i| {
                var layer = &self.layers[i];
                const prev_out = self.layers[i - 1].a;
                layer.forward(prev_out);
            }
            return self.layers[depth - 1].a;
        }
    };
}
