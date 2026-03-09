//! By convention, root.zig is the root source file when making a library.
const std = @import("std");
const print = std.debug.print;

pub const Activation = union(enum(u8)) {
    none,
    relu,
    sigmoid,
    softmax,
};

fn isMatrix(comptime maybe_mat_type: anytype) bool {
    // @compileLog(maybe_mat_type);
    const fields = std.meta.fields(maybe_mat_type);
    // todo : gate on comptime n and m too
    return std.mem.eql(u8, fields[0].name, "data");
}

test "test_isMatrix" {
    var test_mat = Mat(2, 4).create(0);
    const test_mat_ptr = &test_mat;
    try std.testing.expect(isMatrix(Mat(3, 4)) == true);
    try std.testing.expect(isMatrix(@TypeOf(test_mat)) == true);
    try std.testing.expect(isMatrix(@TypeOf(test_mat_ptr.*)) == true);
    try std.testing.expect(isMatrix(struct {not_data: f32}) == false);
}

pub fn Mat(comptime row_ct: usize, comptime col_ct: usize) type {
    return struct {
        const This = @This();
        pub const n = row_ct;
        pub const m = col_ct;
        const Axis = union(enum) {r, c};

        data: [n]@Vector(m, f32), // row-major [2, 4] is 2 rows x 4 columns
        
        pub fn init(mat: *This, fill_with: f32) void {
            mat.fill(fill_with);
        }

        pub fn create(fill_with: f32) This {
            var mat: Mat(n, m) = undefined;
            mat.fill(fill_with);
            return mat;
        }

        pub fn load(mat: *This, arr_mat: [n][m]f32) void {
            for (0..n) |row| {
                for (0..m) |col| {
                    mat.data[row][col] = arr_mat[row][col];
                }
            }
        }

        pub fn show(mat: *const This) void {
            print("Mat: \n", .{});
            for (mat.data) |row| {
                print("{any}\n", .{row});
            }
            print("\n", .{});
        }

        pub inline fn rows(_: This) usize {
            return n;
        }

        pub inline fn cols(_: This) usize {
            return m;
        }

        pub inline fn set(self: *This, row: usize, col: usize, val: f32) void {
            self.data[row][col] = val;
        }

        pub inline fn get(self: *const This, row: usize, col: usize) f32 {
            return self.data[row][col];
        }

        pub fn randomFill(mat: *This, rand: std.Random) void {
            for (&mat.data) |*row| {
                var i: usize = 0;
                while (i < m) : (i+=1) {
                    row[i] = randomNormalizedFloat(rand);
                }
            }
        }

        pub fn fill(mat: *This, fill_with: f32) void {
            for (&mat.data) |*row| {
                row.* = @splat(fill_with);
            }
        }

        pub fn clear(mat: *This) void {
            mat.* = undefined;
            mat.* = This.create(0);
        }

        fn max_row(row: @Vector(m, f32)) f32 {
            return @reduce(.Max, row);
        }

        pub fn max(mat: *const This) Mat(n, 1) {
            var out = Mat(n, 1).create(0);
            for (0..mat.rows()) |row| {
                out.data[row][0] = max_row(mat.data[row]);
            }
            return out;
        }
        
        fn sum_row(row: @Vector(m, f32)) f32 {
            return @reduce(.Add, row);
        }

        pub fn sum(mat: *const This) Mat(n, 1) {
            var out = Mat(n, 1).create(0);
            for (0..mat.rows()) |row| {
                out.data[row][0] = sum_row(mat.data[row]);
            }
            return out;
        }

        fn exp_row(row: @Vector(m, f32)) @Vector(m, f32) {
            var out: @Vector(m, f32) = undefined;
            for (0..m) |i| {
                out[i] =  @exp(row[i]);
            }
            return out;
        }

        pub fn exp(mat: *const This) This {
            var out = Mat(n, m).create(0);
            for (0..mat.rows()) |row| {
                out.data[row] = exp_row(mat.data[row]);
            }
            return out;
        }

        pub fn t(mat: *const This) Mat(m, n) { // transpose
            var out = Mat(m, n).create(0);
            for (0..mat.rows()) |row| {
                for (0..mat.cols()) |col| {
                    out.set(col, row, mat.get(row, col));
                }
            } 
            return out;
        }
        // todo : inplace variants
        pub fn add(a: *const This, b: anytype) This {
            var out = This.create(0);
            switch (comptime addIsDefined(@TypeOf(a.*), @TypeOf(b))) {
                .full => for (0..out.rows()) |row| { // simd per row due to vector type
                            out.data[row] = a.data[row] + b.data[row];
                        },
                .per_row => for (0..out.rows()) |row| { // simd per row due to vector type
                            const bi: f32 = b.data[row][0];
                            out.data[row] = a.data[row] + @as(@Vector(m, f32), @splat(bi));
                        },
                .none => @compileError("Your add is misaligned, A and B must have matching rows!")
            }     
            return out;
        }

        pub fn sub(a: *const This, b: anytype) This {
            var out = This.create(0);
            switch (comptime addIsDefined(@TypeOf(a.*), @TypeOf(b))) {
                .full => for (0..out.rows()) |row| { // simd per row due to vector type
                            out.data[row] = a.data[row] - b.data[row];
                        },
                .per_row => for (0..out.rows()) |row| { // simd per row due to vector type
                            const bi: f32 = b.data[row][0];
                            out.data[row] = a.data[row] - @as(@Vector(m, f32), @splat(bi));
                        },
                .none => @compileError("Your sub is misaligned, A and B must have matching rows!")
            }     
            return out;
        }

        fn addIsDefined(comptime a_type: anytype, comptime b_type: anytype) union(enum) {none, full, per_row} {
                if (a_type.n == b_type.n and a_type.m == b_type.m) return .full;
                if (a_type.n == b_type.n and b_type.m == 1) return .per_row;
                return .none;
            }

            test "test_isAddDefined" {
                try std.testing.expect(addIsDefined(Mat(2, 4), Mat(2, 4)) == .full);
                try std.testing.expect(addIsDefined(Mat(15, 25), Mat(15, 1)) == .per_row);
                try std.testing.expect(addIsDefined(Mat(15, 25), Mat(15, 2)) == .none);
                try std.testing.expect(addIsDefined(Mat(1, 2), Mat(3, 4)) == .none);
            }

        fn dot(comptime l: usize, a: @Vector(l, f32), b: @Vector(l, f32)) f32 {
            return @reduce(.Add, a*b);
        }

        pub fn mul(a: *const This, b: anytype) Mat(n, @TypeOf(b).m) {
            if (!comptime isMatrix(@TypeOf(a.*)) or !isMatrix(@TypeOf(b))) @compileError("The 'matrix' you provided is not really a matrix");
            if (!comptime mulIsDefined(@TypeOf(a.*), @TypeOf(b))) @compileError("Your multipication is misaligned, B must have the same number of rows as A has columns!");

            var out = Mat(n, @TypeOf(b).m).create(0);
            const b_t = b.t();

            for (0..n) |row| { 
                var out_row: @Vector(@TypeOf(b).m, f32) = undefined;
                for (0..@TypeOf(b).m) |col| {
                    out_row[col] = dot(m, a.data[row], b_t.data[col]);
                }
                out.data[row] = out_row;
            }

            return out;
        }

        fn mulIsDefined(comptime a_type: anytype, comptime b_type: anytype) bool {
            return a_type.m == b_type.n;
        }

        test "test_isMulDefined" {
            try std.testing.expect(mulIsDefined(Mat(4, 2), Mat(2, 5)) == true);
            try std.testing.expect(mulIsDefined(Mat(1, 2), Mat(3, 4)) == false);
        }
                 
        pub fn relu(mat: *const This) This {
            var out = Mat(n, m).create(0);

            for (0..mat.rows()) |row| {
                for (0..mat.cols()) |col| {
                    out.set(row, col, @max(mat.get(row, col), 0));
                }
            }

            return out;
        }

        pub fn sigmoid(mat: *const This) This {
            var out = Mat(n, m).create(0);
            for (0..mat.rows()) |row| {
                for (0..mat.cols()) |col| {
                    out.set(row, col, 1 / (1 + @exp(-mat.get(row, col))));
                }
            }
            return out;
         }
        
        pub fn softmax(mat: *const This) This {
            // We transpose the matrix immediately, so that we may compute softmax per column in, but treat them per row for SIMD purposes
            var out = mat.t(); 

            // Here, we subtract the max value from each element's row before exponentiating to avoid overflow
            for (0..out.rows()) |r| {
                const maxv = @reduce(.Max, out.data[r]);
                out.data[r] -= @as(@Vector(@TypeOf(out).m, f32), @splat(maxv));
            }

            // softmax per row (allowed since transposed)
            const e_mat = out.exp();
            const e_sum = e_mat.sum();
            for (0..out.rows()) |i| {
                const inv = 1.0 / e_sum.get(i, 0);
                out.data[i] = e_mat.data[i] * @as(@Vector(n, f32), @splat(inv));
            }
            // transpose the output again to retain correct shape
            return out.t();
        }

        fn randomNormalizedFloat(rand: std.Random) f32 {
            const rand_float = rand.float(f32);
            return 2 * rand_float - 1;
        }
    };
}

const layer_kind = union(enum) {input, hidden, output};
fn Layer(kind: layer_kind, activation: Activation, comptime len: usize, comptime parent_len: usize, batch_size: usize) type {
    return struct{
        weights: Mat(len, parent_len) = undefined,
        z: Mat(len, batch_size) = undefined,
        a: Mat(len, batch_size) = undefined,
        bias: Mat(len, 1) = undefined,
        activation: Activation = undefined,
        kind: layer_kind,

        /// - pass a random type to this function and the weights and biases will be initalized to deterministic (per rand seed) random floats
        /// - random floats are normalized between -1 and 1
        /// - otherwise, the internals will be 0 initalized
        fn init(layer: *@This()) void {
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
                }
            }
        }

        fn random_fill_wb(self: *@This(), prng: std.Random.Xoshiro256) void {
            if (self.kind == .input) @panic("Input layer has no weights or biases to fill");
            self.bias.randomFill(prng.random());
            self.weights.randomFill(prng.random());
        }

        pub fn forward(self: *@This(), layer_input: Mat(parent_len, batch_size)) void {
            if (self.kind == .input) @panic("Do not pass over the input layer. Complete all normalization and preprocessing prior to forward feed");
            self.z = self.weights.mul(layer_input).add(self.bias);
            self.a = a: switch (self.activation) {
                .relu => self.z.relu(),
                .sigmoid => {
                    break :a self.z.sigmoid();
                },
                .softmax => {
                    break :a self.z.softmax();
                },
                .none => unreachable
            };
        }
    };
}

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

        layers[layer] = Layer(
            kind, 
            layer_def[1], 
            nodes_len,
            parent_len, 
            batch_size
        );
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
        pub fn load_from_bin(comptime param_directory_path: []const u8) This {
            var self: This = new();
            const MAX_USIZE_DIGITS = 20;
            for (1..def.len) |i| {
                comptime var layer_str_buf: [MAX_USIZE_DIGITS]u8 = undefined;
                const layer_str = std.fmt.bufPrint(&layer_str_buf, "{d}", .{i}) catch unreachable;
                const w= std.mem.bytesAsSlice(f32, @embedFile(param_directory_path ++ "/w" ++ layer_str ++ ".bin"));
                const b= std.mem.bytesAsSlice(f32, @embedFile(param_directory_path ++ "/b" ++ layer_str ++ ".bin"));

                // downcast to f16 for storage
                // var w_f16: [w.len]f16 = undefined;
                // var b_f16: [b.len]f16 = undefined;
                // inline for (w, 0..) |val, j| {
                //     w_f16[j] = @floatCast(val);
                // }
                // inline for (b, 0..) |val, j| {
                //     b_f16[j] = @floatCast(val);
                // }
                
                self.layer_from_bin(i, w, b);
                // @compileLog(bin.len);
                // @compileLog(bin[0]);
                // @compileLog(bin[1]);
                // @compileLog(bin[2]);
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
                weights.data[i] = @as(@Vector(weights.cols(), f32), w_bin[offset..offset + weights.cols()].*);
            }
            for (0..bias.rows()) |i| {
                const offset = i * bias.cols();
                bias.data[i] = @as(@Vector(bias.cols(), f32), b_bin[offset..offset + bias.cols()].*);
            }
        }

        pub fn show(self: *@This()) void {
            inline for (0.., self.layers) |i, layer| {
                print("Layer {d} | ", .{i});
                print("Nodes: {any}\n", .{layer.a.rows()});
                print("Layer {d} has {d} nodes and {d} connections\n", .{i, layer.a.rows(), layer.weights.cols() * layer.a.rows()});
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

        pub fn forward(self: *@This(), input: Mat(batch_size, def[0][0])) Mat(def[depth-1][0], batch_size) {
            self.layers[0].a = input.t();
             inline for (1..depth) |i| {
                // @compileLog(i);
                var layer = &self.layers[i];
                // print("layer: {d}\n", .{i});
                // print("layer kind: {any}\n", .{layer.kind});
                const prev_out = self.layers[i - 1].a;
                layer.forward(prev_out);
            }
            return self.layers[depth-1].a;
        }

        pub fn destroy(self: *@This()) void {
            _ = self;
            // self.renderer.destroy();
        }
    };
}
