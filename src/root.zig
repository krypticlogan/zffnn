//! By convention, root.zig is the root source file when making a library.
const std = @import("std");
const print = std.debug.print;



fn mix_max_normalize(mat: Mat) void {
    _ = mat;
}

pub const MatOp = union(enum(u8)) {
    dot,
    add,
    relu,
    sigmoid,
    softmax,
};

fn isMatrix(comptime maybe_mat_type: anytype) bool {
    const fields = std.meta.fields(maybe_mat_type);
    if (fields.len != 1) return false;
    return std.mem.eql(u8, fields[0].name, "data");
}



test "test_isMatrix" {
    var test_mat = Mat(.{2, 4}).create(0);
    const test_mat_ptr = &test_mat;
    try std.testing.expect(isMatrix(@TypeOf(test_mat)) == true);
    try std.testing.expect(isMatrix(@TypeOf(test_mat_ptr.*)) == true);
    try std.testing.expect(isMatrix(struct {not_data: f64}) == false);
}

fn mulIsDefined(comptime a_type: anytype, comptime b_type: anytype) bool {
    return a_type.m == b_type.n;
}

test "test_isMulDefined" {
    try std.testing.expect(mulIsDefined(Mat(.{4, 2}), Mat(.{2, 5})) == true);
    try std.testing.expect(mulIsDefined(Mat(.{1, 2}), Mat(.{3, 4})) == false);
}

fn addIsDefined(comptime a_type: anytype, comptime b_type: anytype) union(enum) {none, full, per_row} {
    if (!isMatrix(a_type) or !isMatrix(b_type)) @compileError("The 'matrix' you provided is not really a matrix");
    // @compileLog(a_type.n, a_type.m, " ", b_type.n, b_type.m);
    if (a_type.n == b_type.n and a_type.m == b_type.m) return .full;
    if (a_type.n == b_type.n) return .per_row;
    return .none;
}

test "test_isAddDefined" {
    // try std.testing.expect(addIsDefined(Mat(.{2, 4}), Mat(.{2, 4})) == true);
    // try std.testing.expect(addIsDefined(Mat(.{1, 2}), Mat(.{3, 4})) == false);
}

pub fn Mat(comptime shape: [2]usize) type {
    
    return struct {
        const This = @This();
        pub const n = shape[0]; // rows
        pub const m = shape[1]; // cols
        const Axis = union(enum) {r, c};

        data: [n]@Vector(m, f64), // row-major [2, 4] is 2 rows x 4 columns
        
        pub fn init(mat: *This, fill_with: f64) void {
            mat.fill(fill_with);
        }

        pub fn create(fill_with: f64) This {
            var mat: Mat(shape) = undefined;
            mat.fill(fill_with);
            // print("new {d}x{d} mat\n", .{mat.rows, mat.cols});
            return mat;
        }

        pub fn load(mat: *This, arr_mat: anytype) void {
            // TODO verify the arr mat is an array at comptime and that it has the same shape as the current matrix
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

        pub inline fn set(self: *This, row: usize, col: usize, val: f64) void {
            self.data[row][col] = val;
        }

        pub inline fn get(self: *const This, row: usize, col: usize) f64 {
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

        pub fn fill(mat: *This, fill_with: f64) void {
            for (&mat.data) |*row| {
                row.* = @splat(fill_with);
            }
        }

        fn max_row(row: @Vector(m, f64)) f64 {
            return @reduce(.Max, row);
        }

        pub fn max(mat: *const This) Mat(.{n, 1}) {
            var out = Mat(.{n, 1}).create(0);
            for (0..mat.rows()) |row| {
                out.data[row][0] = max_row(mat.data[row]);
            }
            return out;
        }
        
        fn sum_row(row: @Vector(m, f64)) f64 {
            return @reduce(.Add, row);
        }

        pub fn sum(mat: *const This) Mat(.{n, 1}) {
            var out = Mat(.{n, 1}).create(0);
            for (0..mat.rows()) |row| {
                out.data[row][0] = sum_row(mat.data[row]);
            }
            return out;
        }

        fn exp_row(row: @Vector(m, f64)) @Vector(m, f64) {
            var out: @Vector(m, f64) = undefined;
            for (0..m) |i| {
                out[i] =  @exp(row[i]);
            }
            return out;
        }

        pub fn exp(mat: *const This) This {
            var out = Mat(shape).create(0);
            for (0..mat.rows()) |row| {
                out.data[row] = exp_row(mat.data[row]);
            }
            return out;
        }

        pub fn t(mat: *const This) Mat(.{m, n}) { // transpose
            var out = Mat(.{m, n}).create(0);
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
                            const bi: f64 = b.data[row][0];
                            out.data[row] = a.data[row] + @as(@Vector(m, f64), @splat(bi));
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
                            const bi: f64 = b.data[row][0];
                            out.data[row] = a.data[row] - @as(@Vector(m, f64), @splat(bi));
                        },
                .none => @compileError("Your sub is misaligned, A and B must have matching rows!")
            }     
            return out;
        }

        fn dot(comptime l: usize, a: @Vector(l, f64), b: @Vector(l, f64)) f64 {
            return @reduce(.Add, a*b);
        }

        pub fn mul(a: *const This, b: anytype) Mat(.{n, @TypeOf(b).m}) {
            if (!comptime isMatrix(@TypeOf(a.*)) or !isMatrix(@TypeOf(b))) @compileError("The 'matrix' you provided is not really a matrix");
            if (!comptime mulIsDefined(@TypeOf(a.*), @TypeOf(b))) @compileError("Your multipication is misaligned, B must have the same number of rows as A has columns!");

            var out = Mat(.{n, @TypeOf(b).m}).create(0);
            const b_t = b.t();

            for (0..n) |row| { 
                var out_row: @Vector(@TypeOf(b).m, f64) = undefined;
                for (0..@TypeOf(b).m) |col| {
                    out_row[col] = dot(m, a.data[row], b_t.data[col]);
                }
                out.data[row] = out_row;
            }

            return out;
        }
                 
        pub fn relu(mat: *const This) This {
            var out = Mat(shape).create(0);

            for (0..mat.rows()) |row| {
                for (0..mat.cols()) |col| {
                    out.set(row, col, @max(mat.get(row, col), 0));
                }
            }

            return out;
        }

        pub fn sigmoid(mat: *const This) This {
            var out = Mat(shape).create(0);
            print("sigmoid\n\n", .{});
            for (0..mat.rows()) |row| {
                for (0..mat.cols()) |col| {
                    out.set(row, col, 1 / (1 + @exp(-mat.get(row, col))));
                }
            }
            return out;
         }

        pub fn softmax(mat: *const This) This {
            print("softmax\n\n", .{});
            var out = mat.t();
            for (0..out.rows()) |r| {
                const maxv = @reduce(.Max, out.data[r]);
                out.data[r] -= @as(@Vector(@TypeOf(out).m, f64), @splat(maxv));
            }
            const e_mat = out.exp();
            const e_sum = e_mat.sum();
            for (0..out.rows()) |i| {
                const inv = 1.0 / e_sum.get(i, 0);
                out.data[i] = e_mat.data[i] * @as(@Vector(n, f64), @splat(inv));
            }
            return out.t();
        }
        fn randomNormalizedFloat(rand: std.Random) f64 {
            const rand_float = rand.float(f64);
            // rng_gen += 1;
            return 2 * rand_float - 1;
        }
    };
}

const layer_kind = union(enum) {input, hidden, output};
fn Layer(class: layer_kind, op: MatOp, comptime len: usize, comptime parent_len: usize, batch_size: usize, input: ?[batch_size][len]f64) type {
    return struct{
        weights: Mat(.{len, parent_len}) = undefined,
        z: Mat(.{len, batch_size}) = undefined,
        a: Mat(.{len, batch_size}) = undefined,
        bias: Mat(.{len, 1}) = undefined,
        op: MatOp = undefined,
        kind: layer_kind,
        

        /// - pass a random type to this function and the weights and biases will be initalized to deterministic (per rand seed) random floats
        /// - random floats are normalized between -1 and 1
        /// - otherwise, the internals will be 0 initalized
        fn init(layer: *@This()) void {
            // print("init layer with op: {any}, len: {d}\n", .{op, len});
            layer.op = op;
            layer.kind = class;
            // layer.nodes.init(0);
            // layer.nodes.randomFill(rand);
            switch (layer.kind) {
                .hidden, .output => {
                    layer.bias.init(0);
                    layer.weights.init(0);
                    layer.z.init(0);
                    layer.a.init(0);
                },
                .input => {
                    layer.z.init(0);
                    var temp = Mat(.{batch_size, len}).create(0);
                    temp.load(input.?);
                    layer.a = temp.t();
                }
            }
        }

        fn random_fill_wb(self: *@This(), rand: std.Random) void {
            self.bias.randomFill(rand);
            self.weights.randomFill(rand);
        }

        pub fn pass(self: *@This(), layer_input: Mat(.{parent_len, batch_size})) void {
            // print("weights {d}x{d}\ninput {d}x{d}\n", .{self.weights.rows(), self.weights.cols(), input.rows(), input.cols()});
            if (self.kind == .input) @panic("Do not pass over the input layer. Complete all normalization and preprocessing prior to forward feed");
            self.z = self.weights.mul(layer_input).add(self.bias);
            self.a = a: switch (self.op) {
                .relu => self.z.relu(),
                .sigmoid => {
                    break :a self.z.sigmoid();
                },
                .softmax => {
                    // if (len > 1) return;
                    break :a self.z.softmax();
                },
                else => unreachable
            };
        }
    };
}

/// Generates a new network with the shape specified
/// - The shape should be of a array type
/// - The length of the tuple denotes the depth (number of layers) of the network
/// - Each entry of the tuple denotes how many neurons per layer
pub fn NN(comptime def: []const struct { usize, MatOp }, comptime batch_size: usize, input: [batch_size][def[0][0]]f64) type {
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
            batch_size, 
            if (kind == .input) input else null
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
        pub fn new() This {
            const self: This = comptime blk: {
                // @compileLog("Creating new network with def");
                // @compileLog(def);
                var tmp: This = undefined;
                tmp.num_nodes = 0;

                for (0.., def) |layer, layer_def| {
                    tmp.num_nodes += layer_def[0];
                    tmp.layers[layer].init();
                }
                // @compileLog("Network Created");
                break :blk tmp;
            };
            return self;
        }

        pub fn random_init(self: *This) void {
            for (0..def.len) |layer| {
                self.layers[layer].random_fill_wb();
            }
        }

        // This functions accepts a parent directory that contains the contents of weights and biases for each layer
        // CSV's should be titled according to the layer and matrix that they refer to. i.e. w1.csv, b2.csv, w3.csv
        // the weights (and biases) labeled with 1 should refer to the first hidden layer of the network as the input layer does not recieve any weights
        pub fn load_from_csv(comptime parent_path: []const u8) This {
            _ = parent_path;
            // const MAX_USIZE_DIGITS = 20;
            var self = new();
            _ = &self;
            inline for (1..depth) |i| {
                // comptime var layer_index_buffer: [MAX_USIZE_DIGITS]u8 = undefined;
                // const layer_index_str = comptime std.fmt.bufPrint(&layer_index_buffer, "{d}", .{i}) catch unreachable;

                // const weights_csv = @embedFile(parent_path ++ "/w" ++ layer_index_str ++ ".csv");
                // const bias_csv = @embedFile(parent_path ++ "/b" ++ layer_index_str ++ ".csv");
                // _ = bias_csv;

                const layer_size = def[i][0];
                const parent_layer_size = def[i - 1][0];
                // const weights_shape: struct {usize, usize} = .{layer_size, parent_layer_size};

                // var weights_iter = comptime std.mem.splitSequence(u8, weights_csv, ",");
                
                // print("Weights shape: {d}x{d}\n", .{layer_size, parent_layer_size});
                
                comptime var j: usize = 0;
                // for (0..parent_layer_size) |_| { 
                while (j < parent_layer_size) : (j+=1) {// bypass the header row
                    // _ = comptime weights_iter.next().?;
                }

                comptime var row: usize = 0;

                // for (0..layer_size) |row| {
                while (row < layer_size) : (row+=1) {
                    var col: usize = 0;
                    while (col < parent_layer_size) : (col+=1) {
                        // @compileLog(j);
                        // const weight_str = comptime weights_iter.next() orelse break;
                        // _ = weight_str;
                    }
                }
            }
            return self;
        }

        pub fn load_from_bin(predefined_model_bin: []u8) @This() {
            var self: @This() = undefined;
            _ = &self;
            _ = predefined_model_bin;
        }

        pub fn show(self: *@This()) void {
            inline for (0.., self.layers) |i, layer| {
                print("Layer {d} | ", .{i});
                print("Nodes: {any}\n", .{layer.nodes});
                print("Layer {d} has {d} nodes and {d} connections\n", .{i, layer.nodes.rows(), layer.weights.cols() * layer.nodes.rows()});
                print("\n", .{});
            }
        }

        // pub fn view(self: *@This()) void {
        //     while (!self.renderer.window.shouldClose()) {
        //         self.renderer.updateAndRender(self);
        //     }
        // }

        pub fn train(nn: *@This(), iterations: usize) void {
            const bar_size = 35;
            var bar: [bar_size + 1]u8 = undefined;
            bar[bar_size] = '|';
            var iter: usize = 0;
            while (iter < iterations + 1) : (iter+=1) {
                // loading bar
                const dist: usize = bar_size * iter / iterations;
                for (0..bar_size) |i| {
                    bar[i] = if (i < dist) '*' else '-';
                }
                print("\riter {d}: {s}", .{iter, bar});
                nn.forward();
                // backprop
            }
            print("\n", .{});
        }

        pub fn forward(self: *@This()) void {
            inline for (1..depth) |i| {
                // @compileLog(i);
                var layer = &self.layers[i];
                const prev_out = self.layers[i - 1].a;
                layer.pass(prev_out);
            }
        }

        

        pub fn destroy(self: *@This()) void {
            _ = self;
            // self.renderer.destroy();
        }
    };
}
