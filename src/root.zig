//! By convention, root.zig is the root source file when making a library.
const std = @import("std");
// const opengl = @import("zopengl");
// const glfw = @import("zglfw");
// const zgui = @import("zgui");
const print = std.debug.print;

var rng_gen: usize = 0;
fn randomNormalizedFloat(rand: std.Random) f64 {
    const rand_float = rand.float(f64);
    rng_gen += 1;
    return 2 * rand_float - 1;
}

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

var test_mat = Mat(.{2, 4}).create(0);
var test_mat_ptr = &test_mat;

test "test_isMatrix" {
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
                var i: usize = 0;
                while (i < m) : (i+=1) {
                    row[i] = fill_with;
                }
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
    };
}

const layer_kind = union(enum) {input, hidden, output};
fn Layer(class: layer_kind, op: MatOp, comptime len: usize, comptime parent_len: usize, comptime child_len: usize, batch_size: usize, input: ?[batch_size][len]f64) type {
    _ = child_len;
    return struct{
        weights: Mat(.{len, parent_len}) = undefined,
        z: Mat(.{len, batch_size}) = undefined,
        a: Mat(.{len, batch_size}) = undefined,
        bias: Mat(.{len, 1}) = undefined,
        op: MatOp = undefined,
        kind: layer_kind,
        
        fn init(layer: *@This(), ) void {
            print("init layer with op: {any}, len: {d}\n", .{op, len});
            layer.op = op;
            var prng = std.Random.Xoroshiro128.init(101 + rng_gen);
            const rand = prng.random();
            layer.kind = class;
            // layer.nodes.init(0);
            // layer.nodes.randomFill(rand);

            switch (layer.kind) {
                .hidden, .output => {
                    layer.bias.init(0);
                    layer.bias.randomFill(rand);

                    layer.weights.init(0);
                    layer.weights.randomFill(rand);

                    layer.z.init(0);
                    layer.a.init(0);
                },
                .input => {
                    layer.z.init(0);
                    var temp = Mat(.{batch_size, len}).create(0);
                    temp.load(input.?);
                    layer.a = temp.t();
                    // layer.a.init(0);
                    // layer.a.load(input);
                }
            }
            layer.z.show();
            layer.a.show();
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
    var child_len: usize = 0;
    var parent_len: usize = 0;
    inline for (def, 0..) |layer_def, layer| {
        const nodes_len = layer_def[0];
        if (layer > 0) parent_len = def[layer - 1][0];

        if (layer < depth - 1) child_len = def[layer + 1][0]
        else child_len = 0;

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
            child_len, 
            batch_size, 
            if (kind == .input) input else null
        ); 

        
    }
    const LayersTuple = std.meta.Tuple(&layers);

    return struct {
        layers: LayersTuple = undefined,
        num_nodes: usize = 0,
        // renderer: Renderer(shape),
        // allocator: std.mem.Allocator,

        pub fn new() @This() {
            var self: @This() = undefined;
            inline for (0.., def) |layer, layer_def| {
                self.num_nodes += layer_def[0];
                self.layers[layer].init();
            }


            print("New network with def {any}\n", .{def});
            // const renderer = Renderer(shape);
            // self.renderer = renderer.init(allocator);
            // self.allocator = allocator;
            return self;
        }

        pub fn load(predefined_model_bin: []u8) @This() {
            var self: @This() = undefined;
            _ = &self;
            // self.allocator = allocator;
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

// const Renderer = 
// fn Renderer(comptime nn_def: []const usize) type {
// return struct {
//     allocator: std.mem.Allocator,
//     window: *glfw.Window = undefined,
//     // shape: []const usize = undefined,
    
//     var network_shape: [nn_def.len]usize = undefined;
//     network_shape = blk: {
//         for (elem) |layer_nodes| {
//             nodes += layer_nodes;
//         }
//         break :blk nodes;        
//     }; // possible error when multiple models are loaded at once
//     const num_nodes = blk: {
//         var nodes = 0;
//         for (shape) |layer_nodes| {
//             nodes += layer_nodes;
//         }
//         break :blk nodes;        
//     };

//     fn init(allocator: std.mem.Allocator) Renderer(shape) {
//         glfw.init() catch {
//             @panic("Something happened while initializing glfw");
//         };

//         const gl_version_major: u16 = 4;
//         const gl_version_minor: u16 = 0;
//         glfw.windowHint(.client_api, .opengl_api);
//         glfw.windowHint(.context_version_major, gl_version_major);
//         glfw.windowHint(.context_version_minor, gl_version_minor);
//         glfw.windowHint(.opengl_profile, .opengl_core_profile);
//         glfw.windowHint(.opengl_forward_compat, true);
//         glfw.windowHint(.doublebuffer, true);

//         const win = glfw.Window.create(600, 600, "NN Visualizer", null, null) catch |err| {
//             print("{any}\n", .{err});
//             @panic("Failed to create the window");
//         };

//         glfw.makeContextCurrent(win);
//         glfw.swapInterval(1);

//         opengl.loadCoreProfile (
//             glfw.getProcAddress,
//             gl_version_major,
//             gl_version_minor
//         ) catch @panic("GLFW Proc init failed");

//         zgui.init(allocator);

//         _ = zgui.io.addFontFromFile( "font/CenturyModern.ttf", 32.0);

//         zgui.backend.init(win);
//         zgui.io.setConfigFlags(.{ .viewport_enable = false });
//         return Renderer(shape) {
//             .allocator = allocator,
//             .window = win,
//             // .shape = shape
//         };
        
//     }

//     pub fn destroy(self: *Renderer(shape)) void {
//         zgui.backend.deinit();
//         zgui.deinit();
//         self.window.destroy();
//         glfw.makeContextCurrent(null);
//         glfw.terminate();
//     }

//     // fn node_coord_component_in_screen_space(self: *Renderer(shape)) void {
//     //
//     // }

//     pub fn updateAndRender(self: *Renderer(shape), nn: anytype) void {
//         glfw.pollEvents();
//         // _ = nn;

//         const gl = opengl.bindings; // clear buffer
//         const fb_size = self.window.getFramebufferSize(); // new frame buffer
//         gl.viewport(0, 0, fb_size[0], fb_size[1]);
//         gl.clearBufferfv(gl.COLOR, 0, &[_]f64{ 1.0, 0.7, 0.7, 1.0 });
        
//         const win = self.window.getSize(); // logical units (points)
//         zgui.backend.newFrame(@intCast(win[0]), @intCast(win[1]));

//         const cs = self.window.getContentScale();
//         zgui.io.setDisplayFramebufferScale(cs[0], cs[1]);
//         if (zgui.begin("Controls", .{})) {
//             zgui.text("Tweak params here...", .{});
//             // sliders / buttons etc.
//         }
//         zgui.end();

//         // zgui.spacing();
        
//         // --- NN Graph window ---
//         if (zgui.begin("NN Graph", .{ .flags = .{ .no_scrollbar = true, .no_scroll_with_mouse = true }})) {
//             const canvas_sz = zgui.getContentRegionAvail(); // size available in window
//             const canvas_pos0 = zgui.getCursorScreenPos(); // top-left in screen space
//             const canvas_pos1: [2]f64 = .{canvas_pos0[0] + canvas_sz[0], canvas_pos0[1] + canvas_sz[1]}; // bottom-right in screen space
//             const canvas_center_x, const canvas_center_y = .{canvas_pos0[0] + canvas_sz[0] / 2, canvas_pos0[1] + canvas_sz[1] / 2};

//             _ = zgui.invisibleButton("canvas", .{.w = canvas_sz[0], .h = canvas_sz[1], .flags = .{.mouse_button_right = true, .mouse_button_left = true }});

//             const dl = zgui.getWindowDrawList();

//             // canvas
            
//             dl.addRectFilled(.{ .pmin = canvas_pos0, .pmax = canvas_pos1, .col = white});
//             // dl.addCircleFilled(.{ .p = .{ canvas_center_x, canvas_center_y}, .r = 20.0, .col = red});
            
//             dl.addLine(.{ .p1 = .{canvas_pos0[0], canvas_center_y}, .p2 = .{canvas_pos1[0], canvas_center_y}, .col = red, .thickness = 1.0 });
//             dl.addLine(.{ .p1 = .{canvas_center_x, canvas_pos0[1]}, .p2 = .{canvas_center_x, canvas_pos1[1]}, .col = red, .thickness = 1.0 });
//             const layer_mid: f64 = (@as(f64, @floatFromInt(nn.layers.len)) - 1) / 2;
//             const odd_layers = nn.layers.len % 2 == 1;
             

//             const node_r: f64 = 12.0;
//             const node_spacing: f64 = 10.0 + node_r * 2;
//             const layer_spacing: f64 = 50.0 + node_r * 2;

//             // const node_positions = blk: { 
//                 var temp_pos: [num_nodes][2]f64 = undefined;
//                 // var temp_nodes: [num_nodes]Neuron(comptime weights_len: usize) = undefined;                
//                 var temp_i: usize = 0;
//                 inline for (0.., nn.layers) |i, layer| {
//                     // 3 layers
//                     // +  
//                     //     +
//                     //    
//                     // +
//                     //          +
//                     // +    
//                     //   
//                     //     +
//                     // +
//                     const i_float = @as(f64, @floatFromInt(i));
//                     const nodes = layer.nodes;
                
//                     const node_mid: f64 = (@as(f64, @floatFromInt(nodes.len)) - 1) / 2;
//                     const odd_nodes = nodes.len % 2 == 1;
//                     // print("odd nodes? {any}, nodes mid: {d}\n", .{odd_nodes, node_mid});
//                     const layer_x = switch (odd_layers) {
//                         true => val: {
//                             if (i_float < layer_mid) break :val canvas_center_x - layer_spacing * @abs(i_float - layer_mid);
//                             if (i_float > layer_mid) break :val canvas_center_x + layer_spacing * @abs(i_float - layer_mid);
//                             break :val canvas_center_x;
//                         },
//                         false => val:  {
//                             if (i_float < layer_mid) break :val canvas_center_x - layer_spacing * @abs(i_float - layer_mid);
//                             break :val canvas_center_x + layer_spacing * @abs(i_float - layer_mid);
//                         }
//                     };

//                     // const layer_children_pos: [nodes[0].weights][2]f64;
//                     // pre-compute positions and add to array
//                     for (0.., nodes) |j, node| {
//                         _ = node;
//                         const j_float = @as(f64, @floatFromInt(j));
//                         const node_y = switch (odd_nodes) {
//                             true => val: {
//                                 if (j_float < node_mid) break :val canvas_center_y - node_spacing * @abs(j_float - node_mid);
//                                 if (j_float > node_mid) break :val canvas_center_y + node_spacing * @abs(j_float - node_mid);
//                                 break :val canvas_center_y;
//                             },
//                             false => val:  {
//                                 if (j_float < node_mid) break :val canvas_center_y - node_spacing * @abs(j_float - node_mid);
//                                 break :val canvas_center_y + node_spacing * @abs(j_float - node_mid);
//                             }
//                         };

//                         const node_pos: [2]f64 = .{layer_x, node_y};

                        

//                         temp_pos[temp_i] = node_pos;
//                         // temp_nodes[temp_i] = node;
//                         temp_i += 1;

                        
//                         add_node(dl, node_pos, node_r);
//                     }
//                 }
//                 // break :blk temp_pos;
//             // };

//             // for (node_positions) |pos| {

//                 // add_node(dl, pos, node_r);
//             // }
//         }

//         zgui.end();
//         zgui.backend.draw();
//         self.window.swapBuffers();
//     }

//     // colors (ABGR)
//     // const grayU32 = zgui.colorConvertFloat3ToU32(.{0.7, 0.7, 0.7});
//     const white: u32 = 0xff_ff_ff_ff;
//     const pink: u32 = 0xff_99_99_ff;
//     const gray: u32 = 0xff_99_99_99;
//     const dark_gray: u32 = 0xff_dd_dd_dd;
//     const red: u32 = 0xff_00_00_ff;
//     const green = 0xff_00_ff_00;

//     fn add_node(dl: zgui.DrawList, pos: [2]f64, r: f64) void {
//         dl.addCircleFilled(.{ .p = pos , .r = r, .col = green});
//         dl.addCircle(.{ .p = pos , .r = r, .col = gray, .thickness = 3.0});
//     }

//     fn add_edge(dl: zgui.DrawList, span: [2][2]f64, thickness: f64) void {
//         dl.addLine(.{ .p1 = span[0], .p2 = span[1], .col = dark_gray, .thickness = thickness });
//     }
// };
// }

// test "create new nn" {
//     // print("{any}", .{nn});
//     // try std.testing.expect(add(3, 7) == 10);
// }
