const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

const optimize = builtin.mode;

const zffnn = @import("zffnn");
// const zt = @import("ztracy");

const Mat = zffnn.Mat;
const Activation = zffnn.Activation;
const ModelDef = []const struct { usize, Activation };

fn param_ct(model: ModelDef) usize {
    var ct: usize = 0;
    for (1..model.len) |layer| {
        ct += model[layer - 1][0] * model[layer][0] + model[layer][0];
    }
    return ct;
}

const benchmark_summary = "Model: {s}\nTotal time elapsed: {d} sec\nRuns per model: {d}\nIterations per run: {d}\nBatch size: {d}\n\n";
const stat_summary = "{s}:\n\tmin: {d:.2}\n\tavg: {d:.2}\n\tmax: {d:.2}\n";

fn model2str(allocator: std.mem.Allocator, comptime model: ModelDef, comptime model_layer_bytes: [model.len-1]usize) ![]const u8 {
    var model_str: std.ArrayList(u8) = .empty;
    var layer_len_buf: [20]u8 = undefined;
    var layer_size_buf: [20]u8 = undefined;
    
    try model_str.appendSlice(allocator, @tagName(model[0][1]));
    try model_str.appendSlice(allocator, try std.fmt.bufPrint(&layer_len_buf, "[{d}]", .{model[0][0]}));
    for (model[1..], model_layer_bytes) |layer, layer_bytes| { // builds the models structure as a strings
        try model_str.appendSlice(allocator, "/");
        try model_str.appendSlice(allocator, @tagName(layer[1]));
        try model_str.appendSlice(allocator, try std.fmt.bufPrint(&layer_len_buf, "[{d}]", .{layer[0]}));
        try model_str.appendSlice(allocator, try std.fmt.bufPrint(&layer_size_buf, "({d}b)", .{layer_bytes}));
    }
    return try model_str.toOwnedSlice(allocator);
}

const Benchmark = enum {
    inference,
    ops,
};

const small_model: ModelDef = &.{
    .{ 32, .none },
    .{ 16, .relu },
    .{ 8, .softmax },
};

const medium_model: ModelDef = &.{
    .{ 128, .none },
    .{ 64, .relu },
    .{ 10, .softmax },
};

const large_model: ModelDef = &.{
    .{ 1024, .none },
    .{ 256, .relu },
    .{ 128, .relu },
    .{ 64, .relu },
    .{ 2, .softmax },
};

const deep_model_1: ModelDef = &.{ // 8 layers
    .{ 1, .none },
    .{ 1, .relu },
    .{ 1, .relu },
    .{ 1, .relu },
    .{ 1, .relu },
    .{ 1, .relu },
    .{ 1, .relu },
    .{ 1, .softmax },
};

const deep_model_16: ModelDef = &.{ // 16 layers
    .{ 16, .none },
    .{ 16, .relu },
    .{ 16, .relu },
    .{ 16, .relu },
    .{ 16, .relu },
    .{ 16, .relu },
    .{ 16, .relu },
    .{ 16, .relu },
    .{ 16, .relu },
    .{ 16, .relu },
    .{ 16, .relu },
    .{ 16, .relu },
    .{ 16, .relu },
    .{ 16, .relu },
    .{ 16, .relu },
    .{ 16, .softmax },
};

const deep_model_64: ModelDef = &.{
    .{ 64, .none },
    .{ 64, .relu },
    .{ 64, .relu },
    .{ 64, .relu },
    .{ 64, .relu },
    .{ 64, .relu },
    .{ 64, .relu },
    .{ 64, .softmax },
};

const wide_model: ModelDef = &.{
    .{ 1024, .none },
    .{ 1024, .relu },
    .{ 1024, .softmax },
};

const hourglass_model: ModelDef = &.{
    .{256, .none},
    .{64, .relu},
    .{16, .relu},
    .{512, .relu},
    .{1024, .softmax},
};

const teepee_model: ModelDef = &.{
    .{256, .none},
    .{64, .relu},
    .{16, .relu},
    .{512, .relu},
    .{2, .softmax},
};

const models: []const ModelDef = &.{ small_model, medium_model, large_model, deep_model_1, deep_model_64, wide_model, hourglass_model };

const which: Benchmark = blk: {
    if (std.mem.eql(u8, build_options.benchmark, "inference")) {
        break :blk .inference;
    } else if (std.mem.eql(u8, build_options.benchmark, "ops")) {
        break :blk .ops;
    } else {
        @compileError("Unknown benchmark" ++ build_options.benchmark);
    }
};

const model_def: ModelDef = blk: {
    if (std.mem.eql(u8, build_options.model, "small")) {
        break :blk small_model;
    } else if (std.mem.eql(u8, build_options.model, "medium")) {
        break :blk medium_model;
    } else if (std.mem.eql(u8, build_options.model, "large")) {
        break :blk large_model;
    } else if (std.mem.eql(u8, build_options.model, "deep_1")) {
        break :blk deep_model_1;
    } else if (std.mem.eql(u8, build_options.model, "deep_64")) {
        break :blk deep_model_64;
    } else if (std.mem.eql(u8, build_options.model, "wide")) {
        break :blk wide_model;
    } else if (std.mem.eql(u8, build_options.model, "hourglass")) {
        break :blk hourglass_model;
    } else if (std.mem.eql(u8, build_options.model, "teepee")) {
        break :blk teepee_model;
    } else if (std.mem.eql(u8, build_options.model, "deep_16")) {
        break :blk deep_model_16;
    } else {
        @compileError("Unknown model" ++ build_options.model);
    }
};
const batch = build_options.batch_size;
const seed = build_options.seed;
const write_out = build_options.write_out;

const feature_ct = model_def[0][0];

const zbench = @import("zbench");

const out_ct =  model_def[model_def.len-1][0];

const InferenceBench = struct {
    const Net = zffnn.NN(model_def, batch);
    var net = Net.new();

    input: Mat(batch, feature_ct),
    output: Mat(out_ct, batch),
    pub fn init() InferenceBench {
       var prng: std.Random.Xoshiro256 = .init(seed);
       net.random_init(seed);
       return InferenceBench { .input = .createRandom(&prng),. output = .create(0) };
    }
  
    pub fn run(self: *InferenceBench, _: std.mem.Allocator) void {  
            const iters = 1000;
            var i: usize = 0;
            while (i < iters) : (i += 1) {
                net.forward_(self.input, &self.output);
                std.mem.doNotOptimizeAway(self.output);   
            }
    }
};

const MatMulBench = struct {
    a: Mat(16, 8),
    b: Mat(8, 24),
    batched: bool = false,
    
    pub fn init(batched: bool) MatMulBench {
       var prng: std.Random.Xoshiro256 = .init(seed);
       return .{ 
           .a = .createRandom(&prng),
           .b = .createRandom(&prng),
           .batched = batched,
       };
    }
  
    pub fn run(self: *MatMulBench, _: std.mem.Allocator) void {  
        const iters = 1000;
        var i: usize = 0;
        while (i < iters) : (i += 1) {
            const out = self.a.mul(&self.b, false);
            std.mem.doNotOptimizeAway(out);   
        }
    }
};

pub fn main(init: std.process.Init) !void {
    const io = init.io; 
    const stdout: std.Io.File = .stdout();
    
    comptime var model_layer_bytes: [model_def.len-1]usize = .{0} ** (model_def.len - 1);
    inline for (model_def[1..], 0..) |layer, i| {
        model_layer_bytes[i] = @sizeOf(f32) * (layer[0] * model_def[i][0] + batch * (layer[0] + model_def[i][0]));
    }
    const model_str = model2str(init.gpa, model_def, model_layer_bytes) catch {
        return;
    };
    defer init.gpa.free(model_str);

    var bench = zbench.Benchmark.init(init.gpa, .{});
    defer bench.deinit();

    std.debug.print("Benchmarks | OPTIMIZE={s}\n", .{@tagName(optimize)});

    try bench.addParam("1k inference", &InferenceBench.init(), .{
        .time_budget_ns = 5 * std.time.ns_per_s
    });
    try bench.addParam("1k MatMul (no batch)", &MatMulBench.init(false), .{
        .time_budget_ns = 5 * std.time.ns_per_s
    });
    try bench.addParam("1k MatMul (batched)", &MatMulBench.init(true), .{
        .time_budget_ns = 5 * std.time.ns_per_s
    });
    try bench.run(io, stdout);
}