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
const iterations = build_options.iterations;
const runs = build_options.runs;
const seed = build_options.seed;
const write_out = build_options.write_out;

const feature_ct = model_def[0][0];
const Net = zffnn.NN(model_def, batch);
var net = Net.new();

const clock = std.Io.Clock.awake;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    
    // defer _ = gpa.deinit();    
    
    std.debug.print("OPTIMIZE={s}\n", .{@tagName(optimize)});
    switch (which) {
        .inference => {
            std.debug.print("Running inference benchmark...\n" ++ "=" ** 50 ++ "\n", .{});
            // const zone = zt.ZoneNC(@src(), "inference", 0x00_FF_00_00 );
            // defer zone.End();
            try benchmark_inference(gpa, io);
        },
        .ops => {
            @compileError("Not yet configured");
        },
    }
}

const out_ct =  model_def[model_def.len-1][0];
var output = Mat(out_ct, batch).create(0);


fn inference_test(io: std.Io, nn: *zffnn.NN(model_def, batch), iters: usize) f64 {
    // const zone = zt.ZoneNC(@src(), "inference test", 0x00_FF_00_99 );
    // defer zone.End();    
    nn.random_init(seed);
    
    var prng = std.Random.Xoshiro256.init(seed);
    const input = Mat(feature_ct, batch).createRandom(&prng);

    const start = clock.now(io).nanoseconds;
    for (0..iters) |_| { // benchmark here
        nn.forward_(input, &output);
        // std.mem.doNotOptimizeAway(output);
    }
    const end = clock.now(io).nanoseconds;
    const total_ns: f64 = @floatFromInt(end - start);
    return total_ns;
}

fn benchmark_inference(allocator: std.mem.Allocator, io: std.Io) !void {
    comptime var model_layer_bytes: [model_def.len-1]usize = .{0} ** (model_def.len - 1);
    inline for (model_def[1..], 0..) |layer, i| {
        model_layer_bytes[i] = @sizeOf(f32) * (layer[0] * model_def[i][0] + batch * (layer[0] + model_def[i][0]));
    }
    const model_str = try model2str(allocator, model_def, model_layer_bytes);
    defer allocator.free(model_str);

    var file: ?std.Io.File = null;
    if (write_out) {
        file = try std.Io.Dir.cwd().createFile(io, "benchmarks/zffnn_inference_benchmark.csv", .{ .truncate = false });
        // try file.?.(0);
    }
    defer if (file) |f| f.close(io);

    var total_time_elapsed: f64 = 0;

    var max_batch_latency_ns: f64 = 0;
    var min_batch_latency_ns: f64 = @floatFromInt(std.math.maxInt(usize));
    var total_batch_latency_ns: f64 = 0;

    var max_latency_ns: f64 = 0;
    var min_latency_ns: f64 = @floatFromInt(std.math.maxInt(usize));
    var total_latency_ns: f64 = 0;

    var max_throughput: f64 = 0;
    var min_throughput: f64 = @floatFromInt(std.math.maxInt(usize));
    var total_throughput: f64 = 0;

    for (0..runs) |_| {
        // warm up
        const out = inference_test(io, &net, iterations / 10);
        std.mem.doNotOptimizeAway(out);

        const total_ns = inference_test(io, &net, iterations);
        // const total_ns = benchmark_matmul(iterations, seed);
        total_time_elapsed += total_ns / std.time.ns_per_s;

        const batch_latency_ns = total_ns / iterations;
        if (batch_latency_ns > max_batch_latency_ns) max_batch_latency_ns = batch_latency_ns;
        if (batch_latency_ns < min_batch_latency_ns) min_batch_latency_ns = batch_latency_ns;
        total_batch_latency_ns += batch_latency_ns;

        const total_inferences = iterations * batch;

        const latency_ns = total_ns / total_inferences;
        if (latency_ns > max_latency_ns) max_latency_ns = latency_ns;
        if (latency_ns < min_latency_ns) min_latency_ns = latency_ns;
        total_latency_ns += latency_ns;

        const inferences_per_sec = std.time.ns_per_s / latency_ns;
        if (inferences_per_sec > max_throughput) max_throughput = inferences_per_sec;
        if (inferences_per_sec < min_throughput) min_throughput = inferences_per_sec;
        total_throughput += inferences_per_sec;
    }

    const avg_batch_latency_ns = total_batch_latency_ns / runs;
    const avg_latency_ns = total_latency_ns / runs;
    const avg_throughput = total_throughput / runs;

    const benchmark_output_fmt = benchmark_summary ++ stat_summary ++ stat_summary ++ stat_summary;
    std.debug.print(benchmark_output_fmt, .{ model_str, total_time_elapsed, runs, iterations, batch, "Latency (ns/inference)", min_latency_ns, avg_latency_ns, max_latency_ns, "Batch latency (ns/batch)", min_batch_latency_ns, avg_batch_latency_ns, max_batch_latency_ns, "Throughput (inferences/sec)", min_throughput, avg_throughput, max_throughput });
    std.debug.print("________________________________\n\n", .{});
    if (file) |f| {
        var buf: [2048]u8 = undefined;
        const csv_fmt = "{s},{d},{any},{d},{d:.2},{d:.2},{d:.2},{d:.2},{d:.2},{d:.2}\n";
        const csv_line = try std.fmt.bufPrint(&buf, csv_fmt, .{ model_str, param_ct(model_def), optimize, batch, min_latency_ns, avg_latency_ns, max_latency_ns, min_throughput, avg_throughput, max_throughput });
        try f.writeStreamingAll(io, csv_line);
    }
}


fn benchmark_matmul(comptime iters: usize) f64 {
    var prng = std.Random.Xoshiro256.init(seed);
    const a = Mat(16, 32).createRandom(&prng);
    const b = Mat(32, 64).createRandom(&prng);


    const start = std.time.nanoTimestamp();
    for (0..iters) |_| {
        const c = a.mul(b);
        std.mem.doNotOptimizeAway(c);
    }
    const end = std.time.nanoTimestamp();

    const total_ns: f64 = @floatFromInt(end - start);
    return total_ns;
}
