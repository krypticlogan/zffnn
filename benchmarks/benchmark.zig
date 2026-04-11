const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

const optimize = builtin.mode;

const zffnn = @import("zffnn");

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

fn model2str(allocator: std.mem.Allocator, comptime model: ModelDef) ![]const u8 {
    var model_str: std.ArrayList(u8) = .empty;
    var buf: [20]u8 = undefined;
    try model_str.appendSlice(allocator, @tagName(model[0][1]));
    try model_str.appendSlice(allocator, try std.fmt.bufPrint(&buf, "({d})", .{model[0][0]}));
    for (model[1..]) |layer| { // builds the models structure as a strings
        try model_str.appendSlice(allocator, "-");
        try model_str.appendSlice(allocator, @tagName(layer[1]));
        try model_str.appendSlice(allocator, try std.fmt.bufPrint(&buf, "({d})", .{layer[0]}));
    }
    return try model_str.toOwnedSlice(allocator);
}

const Benchmark = enum {
    inference,
    batch_sweep,
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

// const models: []const ModelDef = &.{ small_model, medium_model, large_model };

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const which: Benchmark = blk: {
        if (comptime std.mem.eql(u8, build_options.benchmark, "inference")) {
            break :blk .inference;
        } else if (comptime std.mem.eql(u8, build_options.benchmark, "batch_sweep")) {
            break :blk .batch_sweep;
        } else if (comptime std.mem.eql(u8, build_options.benchmark, "ops")) {
            break :blk .ops;
        } else {
            @compileError("Unknown benchmark" ++ build_options.benchmark);
        }
    };
    const model: ModelDef = blk: {
        if (comptime std.mem.eql(u8, build_options.model, "small")) {
            break :blk small_model;
        } else if (comptime std.mem.eql(u8, build_options.model, "medium")) {
            break :blk medium_model;
        } else if (comptime std.mem.eql(u8, build_options.model, "large")) {
            break :blk large_model;
        } else {
            @compileError("Unknown model" ++ build_options.model);
        }
    };
    const batch = build_options.batch_size;
    const iterations = build_options.iterations;
    const runs = build_options.runs;
    const seed = build_options.seed;
    const write_out = build_options.write_out;

    std.debug.print("OPTIMIZE={s}\n", .{@tagName(optimize)});
    switch (which) {
        .inference => {
            std.debug.print("Running inference benchmark...\n" ++ "=" ** 50 ++ "\n", .{});
            try benchmark_inference(gpa.allocator(), model, batch, iterations, runs, seed, write_out);
        },
        .batch_sweep => {
            std.debug.print("Running batch sweep benchmark...\n" ++ "=" ** 50 ++ "\n", .{});
            try batch_sweep(gpa.allocator(), model, batch, iterations, runs, seed, write_out);
        },
        .ops => {
            @compileError("Not yet configured");
        },
    }
}

fn inference_test(comptime model: ModelDef, comptime iterations: usize, comptime batch_size: usize, comptime seed: usize) f64 {
    const feature_ct = model[0][0];
    var nn = zffnn.NN(model, batch_size).new();
    nn.random_init(seed);

    var prng = std.Random.Xoshiro256.init(seed);
    const input = Mat(batch_size, feature_ct).createRandom(&prng);

    const start = std.time.nanoTimestamp();
    for (0..iterations) |_| { // benchmark here
        const output = nn.forward(input);
        std.mem.doNotOptimizeAway(output);
    }
    const end = std.time.nanoTimestamp();

    const total_ns: f64 = @floatFromInt(end - start);
    return total_ns;
}

fn benchmark_inference(allocator: std.mem.Allocator, comptime model: ModelDef, comptime batch: usize, comptime iterations: usize, comptime runs: usize, comptime seed: usize, write_out: bool) !void {
    const model_str = try model2str(allocator, model);
    defer allocator.free(model_str);

    var file: ?std.fs.File = null;
    if (write_out) {
        file = try std.fs.cwd().createFile("benchmarks/zffnn_batchsweep_benchmark.csv", .{ .truncate = false });
        try file.?.seekFromEnd(0);
    }
    defer if (file) |f| f.close();

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
        const out = inference_test(model, iterations / 10, batch, seed);
        std.mem.doNotOptimizeAway(out);

        const total_ns = inference_test(model, iterations, batch, seed);
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
        const csv_line = try std.fmt.bufPrint(&buf, csv_fmt, .{ model_str, param_ct(model), optimize, batch, min_latency_ns, avg_latency_ns, max_latency_ns, min_throughput, avg_throughput, max_throughput });
        try f.writeAll(csv_line);
    }
}

fn batch_sweep(allocator: std.mem.Allocator, comptime model: ModelDef, comptime iterations: usize, comptime runs: usize, comptime seed: usize, write_out: bool) !void {
    const model_str = try model2str(allocator, model);
    defer allocator.free(model_str);

    var file: ?std.fs.File = null;
    if (write_out) {
        file = try std.fs.cwd().createFile("benchmarks/zffnn_batchsweep_benchmark.csv", .{ .truncate = false });
        try file.?.seekFromEnd(0);
    }
    defer if (file) |f| f.close();

    // file.writeAll("model,param_ct,optimize,batch_size,latency_min(ns/inference),latency_avg_ns,latency_max,throughput_min(inferences/sec),throughput_avg,throughput_max\n") catch unreachable;

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
    const batch_sizes = [_]usize{ 1, 2, 4, 16, 32, 64, 128, 256, 512, 1024 };
    for (batch_sizes) |batch| {
        // warm up
        const out = inference_test(model, iterations / 10, batch, seed);
        std.mem.doNotOptimizeAway(out);

        for (0..runs) |_| {
            const total_ns = inference_test(model, iterations, batch, seed);
            total_time_elapsed += total_ns;
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
        std.debug.print(benchmark_output_fmt, .{ model_str, total_time_elapsed / std.time.ns_per_s, runs, iterations, batch, "Latency (ns/inference)", min_latency_ns, avg_latency_ns, max_latency_ns, "Batch latency (ns/batch)", min_batch_latency_ns, avg_batch_latency_ns, max_batch_latency_ns, "Throughput (inferences/sec)", min_throughput, avg_throughput, max_throughput });
        std.debug.print("________________________________\n\n", .{});

        if (file) |f| {
            var buf: [2048]u8 = undefined;
            const csv_fmt = "{s},{d},{any},{d},{d:.2},{d:.2},{d:.2},{d:.2},{d:.2},{d:.2}\n";
            const csv_line = try std.fmt.bufPrint(&buf, csv_fmt, .{ model_str, param_ct(model), optimize, batch, min_latency_ns, avg_latency_ns, max_latency_ns, min_throughput, avg_throughput, max_throughput });
            try f.writeAll(csv_line);
        }
    }
}

fn benchmark_matmul(comptime iterations: usize, comptime seed: usize) f64 {
    var prng = std.Random.Xoshiro256.init(seed);
    const a = Mat(16, 32).createRandom(&prng);
    const b = Mat(32, 64).createRandom(&prng);

    const start = std.time.nanoTimestamp();
    for (0..iterations) |_| {
        const c = a.mul(b);
        std.mem.doNotOptimizeAway(c);
    }
    const end = std.time.nanoTimestamp();

    const total_ns: f64 = @floatFromInt(end - start);
    return total_ns;
}
