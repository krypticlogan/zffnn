const std = @import("std");
const builtin = @import("builtin");

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

const models: []const ModelDef = &.{ small_model, medium_model, large_model };

pub fn main() !void {
    // const args = std.process.args();
    // const layers = args.next().?;
    // const seed = args.next().?;
    // const layer_size = args.next().?;
    // const batch_size = args.next().?;
    // std.debug.print("Running Infernce Benchmark\n" ++ "=" ** 50 ++ "\n\n", .{});
    // try benchmark_inference();
    std.debug.print("Running Batch Sweep Benchmark\n" ++ "=" ** 50 ++ "\n\n", .{});
    try batch_sweep(large_model);
}

fn batch_sweep(comptime model: ModelDef) !void {
    const file = try std.fs.cwd().createFile("benchmarks/zffnn_batchsweep_benchmark.csv", .{ .truncate = false });
    defer file.close();
    try file.seekFromEnd(0);

    // file.writeAll("model,param_ct,optimize,batch_size,latency_min(ns/inference),latency_avg_ns,latency_max,throughput_min(inferences/sec),throughput_avg,throughput_max\n") catch unreachable;
    const csv_fmt = "{s},{d},{any},{d},{d:.2},{d:.2},{d:.2},{d:.2},{d:.2},{d:.2}\n";

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    // defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const batch_sizes: []const usize = &.{ 1, 4, 16, 32, 64, 128 };
    // const batch_sizes: []const usize = &.{ 128 };
    const iterations_per_batch: []const usize = &.{ 10_000, 500, 500, 250, 250, 100 };
    // const model_iterations: usize = 100_000;
    const seed = 500;
    const runs_per_model = 2;

    inline for (batch_sizes, 0..) |batch_size, batch_idx| {
        const iterations = iterations_per_batch[batch_idx];

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

        // warm up
        const out = inference_test(model, iterations / 10, batch_sizes[0], seed);
        std.mem.doNotOptimizeAway(out);

        for (0..runs_per_model) |_| {
            const total_ns = inference_test(model, iterations, batch_size, seed);
            total_time_elapsed += total_ns;
            total_time_elapsed += total_ns / std.time.ns_per_s;

            const batch_latency_ns = total_ns / iterations;
            if (batch_latency_ns > max_batch_latency_ns) max_batch_latency_ns = batch_latency_ns;
            if (batch_latency_ns < min_batch_latency_ns) min_batch_latency_ns = batch_latency_ns;
            total_batch_latency_ns += batch_latency_ns;

            const total_inferences = iterations * batch_size;

            const latency_ns = total_ns / total_inferences;
            if (latency_ns > max_latency_ns) max_latency_ns = latency_ns;
            if (latency_ns < min_latency_ns) min_latency_ns = latency_ns;
            total_latency_ns += latency_ns;

            const inferences_per_sec = std.time.ns_per_s / latency_ns;
            if (inferences_per_sec > max_throughput) max_throughput = inferences_per_sec;
            if (inferences_per_sec < min_throughput) min_throughput = inferences_per_sec;
            total_throughput += inferences_per_sec;
        }

        const avg_batch_latency_ns = total_batch_latency_ns / runs_per_model;
        const avg_latency_ns = total_latency_ns / runs_per_model;
        const avg_throughput = total_throughput / runs_per_model;

        const benchmark_output_fmt = benchmark_summary ++ stat_summary ++ stat_summary ++ stat_summary;
        std.debug.print(benchmark_output_fmt, .{ try model2str(allocator, model), total_time_elapsed / std.time.ns_per_s, runs_per_model, iterations, batch_size, "Latency (ns/inference)", min_latency_ns, avg_latency_ns, max_latency_ns, "Batch latency (ns/batch)", min_batch_latency_ns, avg_batch_latency_ns, max_batch_latency_ns, "Throughput (inferences/sec)", min_throughput, avg_throughput, max_throughput });
        std.debug.print("________________________________\n\n", .{});

        var buf: [2048]u8 = undefined;
        const csv_line = try std.fmt.bufPrint(&buf, csv_fmt, .{ try model2str(allocator, model), param_ct(model), optimize, batch_size, min_latency_ns, avg_latency_ns, max_latency_ns, min_throughput, avg_throughput, max_throughput });
        try file.writeAll(csv_line);
    }
}

fn benchmark_inference() !void {
    var file = try std.fs.cwd().createFile("benchmarks/zffnn_inference_benchmark.csv", .{ .truncate = false });
    defer file.close();

    try file.seekFromEnd(0);
    // file.writeAll("model,param_ct,optimize,batch_size,latency_min(ns/inference),latency_avg_ns,latency_max,throughput_min(inferences/sec),throughput_avg,throughput_max\n") catch unreachable;
    const csv_fmt = "{s},{d},{any},{d},{d:.2},{d:.2},{d:.2},{d:.2},{d:.2},{d:.2}\n";

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    var buf: [2048]u8 = undefined;

    const batch_size = 32;
    const seed = 500;
    const runs_per_model = 3;

    const flat_model_iterations: []const usize = &.{ 5_000_000, 200_000, 5_000 };
    const batched_model_iterations: []const usize = &.{ 500_000, 50_000, 500 };
    const model_iterations = if (batch_size == 1) flat_model_iterations else batched_model_iterations;

    inline for (models, model_iterations) |model, iterations| {
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

        for (0..runs_per_model) |_| {
            // warm up
            const out = inference_test(model, iterations / 10, batch_size, seed);
            std.mem.doNotOptimizeAway(out);

            const total_ns = inference_test(model, iterations, batch_size, seed);
            // const total_ns = benchmark_matmul(iterations, seed);
            total_time_elapsed += total_ns / std.time.ns_per_s;

            const batch_latency_ns = total_ns / iterations;
            if (batch_latency_ns > max_batch_latency_ns) max_batch_latency_ns = batch_latency_ns;
            if (batch_latency_ns < min_batch_latency_ns) min_batch_latency_ns = batch_latency_ns;
            total_batch_latency_ns += batch_latency_ns;

            const total_inferences = iterations * batch_size;

            const latency_ns = total_ns / total_inferences;
            if (latency_ns > max_latency_ns) max_latency_ns = latency_ns;
            if (latency_ns < min_latency_ns) min_latency_ns = latency_ns;
            total_latency_ns += latency_ns;

            const inferences_per_sec = std.time.ns_per_s / latency_ns;
            if (inferences_per_sec > max_throughput) max_throughput = inferences_per_sec;
            if (inferences_per_sec < min_throughput) min_throughput = inferences_per_sec;
            total_throughput += inferences_per_sec;
        }

        const avg_batch_latency_ns = total_batch_latency_ns / runs_per_model;
        const avg_latency_ns = total_latency_ns / runs_per_model;
        const avg_throughput = total_throughput / runs_per_model;

        const benchmark_output_fmt = benchmark_summary ++ stat_summary ++ stat_summary ++ stat_summary;
        std.debug.print(benchmark_output_fmt, .{ try model2str(allocator, model), total_time_elapsed, runs_per_model, iterations, batch_size, "Latency (ns/inference)", min_latency_ns, avg_latency_ns, max_latency_ns, "Batch latency (ns/batch)", min_batch_latency_ns, avg_batch_latency_ns, max_batch_latency_ns, "Throughput (inferences/sec)", min_throughput, avg_throughput, max_throughput });
        std.debug.print("________________________________\n\n", .{});
        const csv_line = try std.fmt.bufPrint(&buf, csv_fmt, .{ try model2str(allocator, model), param_ct(model), optimize, batch_size, min_latency_ns, avg_latency_ns, max_latency_ns, min_throughput, avg_throughput, max_throughput });
        try file.writeAll(csv_line);
        buf = .{0} ** 2048;
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
