const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

const run_count = build_options.runs;

comptime {
    if (run_count == 0) @compileError("benchmark runs must be greater than zero");
}

pub fn main(init: std.process.Init) void {
    if (comptime std.mem.eql(u8, build_options.op, "all")) {
        inline for (all_benchmarks) |Benchmark| runBenchmark(Benchmark, init);
        return;
    }
    runBenchmark(selectedBenchmark(), init);
}

fn selectedBenchmark() type {
    const matmul = @import("ops/matmul.zig");
    const elementwise = @import("ops/elementwise.zig");
    const reduction = @import("ops/reduction.zig");

    if (std.mem.eql(u8, build_options.op, "relu")) return @import("ops/relu.zig").Benchmark;
    if (std.mem.eql(u8, build_options.op, "add")) return elementwise.AddContiguous;
    if (std.mem.eql(u8, build_options.op, "add-strided")) return elementwise.AddStrided;
    if (std.mem.eql(u8, build_options.op, "add-broadcast")) return elementwise.AddBroadcast;
    if (std.mem.eql(u8, build_options.op, "exp")) return elementwise.ExpContiguous;
    if (std.mem.eql(u8, build_options.op, "sum")) return reduction.SumContiguous;
    if (std.mem.eql(u8, build_options.op, "sum-strided")) return reduction.SumStrided;
    if (std.mem.eql(u8, build_options.op, "softmax")) return reduction.SoftmaxContiguous;
    if (std.mem.eql(u8, build_options.op, "softmax-strided")) return reduction.SoftmaxStrided;
    if (std.mem.eql(u8, build_options.op, "matmul-32")) return matmul.Square32;
    if (std.mem.eql(u8, build_options.op, "matmul-64")) return matmul.Square64;
    if (std.mem.eql(u8, build_options.op, "matmul-128")) return matmul.Square128;
    if (std.mem.eql(u8, build_options.op, "matmul-rect")) return matmul.Rectangular;
    if (std.mem.eql(u8, build_options.op, "matmul-lhs-strided")) return matmul.LhsStrided;
    if (std.mem.eql(u8, build_options.op, "matmul-rhs-strided")) return matmul.RhsStrided;
    if (std.mem.eql(u8, build_options.op, "matmul-output-strided")) return matmul.OutputStrided;
    @compileError("unknown tensor benchmark: " ++ build_options.op);
}

const all_benchmarks = .{
    @import("ops/relu.zig").Benchmark,
    @import("ops/elementwise.zig").AddContiguous,
    @import("ops/elementwise.zig").AddStrided,
    @import("ops/elementwise.zig").AddBroadcast,
    @import("ops/elementwise.zig").ExpContiguous,
    @import("ops/reduction.zig").SumContiguous,
    @import("ops/reduction.zig").SumStrided,
    @import("ops/reduction.zig").SoftmaxContiguous,
    @import("ops/reduction.zig").SoftmaxStrided,
    @import("ops/matmul.zig").Square32,
    @import("ops/matmul.zig").Square64,
    @import("ops/matmul.zig").Square128,
    @import("ops/matmul.zig").Rectangular,
    @import("ops/matmul.zig").LhsStrided,
    @import("ops/matmul.zig").RhsStrided,
    @import("ops/matmul.zig").OutputStrided,
};

fn runBenchmark(comptime Benchmark: type, init: std.process.Init) void {
    const iterations = if (build_options.iterations == 0)
        Benchmark.default_iterations
    else
        build_options.iterations;
    const warmup_iterations = if (build_options.warmup_iterations == 0)
        Benchmark.default_warmup_iterations
    else
        build_options.warmup_iterations;

    const clock = std.Io.Clock.awake;
    var selected = Benchmark.init();

    selected.run(warmup_iterations);

    var samples_ns: [run_count]f64 = undefined;
    for (&samples_ns) |*sample| {
        const start = clock.now(init.io).nanoseconds;
        selected.run(iterations);
        const end = clock.now(init.io).nanoseconds;
        const elapsed_ns: f64 = @floatFromInt(end - start);
        sample.* = elapsed_ns / @as(f64, @floatFromInt(iterations));
    }

    var minimum_ns = std.math.inf(f64);
    var maximum_ns: f64 = 0;
    var total_ns: f64 = 0;
    for (samples_ns) |sample| {
        minimum_ns = @min(minimum_ns, sample);
        maximum_ns = @max(maximum_ns, sample);
        total_ns += sample;
    }
    const average_ns = total_ns / @as(f64, @floatFromInt(run_count));

    var squared_deviation: f64 = 0;
    for (samples_ns) |sample| {
        const difference = sample - average_ns;
        squared_deviation += difference * difference;
    }
    const standard_deviation_ns = @sqrt(
        squared_deviation / @as(f64, @floatFromInt(run_count)),
    );

    const invocations_per_second = @as(f64, std.time.ns_per_s) / average_ns;
    const work_items_per_second = invocations_per_second * Benchmark.work_items_per_invocation;
    const bytes_per_second = invocations_per_second * Benchmark.bytes_per_invocation;
    const coefficient_of_variation = standard_deviation_ns / average_ns * 100;

    std.debug.print(
        \\Benchmark: {s} ({s})
        \\  configuration  {d} warmup | {d} runs x {d} iterations
        \\  latency (ns)   min {d:>10.3} | avg {d:>10.3} | max {d:>10.3}
        \\  variability    stddev {d:.3} ns | CV {d:.2}%
        \\  throughput     {d:.3} invocations/s | {d:.3} {s}/s | {d:.3} GiB/s
        \\
    , .{
        Benchmark.name,
        @tagName(builtin.mode),
        warmup_iterations,
        run_count,
        iterations,
        minimum_ns,
        average_ns,
        maximum_ns,
        standard_deviation_ns,
        coefficient_of_variation,
        invocations_per_second,
        work_items_per_second,
        Benchmark.work_unit,
        bytes_per_second / (1024 * 1024 * 1024),
    });
}
