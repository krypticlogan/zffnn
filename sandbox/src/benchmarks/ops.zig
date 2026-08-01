const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

const SelectedBenchmark = if (std.mem.eql(u8, build_options.op, "relu"))
    @import("ops/relu.zig").Benchmark
else
    @compileError("unknown or unimplemented tensor op benchmark: " ++ build_options.op);

const iterations = build_options.iterations;
const run_count = build_options.runs;
const warmup_iterations = build_options.warmup_iterations;

comptime {
    if (iterations == 0) @compileError("benchmark iterations must be greater than zero");
    if (run_count == 0) @compileError("benchmark runs must be greater than zero");
}

pub fn main(init: std.process.Init) void {
    const clock = std.Io.Clock.awake;
    var selected = SelectedBenchmark.init();

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
    const work_items_per_second = invocations_per_second * SelectedBenchmark.work_items_per_invocation;
    const bytes_per_second = invocations_per_second * SelectedBenchmark.bytes_per_invocation;

    std.debug.print(
        \\Operation benchmark
        \\  op: {s}
        \\  optimize: {s}
        \\  warmup iterations: {d}
        \\  timed runs: {d}
        \\  iterations/run: {d}
        \\  timer reads/invocation: {d:.6}
        \\  latency: min={d:.3} ns avg={d:.3} ns max={d:.3} ns stddev={d:.3} ns
        \\  throughput: {d:.3} invocations/s
        \\  work throughput: {d:.3} items/s
        \\  effective bandwidth: {d:.3} GiB/s
        \\
    , .{
        SelectedBenchmark.name,
        @tagName(builtin.mode),
        warmup_iterations,
        run_count,
        iterations,
        @as(f64, 2) / @as(f64, @floatFromInt(iterations)),
        minimum_ns,
        average_ns,
        maximum_ns,
        standard_deviation_ns,
        invocations_per_second,
        work_items_per_second,
        bytes_per_second / (1024 * 1024 * 1024),
    });
}
