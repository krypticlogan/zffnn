const std = @import("std");
const zgc = @import("zgc");

const Sources = enum(usize) {
    input,
    rank_four,
};

fn buildRelu(builder: anytype) void {
    const input = builder.input(Sources.input, .f32, &.{ 2, 3 });
    builder.output(builder.relu(input));
}

test "model loads a source, executes relu, and exposes its output" {
    const capacities = comptime blk: {
        var backend = zgc.CountingBackend{};
        var builder = zgc.Builder(zgc.CountingBackend){ .backend = &backend };
        buildRelu(&builder);
        break :blk backend.counts;
    };
    const graph = comptime blk: {
        const Backend = zgc.GraphBackend(capacities);
        var backend = Backend.init();
        var builder = zgc.Builder(Backend){ .backend = &backend };
        buildRelu(&builder);
        break :blk backend.finish();
    };

    const ReluModel = zgc.Model(capacities, graph);
    var model = ReluModel.init();

    for (model.memory) |byte| {
        try std.testing.expectEqual(@as(u8, 0), byte);
    }

    const input_values = [_]f32{ -3.5, -0.0, 0.0, 2.25, -1.0, 8.0 };
    const input_bytes: [@sizeOf(@TypeOf(input_values))]u8 = @bitCast(input_values);
    model.Source(Sources.input, &input_bytes);

    const before_run = model.outputView(0);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0, 0, 0 }, before_run.storage);

    model.run();

    const output = model.outputView(0);
    try std.testing.expectEqual([2]usize{ 2, 3 }, output.shape);
    try std.testing.expectEqualSlices(
        f32,
        &.{ 0.0, 0.0, 0.0, 2.25, 0.0, 8.0 },
        output.storage,
    );
}

fn buildMixedRankOutputs(builder: anytype) void {
    const matrix = builder.input(Sources.input, .f32, &.{ 2, 3 });
    const rank_four = builder.input(Sources.rank_four, .f32, &.{ 2, 3, 4, 5 });
    builder.output(matrix);
    builder.output(rank_four);
}

test "model narrows graph layouts to each runtime view rank" {
    const capacities = comptime blk: {
        var backend = zgc.CountingBackend{};
        var builder = zgc.Builder(zgc.CountingBackend){ .backend = &backend };
        buildMixedRankOutputs(&builder);
        break :blk backend.counts;
    };
    const graph = comptime blk: {
        const Backend = zgc.GraphBackend(capacities);
        var backend = Backend.init();
        var builder = zgc.Builder(Backend){ .backend = &backend };
        buildMixedRankOutputs(&builder);
        break :blk backend.finish();
    };

    const MixedRankModel = zgc.Model(capacities, graph);
    var model = MixedRankModel.init();
    const matrix = model.outputView(0);
    const rank_four = model.outputView(1);

    try std.testing.expectEqual([2]usize{ 2, 3 }, matrix.shape);
    try std.testing.expectEqual([2]isize{ 3, 1 }, matrix.strides);
    try std.testing.expectEqual([4]usize{ 2, 3, 4, 5 }, rank_four.shape);
    try std.testing.expectEqual([4]isize{ 60, 20, 5, 1 }, rank_four.strides);
}

fn buildTransposeOutput(builder: anytype) void {
    const input = builder.input(Sources.input, .f32, &.{ 2, 3 });
    builder.output(builder.transpose(input, 0, 1));
}

test "model exposes transpose as a strided view without another allocation" {
    const capacities = comptime blk: {
        var backend = zgc.CountingBackend{};
        var builder = zgc.Builder(zgc.CountingBackend){ .backend = &backend };
        buildTransposeOutput(&builder);
        break :blk backend.counts;
    };
    const graph = comptime blk: {
        const Backend = zgc.GraphBackend(capacities);
        var backend = Backend.init();
        var builder = zgc.Builder(Backend){ .backend = &backend };
        buildTransposeOutput(&builder);
        break :blk backend.finish();
    };

    const TransposeModel = zgc.Model(capacities, graph);
    var model = TransposeModel.init();
    const input_values = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const input_bytes: [@sizeOf(@TypeOf(input_values))]u8 = @bitCast(input_values);
    model.Source(Sources.input, &input_bytes);
    model.run();

    const output = model.outputView(0);
    try std.testing.expectEqual(@sizeOf(@TypeOf(input_values)), TransposeModel.memory_plan.byte_count);
    try std.testing.expectEqual([2]usize{ 3, 2 }, output.shape);
    try std.testing.expectEqual([2]isize{ 1, 3 }, output.strides);
    try std.testing.expectEqual(@as(f32, 1), output.get(.{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 4), output.get(.{ 0, 1 }));
    try std.testing.expectEqual(@as(f32, 2), output.get(.{ 1, 0 }));
    try std.testing.expectEqual(@as(f32, 6), output.get(.{ 2, 1 }));
}

fn buildTransposeRelu(builder: anytype) void {
    const input = builder.input(Sources.input, .f32, &.{ 2, 3 });
    const transposed = builder.transpose(input, 0, 1);
    builder.output(builder.relu(transposed));
}

test "compute kernels consume graph-produced strided views" {
    const capacities = comptime blk: {
        var backend = zgc.CountingBackend{};
        var builder = zgc.Builder(zgc.CountingBackend){ .backend = &backend };
        buildTransposeRelu(&builder);
        break :blk backend.counts;
    };
    const graph = comptime blk: {
        const Backend = zgc.GraphBackend(capacities);
        var backend = Backend.init();
        var builder = zgc.Builder(Backend){ .backend = &backend };
        buildTransposeRelu(&builder);
        break :blk backend.finish();
    };

    const TransposeReluModel = zgc.Model(capacities, graph);
    var model = TransposeReluModel.init();
    const input_values = [_]f32{ -1, 2, -3, 4, -5, 6 };
    const input_bytes: [@sizeOf(@TypeOf(input_values))]u8 = @bitCast(input_values);
    model.Source(Sources.input, &input_bytes);
    model.run();

    const output = model.outputView(0);
    try std.testing.expectEqual([2]usize{ 3, 2 }, output.shape);
    try std.testing.expectEqual([2]isize{ 2, 1 }, output.strides);
    try std.testing.expectEqualSlices(
        f32,
        &.{ 0, 4, 2, 0, 0, 6 },
        output.storage,
    );
}

fn buildTransposeSoftmaxSum(builder: anytype) void {
    const input = builder.input(Sources.input, .f32, &.{ 2, 3 });
    const transposed = builder.transpose(input, 0, 1);
    const probabilities = builder.softmax(transposed, 1);
    builder.output(builder.sum(probabilities, 1));
}

test "model runs softmax and sum after a strided transpose" {
    const capacities = comptime blk: {
        var backend = zgc.CountingBackend{};
        var builder = zgc.Builder(zgc.CountingBackend){ .backend = &backend };
        buildTransposeSoftmaxSum(&builder);
        break :blk backend.counts;
    };
    const graph = comptime blk: {
        const Backend = zgc.GraphBackend(capacities);
        var backend = Backend.init();
        var builder = zgc.Builder(Backend){ .backend = &backend };
        buildTransposeSoftmaxSum(&builder);
        break :blk backend.finish();
    };

    const SoftmaxSumModel = zgc.Model(capacities, graph);
    var model = SoftmaxSumModel.init();
    const input_values = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const input_bytes: [@sizeOf(@TypeOf(input_values))]u8 = @bitCast(input_values);
    model.Source(Sources.input, &input_bytes);
    model.run();

    const output = model.outputView(0);
    try std.testing.expectEqual([1]usize{3}, output.shape);
    for (output.storage) |value| {
        try std.testing.expectApproxEqAbs(@as(f32, 1), value, 1e-6);
    }
}

fn buildBroadcastBiasAdd(builder: anytype) void {
    const input = builder.input(Sources.input, .f32, &.{ 2, 3 });
    const bias = builder.parameter(Sources.rank_four, .f32, &.{3});
    builder.output(builder.add(input, bias));
}

test "model executes elementwise broadcasting inferred by the graph" {
    const capacities = comptime blk: {
        var backend = zgc.CountingBackend{};
        var builder = zgc.Builder(zgc.CountingBackend){ .backend = &backend };
        buildBroadcastBiasAdd(&builder);
        break :blk backend.counts;
    };
    const graph = comptime blk: {
        const Backend = zgc.GraphBackend(capacities);
        var backend = Backend.init();
        var builder = zgc.Builder(Backend){ .backend = &backend };
        buildBroadcastBiasAdd(&builder);
        break :blk backend.finish();
    };

    const BroadcastModel = zgc.Model(capacities, graph);
    var model = BroadcastModel.init();
    const input_values = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const bias_values = [_]f32{ 10, 20, 30 };
    const input_bytes: [@sizeOf(@TypeOf(input_values))]u8 = @bitCast(input_values);
    const bias_bytes: [@sizeOf(@TypeOf(bias_values))]u8 = @bitCast(bias_values);
    model.Source(Sources.input, &input_bytes);
    model.Source(Sources.rank_four, &bias_bytes);
    model.run();

    const output = model.outputView(0);
    try std.testing.expectEqual([2]usize{ 2, 3 }, output.shape);
    try std.testing.expectEqualSlices(
        f32,
        &.{ 11, 22, 33, 14, 25, 36 },
        output.storage,
    );
}
