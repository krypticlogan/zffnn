const std = @import("std");
const zgc = @import("zgc");
const embedded_parameters = @import("embed_params");

const Sources = enum(usize) {
    input,
    rank_four,
};
const Definition = zgc.DefinitionBackend(Sources, .{
    .max_rank = 4,
    .max_nodes = 4,
    .max_tensors = 6,
    .max_input_refs = 6,
    .max_outputs = 2,
});

fn buildRelu(builder: *Definition) void {
    const input = builder.input(Sources.input, .f32, &.{ 2, 3 });
    builder.output(builder.relu(input));
}

test "model loads a source, executes relu, and exposes its output" {
    const definition = comptime blk: {
        var builder = Definition.init();
        buildRelu(&builder);
        break :blk builder.finish();
    };
    const ReluModel = definition.model();
    var model = ReluModel.init();

    for (model.memory) |byte| {
        try std.testing.expectEqual(@as(u8, 0), byte);
    }

    const input_values = [_]f32{ -3.5, -0.0, 0.0, 2.25, -1.0, 8.0 };
    try model.copyInput(.input, &input_values);

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

fn buildMixedRankOutputs(builder: *Definition) void {
    const matrix = builder.input(Sources.input, .f32, &.{ 2, 3 });
    const rank_four = builder.input(Sources.rank_four, .f32, &.{ 2, 3, 4, 5 });
    builder.output(matrix);
    builder.output(rank_four);
}

test "model narrows graph layouts to each runtime view rank" {
    const definition = comptime blk: {
        var builder = Definition.init();
        buildMixedRankOutputs(&builder);
        break :blk builder.finish();
    };
    const MixedRankModel = definition.model();
    var model = MixedRankModel.init();
    const matrix = model.outputView(0);
    const rank_four = model.outputView(1);

    try std.testing.expectEqual([2]usize{ 2, 3 }, matrix.shape);
    try std.testing.expectEqual([2]isize{ 3, 1 }, matrix.strides);
    try std.testing.expectEqual([4]usize{ 2, 3, 4, 5 }, rank_four.shape);
    try std.testing.expectEqual([4]isize{ 60, 20, 5, 1 }, rank_four.strides);
}

fn buildTransposeOutput(builder: *Definition) void {
    const input = builder.input(Sources.input, .f32, &.{ 2, 3 });
    builder.output(builder.transpose(input, 0, 1));
}

test "model exposes transpose as a strided view without another allocation" {
    const definition = comptime blk: {
        var builder = Definition.init();
        buildTransposeOutput(&builder);
        break :blk builder.finish();
    };
    const TransposeModel = definition.model();
    var model = TransposeModel.init();
    const input_values = [_]f32{ 1, 2, 3, 4, 5, 6 };
    try model.copyInput(.input, &input_values);
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

fn buildTransposeRelu(builder: *Definition) void {
    const input = builder.input(Sources.input, .f32, &.{ 2, 3 });
    const transposed = builder.transpose(input, 0, 1);
    builder.output(builder.relu(transposed));
}

test "compute kernels consume graph-produced strided views" {
    const definition = comptime blk: {
        var builder = Definition.init();
        buildTransposeRelu(&builder);
        break :blk builder.finish();
    };
    const TransposeReluModel = definition.model();
    var model = TransposeReluModel.init();
    const input_values = [_]f32{ -1, 2, -3, 4, -5, 6 };
    try model.copyInput(.input, &input_values);
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

fn buildTransposeSoftmaxSum(builder: *Definition) void {
    const input = builder.input(Sources.input, .f32, &.{ 2, 3 });
    const transposed = builder.transpose(input, 0, 1);
    const probabilities = builder.softmax(transposed, 1);
    builder.output(builder.sum(probabilities, 1));
}

test "model runs softmax and sum after a strided transpose" {
    const definition = comptime blk: {
        var builder = Definition.init();
        buildTransposeSoftmaxSum(&builder);
        break :blk builder.finish();
    };
    const SoftmaxSumModel = definition.model();
    var model = SoftmaxSumModel.init();
    const input_values = [_]f32{ 1, 2, 3, 4, 5, 6 };
    try model.copyInput(.input, &input_values);
    model.run();

    const output = model.outputView(0);
    try std.testing.expectEqual([1]usize{3}, output.shape);
    for (output.storage) |value| {
        try std.testing.expectApproxEqAbs(@as(f32, 1), value, 1e-6);
    }
}

fn buildBroadcastBiasAdd(builder: *Definition) void {
    const input = builder.input(Sources.input, .f32, &.{ 2, 3 });
    const bias = builder.parameter(Sources.rank_four, .f32, &.{3});
    builder.output(builder.add(input, bias));
}

test "model executes elementwise broadcasting inferred by the graph" {
    const definition = comptime blk: {
        var builder = Definition.init();
        buildBroadcastBiasAdd(&builder);
        break :blk builder.finish();
    };
    const BroadcastModel = definition.model();
    var model = BroadcastModel.init();
    const input_values = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const bias_values = [_]f32{ 10, 20, 30 };
    try model.copyInput(.input, &input_values);
    try model.copySource(.rank_four, &bias_values);
    model.run();

    const output = model.outputView(0);
    try std.testing.expectEqual([2]usize{ 2, 3 }, output.shape);
    try std.testing.expectEqualSlices(
        f32,
        &.{ 11, 22, 33, 14, 25, 36 },
        output.storage,
    );
}

fn buildEmbeddedParameterAdd(builder: *Definition) void {
    const input = builder.input(.input, .f32, &.{ 2, 2 });
    const weights = builder.parameter(.rank_four, .f32, &.{ 2, 2 });
    builder.output(builder.add(input, weights));
}

test "model reads an embedded parameter and copies a typed runtime input" {
    const definition = comptime blk: {
        var builder = Definition.init();
        buildEmbeddedParameterAdd(&builder);
        break :blk builder.finish();
    };
    const EmbeddedModel = definition.modelWith(.{
        .rank_four = zgc.Source.embed(embedded_parameters.weights[0]),
    });
    var model = EmbeddedModel.init();

    // Only the input and compute result occupy mutable model memory. The
    // embedded parameter remains in read-only program data.
    try std.testing.expectEqual(@as(usize, 8 * @sizeOf(f32)), EmbeddedModel.memory_plan.byte_count);

    const input_values = [_]f32{ 1, 2, 3, 4 };
    try model.copyInput(.input, &input_values);
    model.run();

    try std.testing.expectEqualSlices(
        f32,
        &.{ 3, 1, 3.5, 7 },
        model.outputView(0).storage,
    );
}

test "model borrows a typed runtime input without reserving input storage" {
    const definition = comptime blk: {
        var builder = Definition.init();
        buildEmbeddedParameterAdd(&builder);
        break :blk builder.finish();
    };
    const BoundInputModel = definition.modelWith(.{
        .input = zgc.Source.bound,
        .rank_four = zgc.Source.embed(embedded_parameters.weights[0]),
    });
    var model = BoundInputModel.init();

    // Only the compute result is model-owned.
    try std.testing.expectEqual(@as(usize, 4 * @sizeOf(f32)), BoundInputModel.memory_plan.byte_count);
    try std.testing.expect(!model.inputIsBound(.input));

    var input_values = [_]f32{ 1, 2, 3, 4 };
    try model.bindInput(.input, &input_values);
    try std.testing.expect(model.inputIsBound(.input));
    model.run();
    try std.testing.expectEqualSlices(
        f32,
        &.{ 3, 1, 3.5, 7 },
        model.outputView(0).storage,
    );

    // The model borrows the slice, so subsequent runs observe caller updates.
    input_values[0] = 10;
    model.run();
    try std.testing.expectEqualSlices(
        f32,
        &.{ 12, 1, 3.5, 7 },
        model.outputView(0).storage,
    );
}

test "typed input loading rejects an incorrect element count" {
    const definition = comptime blk: {
        var builder = Definition.init();
        buildRelu(&builder);
        break :blk builder.finish();
    };
    const ReluModel = definition.model();
    var model = ReluModel.init();
    const too_short = [_]f32{ 1, 2, 3 };

    try std.testing.expectError(
        error.SourceSizeMismatch,
        model.copyInput(.input, &too_short),
    );
}

test "copied inputs are packed for a lowered batch-contiguous matmul" {
    const Keys = enum(usize) { input, weights };
    const BatchDefinition = zgc.DefinitionBackend(Keys, .{
        .max_rank = 2,
        .max_nodes = 1,
        .max_tensors = 3,
        .max_input_refs = 2,
        .max_outputs = 1,
    });
    const batch = std.simd.suggestVectorLength(f32) orelse return error.SkipZigTest;
    const definition = comptime blk: {
        var builder = BatchDefinition.init();
        const input = builder.input(.input, .f32, &.{ batch, 3 });
        const weights = builder.parameter(.weights, .f32, &.{ 3, 2 });
        builder.output(builder.matmul(input, weights));
        break :blk builder.finish();
    };
    const BatchModel = definition.model();
    var model = BatchModel.init();
    var input_values: [batch * 3]f32 = undefined;
    for (0..batch) |row| {
        input_values[row * 3 ..][0..3].* = .{
            @floatFromInt(row + 1),
            2,
            -1,
        };
    }
    const weights = [_]f32{
        1, 2,
        3, 4,
        5, 6,
    };

    try model.copyInput(.input, &input_values);
    try model.copySource(.weights, &weights);
    model.run();

    try std.testing.expectEqual([2]isize{ 1, batch }, BatchModel.sourceLayout(.input).strides[0..2].*);
    try std.testing.expectEqual([2]isize{ 1, 3 }, BatchModel.sourceLayout(.weights).strides[0..2].*);
    const output = model.outputView(0);
    for (0..batch) |row| {
        const first: f32 = @floatFromInt(row + 2);
        const second: f32 = @floatFromInt(2 * row + 4);
        try std.testing.expectApproxEqAbs(first, output.get(.{ row, 0 }), 1e-6);
        try std.testing.expectApproxEqAbs(second, output.get(.{ row, 1 }), 1e-6);
    }
}

test "logical embedded weights are packed for lowered matmul storage" {
    const Keys = enum(usize) { input, weights };
    const MatmulDefinition = zgc.DefinitionBackend(Keys, .{
        .max_rank = 2,
        .max_nodes = 1,
        .max_tensors = 3,
        .max_input_refs = 2,
        .max_outputs = 1,
    });
    const logical_weights = [_]f32{
        1, 2,
        3, 4,
        5, 6,
    };
    const definition = comptime blk: {
        var builder = MatmulDefinition.init();
        const input = builder.input(.input, .f32, &.{ 1, 3 });
        const weights = builder.parameter(.weights, .f32, &.{ 3, 2 });
        builder.output(builder.matmul(input, weights));
        break :blk builder.finish();
    };
    const EmbeddedMatmul = definition.modelWith(.{
        .weights = zgc.Source.embed(std.mem.asBytes(&logical_weights)),
    });
    var model = EmbeddedMatmul.init();
    try model.copyInput(.input, &[_]f32{ 2, -1, 3 });
    model.run();

    try std.testing.expectEqual([2]isize{ 1, 3 }, EmbeddedMatmul.sourceLayout(.weights).strides[0..2].*);
    try std.testing.expectEqualSlices(f32, &.{ 14, 18 }, model.outputView(0).storage);
}

test "physically packed embedded weights can be consumed without repacking" {
    const Keys = enum(usize) { input, weights };
    const MatmulDefinition = zgc.DefinitionBackend(Keys, .{
        .max_rank = 2,
        .max_nodes = 1,
        .max_tensors = 3,
        .max_input_refs = 2,
        .max_outputs = 1,
    });
    const packed_weights = [_]f32{
        1, 3, 5,
        2, 4, 6,
    };
    const definition = comptime blk: {
        var builder = MatmulDefinition.init();
        const input = builder.input(.input, .f32, &.{ 1, 3 });
        const weights = builder.parameter(.weights, .f32, &.{ 3, 2 });
        builder.output(builder.matmul(input, weights));
        break :blk builder.finish();
    };
    const EmbeddedMatmul = definition.modelWith(.{
        .weights = zgc.Source.embedPacked(std.mem.asBytes(&packed_weights)),
    });
    var model = EmbeddedMatmul.init();
    try model.copyInput(.input, &[_]f32{ 2, -1, 3 });
    model.run();

    try std.testing.expectEqualSlices(f32, &.{ 14, 18 }, model.outputView(0).storage);
}

test "large logical embeddings receive sufficient compile-time packing quota" {
    const Keys = enum(usize) { input, weights };
    const MatmulDefinition = zgc.DefinitionBackend(Keys, .{
        .max_rank = 2,
        .max_nodes = 1,
        .max_tensors = 3,
        .max_input_refs = 2,
        .max_outputs = 1,
    });
    const width = 32;
    const logical_weights: [width * width]f32 = @splat(1);
    const definition = comptime blk: {
        var builder = MatmulDefinition.init();
        const input = builder.input(.input, .f32, &.{ 1, width });
        const weights = builder.parameter(.weights, .f32, &.{ width, width });
        builder.output(builder.matmul(input, weights));
        break :blk builder.finish();
    };
    const EmbeddedMatmul = definition.modelWith(.{
        .weights = zgc.Source.embed(std.mem.asBytes(&logical_weights)),
    });
    var model = EmbeddedMatmul.init();
    const input: [width]f32 = @splat(1);
    try model.copyInput(.input, &input);
    model.run();

    for (model.outputView(0).storage) |value| {
        try std.testing.expectEqual(@as(f32, width), value);
    }
}
