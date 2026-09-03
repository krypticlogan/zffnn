const std = @import("std");
const expect = std.testing.expect;
const zgc = @import("zgc");

const Sources = enum(usize) { lhs, rhs, rank_four };
const Definition = zgc.DefinitionBackend(Sources, .{
    .max_rank = 4,
    .max_nodes = 4,
    .max_tensors = 6,
    .max_input_refs = 8,
    .max_outputs = 2,
});
const DefaultSizedDefinition = zgc.DefinitionBackend(Sources, .{ .max_rank = 2 });

test "definition limits default while counting still produces exact capacities" {
    const definition = comptime blk: {
        var builder = DefaultSizedDefinition.init();
        const input = builder.input(.lhs, .f32, &.{ 2, 3 });
        builder.output(builder.relu(input));
        break :blk builder.finish();
    };
    const Compiled = definition.model();

    try std.testing.expectEqual(@as(usize, 1), Compiled.internal_capacity.max_nodes);
    try std.testing.expectEqual(@as(usize, 2), Compiled.internal_capacity.max_tensors);
    try std.testing.expectEqual(@as(usize, 1), Compiled.internal_capacity.max_input_refs);
    try std.testing.expectEqual(@as(usize, 1), Compiled.internal_capacity.max_outputs);
    try std.testing.expectEqual(@as(usize, 2), Compiled.internal_capacity.max_rank);
}

fn buildMatMul(builder: *Definition) void {
    const a = builder.parameter(.lhs, .f32, &.{ 3, 4 });
    const b = builder.parameter(.rhs, .f32, &.{ 4, 7 });
    builder.output(builder.relu(builder.matmul(a, b)));
}

test "definition is counted and lowered without replaying model code" {
    const definition = comptime blk: {
        var builder = Definition.init();
        buildMatMul(&builder);
        break :blk builder.finish();
    };
    const Compiled = definition.model();
    const counts = Compiled.internal_capacity;
    const graph = Compiled.build_graph;

    try expect(graph.node_ct == 2);
    try expect(graph.input_ref_ct == 3);
    try expect(graph.tensor_ct == 4);
    try expect(graph.output_ct == 1);
    try expect(counts.max_nodes == 2);
    try expect(counts.max_tensors == 4);
    try expect(counts.max_input_refs == 3);
    try expect(counts.max_outputs == 1);
    try expect(counts.max_sources == 2);
    try expect(graph.sources[@intFromEnum(Sources.lhs)].?.tensor == 0);
    try expect(graph.sources[@intFromEnum(Sources.rhs)].?.tensor == 1);
    try expect(counts.max_rank == 2);
    try expect(@TypeOf(graph.tensors[0].?.shape).rank_capacity == 2);
    try expect(graph.nodes[0].?.result == 2);
    try expect(graph.input_refs[0].? == 0);
    try expect(graph.input_refs[1].? == 1);
    try expect(graph.nodes[1].?.result == 3);
    try expect(graph.input_refs[2].? == 2);
    try expect(graph.outputs[0].? == 3);
    try expect(graph.tensors[2].?.shape.at(0) == 3);
    try expect(graph.tensors[2].?.shape.at(1) == 7);
    try expect(graph.tensors[0].?.layout.offset == 0);
    try std.testing.expectEqual([2]isize{ 4, 1 }, graph.tensors[0].?.layout.strides);
    try std.testing.expectEqual([2]isize{ 7, 1 }, graph.tensors[2].?.layout.strides);
    try expect(graph.tensors[0].?.storage_tensor == 0);
    try expect(graph.tensors[2].?.storage_tensor == 2);
    try expect(graph.nodes[0].?.kind == .compute);
    try std.testing.expectEqual(
        zgc.Matmul.Strategy.contracted_axis,
        graph.nodes[0].?.op.compute.matmul.strategy,
    );
}

test "lowering selects and propagates a batch-contiguous dense layout" {
    const batch = std.simd.suggestVectorLength(f32) orelse return error.SkipZigTest;
    const definition = comptime blk: {
        var builder = Definition.init();
        const input = builder.input(.lhs, .f32, &.{ batch, 4 });
        const weights = builder.parameter(.rhs, .f32, &.{ 4, 3 });
        const bias = builder.parameter(.rank_four, .f32, &.{3});
        const product = builder.matmul(input, weights);
        const biased = builder.add(product, bias);
        builder.output(builder.relu(biased));
        break :blk builder.finish();
    };
    const graph = definition.model().build_graph;
    const expected = [2]isize{ 1, batch };

    try std.testing.expectEqual(expected, graph.tensors[0].?.layout.strides);
    try std.testing.expectEqual([2]isize{ 1, 4 }, graph.tensors[1].?.layout.strides);
    try std.testing.expectEqual(expected, graph.tensors[3].?.layout.strides);
    try std.testing.expectEqual(expected, graph.tensors[4].?.layout.strides);
    try std.testing.expectEqual(expected, graph.tensors[5].?.layout.strides);
    try std.testing.expectEqual(
        zgc.Matmul.Strategy.output_rows,
        graph.nodes[0].?.op.compute.matmul.strategy,
    );
}

test "counting pass shrinks the user-provided maximum rank" {
    const definition = comptime blk: {
        var builder = Definition.init();
        const tensor = builder.input(.rank_four, .f32, &.{ 2, 3, 4, 5 });
        builder.output(tensor);
        break :blk builder.finish();
    };
    const Compiled = definition.model();
    const counts = Compiled.internal_capacity;
    const graph = Compiled.build_graph;

    try expect(counts.max_rank == 4);
    try expect(counts.max_sources == 3);
    try expect(graph.source_ct == 1);
    try expect(graph.sources[@intFromEnum(Sources.rank_four)].?.tensor == 0);
    try expect(@TypeOf(graph.tensors[0].?.shape).rank_capacity == 4);
    try expect(graph.tensors[0].?.shape.rank == 4);
    try expect(graph.tensors[0].?.shape.at(3) == 5);
}

fn buildTranspose(builder: *Definition) void {
    const input = builder.input(.rank_four, .f32, &.{ 2, 3, 4 });
    builder.output(builder.transpose(input, 0, 2));
}

test "transpose creates an aliasing view node" {
    const definition = comptime blk: {
        var builder = Definition.init();
        buildTranspose(&builder);
        break :blk builder.finish();
    };
    const graph = definition.model().build_graph;
    const source = graph.tensors[0].?;
    const transposed = graph.tensors[1].?;

    try expect(graph.nodes[0].?.kind == .view);
    try expect(transposed.storage_tensor == source.storage_tensor);
    try std.testing.expectEqualSlices(usize, &.{ 4, 3, 2 }, transposed.shape.slice());
    try std.testing.expectEqual([3]isize{ 1, 4, 12 }, transposed.layout.strides);
}

fn buildReduction(builder: *Definition) void {
    const input = builder.input(.rank_four, .f32, &.{ 2, 3, 4 });
    builder.output(builder.softmax(builder.sum(input, 1), 1));
}

test "sum removes its selected axis and softmax preserves the result shape" {
    const definition = comptime blk: {
        var builder = Definition.init();
        buildReduction(&builder);
        break :blk builder.finish();
    };
    const graph = definition.model().build_graph;

    try expect(graph.nodes[0].?.kind == .compute);
    try expect(graph.nodes[1].?.kind == .compute);
    try std.testing.expectEqualSlices(usize, &.{ 2, 4 }, graph.tensors[1].?.shape.slice());
    try std.testing.expectEqualSlices(usize, &.{ 2, 4 }, graph.tensors[2].?.shape.slice());
    try std.testing.expectEqual([3]isize{ 4, 1, 0 }, graph.tensors[1].?.layout.strides);
}

fn buildBroadcastAdd(builder: *Definition) void {
    const matrix = builder.input(.lhs, .f32, &.{ 2, 1, 4 });
    const bias = builder.parameter(.rhs, .f32, &.{ 3, 1 });
    builder.output(builder.add(matrix, bias));
}

test "binary elementwise operations infer their broadcast shape" {
    const definition = comptime blk: {
        var builder = Definition.init();
        buildBroadcastAdd(&builder);
        break :blk builder.finish();
    };
    const Compiled = definition.model();
    const graph = Compiled.build_graph;

    try std.testing.expectEqual(@as(usize, 3), Compiled.internal_capacity.max_rank);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3, 4 }, graph.tensors[2].?.shape.slice());
    try std.testing.expectEqual([3]isize{ 12, 4, 1 }, graph.tensors[2].?.layout.strides);
}
