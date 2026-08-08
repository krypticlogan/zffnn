const std = @import("std");
const expect = std.testing.expect;
const zgc = @import("zgc");

const Builder = zgc.Builder;
const GraphBackend = zgc.GraphBackend;
const CountingBackend = zgc.CountingBackend;

const Sources = enum(usize) {
    lhs,
    rhs,
    rank_four,
};

fn buildMatMul(builder: anytype) void {
    const dtype = zgc.Dtype.f32;
    const a = builder.parameter(Sources.lhs, dtype, &.{ 3, 4 });
    const b = builder.parameter(Sources.rhs, dtype, &.{ 4, 7 });
    const product = builder.matmul(a, b);
    const res = builder.relu(product);
    builder.output(res);
}

test "builder.zig" {
    const counts = comptime blk: {
        var backend = CountingBackend{};
        var builder = Builder(CountingBackend){
            .backend = &backend,
        };

        buildMatMul(&builder);
        break :blk backend.counts;
    };

    const graph = comptime blk: {
        var backend = GraphBackend(counts).init();
        var builder = Builder(@TypeOf(backend)){
            .backend = &backend,
        };

        buildMatMul(&builder);
        break :blk backend.finish();
    };

    try expect(graph.node_ct == 2);
    try expect(graph.input_ref_ct == 3);
    try expect(graph.tensor_ct == 4);
    try expect(graph.output_ct == 1);
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
}

test "counting pass discovers maximum shape rank" {
    const counts = comptime blk: {
        var backend = CountingBackend{};
        var builder = Builder(CountingBackend){ .backend = &backend };
        const tensor = builder.input(Sources.rank_four, .f32, &.{ 2, 3, 4, 5 });
        builder.output(tensor);
        break :blk backend.counts;
    };

    const graph = comptime blk: {
        const backend_t = GraphBackend(counts);
        var backend: backend_t = .init();
        var builder = Builder(backend_t){ .backend = &backend };
        const tensor = builder.input(Sources.rank_four, .f32, &.{ 2, 3, 4, 5 });
        builder.output(tensor);
        break :blk backend.finish();
    };

    try expect(counts.max_rank == 4);
    try expect(counts.max_sources == 3);
    try expect(graph.source_ct == 1);
    try expect(graph.sources[@intFromEnum(Sources.rank_four)].?.tensor == 0);
    try expect(@TypeOf(graph.tensors[0].?.shape).rank_capacity == 4);
    try expect(graph.tensors[0].?.shape.rank == 4);
    try expect(graph.tensors[0].?.shape.at(3) == 5);
}

fn buildTranspose(builder: anytype) void {
    const input = builder.input(Sources.rank_four, .f32, &.{ 2, 3, 4 });
    builder.output(builder.transpose(input, 0, 2));
}

test "transpose creates an aliasing view node" {
    const counts = comptime blk: {
        var backend = CountingBackend{};
        var builder = Builder(CountingBackend){ .backend = &backend };
        buildTranspose(&builder);
        break :blk backend.counts;
    };
    const graph = comptime blk: {
        const Backend = GraphBackend(counts);
        var backend = Backend.init();
        var builder = Builder(Backend){ .backend = &backend };
        buildTranspose(&builder);
        break :blk backend.finish();
    };

    const source = graph.tensors[0].?;
    const transposed = graph.tensors[1].?;
    try expect(graph.nodes[0].?.kind == .view);
    try expect(transposed.storage_tensor == source.storage_tensor);
    try std.testing.expectEqual([3]usize{ 4, 3, 2 }, transposed.shape.dims);
    try std.testing.expectEqual([3]isize{ 1, 4, 12 }, transposed.layout.strides);
}

fn buildReduction(builder: anytype) void {
    const input = builder.input(Sources.rank_four, .f32, &.{ 2, 3, 4 });
    const reduced = builder.sum(input, 1);
    builder.output(builder.softmax(reduced, 1));
}

test "sum removes its selected axis and softmax preserves the result shape" {
    const counts = comptime blk: {
        var backend = CountingBackend{};
        var builder = Builder(CountingBackend){ .backend = &backend };
        buildReduction(&builder);
        break :blk backend.counts;
    };
    const graph = comptime blk: {
        const Backend = GraphBackend(counts);
        var backend = Backend.init();
        var builder = Builder(Backend){ .backend = &backend };
        buildReduction(&builder);
        break :blk backend.finish();
    };

    try expect(graph.nodes[0].?.kind == .compute);
    try expect(graph.nodes[1].?.kind == .compute);
    try std.testing.expectEqualSlices(
        usize,
        &.{ 2, 4 },
        graph.tensors[1].?.shape.slice(),
    );
    try std.testing.expectEqualSlices(
        usize,
        &.{ 2, 4 },
        graph.tensors[2].?.shape.slice(),
    );
    try std.testing.expectEqual([3]isize{ 4, 1, 0 }, graph.tensors[1].?.layout.strides);
}

fn buildBroadcastAdd(builder: anytype) void {
    const matrix = builder.input(Sources.lhs, .f32, &.{ 2, 1, 4 });
    const bias = builder.parameter(Sources.rhs, .f32, &.{ 3, 1 });
    builder.output(builder.add(matrix, bias));
}

test "binary elementwise operations infer their broadcast shape" {
    const counts = comptime blk: {
        var backend = CountingBackend{};
        var builder = Builder(CountingBackend){ .backend = &backend };
        buildBroadcastAdd(&builder);
        break :blk backend.counts;
    };
    const graph = comptime blk: {
        const Backend = GraphBackend(counts);
        var backend = Backend.init();
        var builder = Builder(Backend){ .backend = &backend };
        buildBroadcastAdd(&builder);
        break :blk backend.finish();
    };

    try std.testing.expectEqual(@as(usize, 3), counts.max_rank);
    try std.testing.expectEqualSlices(
        usize,
        &.{ 2, 3, 4 },
        graph.tensors[2].?.shape.slice(),
    );
    try std.testing.expectEqual([3]isize{ 12, 4, 1 }, graph.tensors[2].?.layout.strides);
}
