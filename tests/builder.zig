const std = @import("std");
const expect = std.testing.expect;
const zgc = @import("zgc");

const Builder = zgc.Builder;
const GraphBackend = zgc.GraphBackend;
const CountingBackend = zgc.CountingBackend;

fn buildMatMul(builder: anytype) void {
    const dtype = zgc.Dtype.f32;
    const a = builder.parameter(dtype, &.{ 3, 4 });
    const b = builder.parameter(dtype, &.{ 4, 7 });
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
}

test "counting pass discovers maximum shape rank" {
    const counts = comptime blk: {
        var backend = CountingBackend{};
        var builder = Builder(CountingBackend){ .backend = &backend };
        const tensor = builder.input(.f32, &.{ 2, 3, 4, 5 });
        builder.output(tensor);
        break :blk backend.counts;
    };

    const graph = comptime blk: {
        const backend_t = GraphBackend(counts);
        var backend: backend_t = .init();
        var builder = Builder(backend_t){ .backend = &backend };
        const tensor = builder.input(.f32, &.{ 2, 3, 4, 5 });
        builder.output(tensor);
        break :blk backend.finish();
    };

    try expect(counts.max_rank == 4);
    try expect(@TypeOf(graph.tensors[0].?.shape).rank_capacity == 4);
    try expect(graph.tensors[0].?.shape.rank == 4);
    try expect(graph.tensors[0].?.shape.at(3) == 5);
}
