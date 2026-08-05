const std = @import("std");
const zgc = @import("zgc");

pub const Sources = enum(usize) {
    a,
    b
};

fn defineGraph(builder: anytype) void {
    const a = builder.input(Sources.a, .f32, &.{ 3, 3 });
    const b = builder.input(Sources.a, .f32, &.{ 3, 3 });
    const sum = builder.add(a, b);
    const product = builder.matmul(sum, a);
    builder.output(builder.add(product, sum));
}

pub const capacity = blk: {
    var backend = zgc.CountingBackend{};
    var builder = zgc.Builder(zgc.CountingBackend){ .backend = &backend };
    defineGraph(&builder);
    break :blk backend.counts;
};

pub const graph = blk: {
    const Backend = zgc.GraphBackend(capacity);
    var backend = Backend.init();
    var builder = zgc.Builder(Backend){ .backend = &backend };
    defineGraph(&builder);
    break :blk backend.finish();
};

pub const Model = zgc.Model(capacity, graph);
pub const a_values = [_]f32{ 
    -3.5, 2.0, 0.8, 
    2.25, 1.0, 5.2, 
    2.3, 4.2, 7.0 
};
pub const b_values = [_]f32{ -
    2.5, 13.4, 17.3, 
    6.6, 1.0, 5.9, 
    9.3, 4.6, 3.5 
};

pub fn loadInput(model: *Model) void {
    const a_bytes: [@sizeOf(@TypeOf(a_values))]u8 = @bitCast(a_values);
    const b_bytes: [@sizeOf(@TypeOf(a_values))]u8 = @bitCast(b_values);
    model.Source(Sources.a, &a_bytes);
    model.Source(Sources.a, &b_bytes);
}

pub fn keepOutputAlive(model: *const Model) void {
    std.mem.doNotOptimizeAway(model.outputView(0).data);
}
