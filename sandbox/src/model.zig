const std = @import("std");
const zgc = @import("zgc");

pub const Sources = enum(usize) {
    a,
    b
};

fn defineGraph(builder: anytype) void {
    const a = builder.input(Sources.a, .f32, &.{ 2, 3 });
    const b = builder.input(Sources.a, .f32, &.{ 2, 3 });
    const c = builder.add(a, b);
    builder.output(builder.relu(c));
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
pub const a_values = [_]f32{ -3.5, -0.0, 0.0, 2.25, -1.0, 8.0 };
pub const b_values = [_]f32{ -3.5, -0.0, 0.0, 2.25, -1.0, 8.0 };

pub fn loadInput(model: *Model) void {
    const a_bytes: [@sizeOf(@TypeOf(a_values))]u8 = @bitCast(a_values);
    const b_bytes: [@sizeOf(@TypeOf(a_values))]u8 = @bitCast(b_values);
    model.Source(Sources.a, &a_bytes);
    model.Source(Sources.a, &b_bytes);
}

pub fn keepOutputAlive(model: *const Model) void {
    std.mem.doNotOptimizeAway(model.outputView(0).data);
}
