const std = @import("std");
const zgc = @import("zgc");

pub const Sources = enum(usize) {
    input,
};

fn defineGraph(builder: anytype) void {
    const input = builder.input(Sources.input, .f32, &.{ 2, 3 });
    builder.output(builder.relu(input));
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
pub const input_values = [_]f32{ -3.5, -0.0, 0.0, 2.25, -1.0, 8.0 };

pub fn loadInput(model: *Model) void {
    const bytes: [@sizeOf(@TypeOf(input_values))]u8 = @bitCast(input_values);
    model.Source(Sources.input, &bytes);
}

pub fn keepOutputAlive(model: *const Model) void {
    std.mem.doNotOptimizeAway(model.outputView(0).data);
}
