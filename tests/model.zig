const std = @import("std");
const zgc = @import("zgc");

const Sources = enum(usize) {
    input,
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
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0, 0, 0 }, before_run.data);

    model.run();

    const output = model.outputView(0);
    try std.testing.expectEqual([2]usize{ 2, 3 }, output.shape);
    try std.testing.expectEqualSlices(
        f32,
        &.{ 0.0, 0.0, 0.0, 2.25, 0.0, 8.0 },
        output.data,
    );
}
