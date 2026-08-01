const std = @import("std");
const zgc = @import("zgc");

test "relu writes clamped values to its output view" {
    var input_data = [_]f32{ -3.5, -0.0, 0.0, 2.25, -1.0, 8.0 };
    var output_data: [input_data.len]f32 = @splat(std.math.nan(f32));

    const input: zgc.Tensor.View(f32, 2) = .{
        .data = &input_data,
        .shape = .{ 2, 3 },
    };
    const output: zgc.Tensor.View(f32, 2) = .{
        .data = &output_data,
        .shape = .{ 2, 3 },
    };

    const op: zgc.Op = .relu;
    op.execute(.{input}, output);

    try std.testing.expectEqualSlices(
        f32,
        &.{ 0.0, 0.0, 0.0, 2.25, 0.0, 8.0 },
        &output_data,
    );
}

test "relu preserves f16 dtype semantics" {
    var input_data = [_]f16{ -3.5, -0.0, 0.0, 2.25, -1.0, 8.0 };
    var output_data: [input_data.len]f16 = undefined;

    const input: zgc.Tensor.View(f16, 1) = .{
        .data = &input_data,
        .shape = .{input_data.len},
    };
    const output: zgc.Tensor.View(f16, 1) = .{
        .data = &output_data,
        .shape = .{output_data.len},
    };

    comptime {
        std.debug.assert(@TypeOf(input).dtype == .f16);
        std.debug.assert(@TypeOf(input).scalar_type == f16);
        std.debug.assert(@TypeOf(input).rank == 1);
    }

    const op: zgc.Op = .relu;
    op.execute(.{input}, output);

    try std.testing.expectEqualSlices(
        f16,
        &.{ 0.0, 0.0, 0.0, 2.25, 0.0, 8.0 },
        &output_data,
    );
}

test "relu preserves i8 dtype semantics" {
    var input_data = [_]i8{ -128, -3, 0, 2, 127 };
    var output_data: [input_data.len]i8 = undefined;

    const input: zgc.Tensor.View(i8, 1) = .{
        .data = &input_data,
        .shape = .{input_data.len},
    };
    const output: zgc.Tensor.View(i8, 1) = .{
        .data = &output_data,
        .shape = .{output_data.len},
    };

    comptime std.debug.assert(@TypeOf(input).dtype.kind() == .signed_integer);

    const op: zgc.Op = .relu;
    op.execute(.{input}, output);

    try std.testing.expectEqualSlices(i8, &.{ 0, 0, 0, 2, 127 }, &output_data);
}
