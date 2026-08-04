const std = @import("std");
const zgc = @import("zgc");

test "add writes elementwise sums to its output view" {
    var a_data = [_]f32{ -3.5, -1.75, 1.0, 1.25, -1.0, 4.5 };
    var b_data = [_]f32{ 6.5, 3.0, 5.25, 2.25, -1.0, 3.5 };
    var output_data: [a_data.len]f32 = @splat(std.math.nan(f32));

    const a: zgc.Tensor.View(f32, 2) = .{
        .data = &a_data,
        .shape = .{ 2, 3 },
    };
    const b: zgc.Tensor.View(f32, 2) = .{
        .data = &b_data,
        .shape = .{ 2, 3 },
    };
    const output: zgc.Tensor.View(f32, 2) = .{
        .data = &output_data,
        .shape = .{ 2, 3 },
    };

    const op: zgc.Op = .add;
    op.execute(.{ a, b }, output);

    try std.testing.expectEqualSlices(
        f32,
        &.{ 3.0, 1.25, 6.25, 3.5, -2.0, 8.0 },
        &output_data,
    );
}

test "add preserves f16 dtype semantics" {
    var a_data = [_]f16{ -3.5, -1.0, 0.0, 2.25, 4.0, 8.0 };
    var b_data = [_]f16{ 1.5, -2.0, 0.5, 1.75, -1.0, 8.0 };
    var output_data: [a_data.len]f16 = undefined;

    const a: zgc.Tensor.View(f16, 1) = .{
        .data = &a_data,
        .shape = .{a_data.len},
    };
    const b: zgc.Tensor.View(f16, 1) = .{
        .data = &b_data,
        .shape = .{b_data.len},
    };
    const output: zgc.Tensor.View(f16, 1) = .{
        .data = &output_data,
        .shape = .{output_data.len},
    };

    comptime {
        std.debug.assert(@TypeOf(a).dtype == .f16);
        std.debug.assert(@TypeOf(a).scalar_type == f16);
        std.debug.assert(@TypeOf(a).rank == 1);
    }

    const op: zgc.Op = .add;
    op.execute(.{ a, b }, output);

    try std.testing.expectEqualSlices(
        f16,
        &.{ -2.0, -3.0, 0.5, 4.0, 3.0, 16.0 },
        &output_data,
    );
}

test "add preserves i8 dtype semantics without overflow" {
    var a_data = [_]i8{ -100, -3, 0, 2, 100 };
    var b_data = [_]i8{ 20, -4, 0, 5, 27 };
    var output_data: [a_data.len]i8 = undefined;

    const a: zgc.Tensor.View(i8, 1) = .{
        .data = &a_data,
        .shape = .{a_data.len},
    };
    const b: zgc.Tensor.View(i8, 1) = .{
        .data = &b_data,
        .shape = .{b_data.len},
    };
    const output: zgc.Tensor.View(i8, 1) = .{
        .data = &output_data,
        .shape = .{output_data.len},
    };

    comptime std.debug.assert(@TypeOf(a).dtype.kind() == .signed_integer);

    const op: zgc.Op = .add;
    op.execute(.{ a, b }, output);

    try std.testing.expectEqualSlices(i8, &.{ -80, -7, 0, 7, 127 }, &output_data);
}
