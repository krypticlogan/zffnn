const std = @import("std");
const zgc = @import("zgc");

test "matmul multiplies contiguous row-major rank-2 tensors" {
    var lhs_data = [_]f32{
        1, 2, 3,
        4, 5, 6,
    };
    var rhs_data = [_]f32{
        7,  8,
        9,  10,
        11, 12,
    };
    var output_data: [4]f32 = @splat(std.math.nan(f32));

    const lhs: zgc.Tensor.View(f32, 2) = .{
        .data = &lhs_data,
        .shape = .{ 2, 3 },
    };
    const rhs: zgc.Tensor.View(f32, 2) = .{
        .data = &rhs_data,
        .shape = .{ 3, 2 },
    };
    const output: zgc.Tensor.View(f32, 2) = .{
        .data = &output_data,
        .shape = .{ 2, 2 },
    };

    const op: zgc.Op = .matmul;
    op.execute(.{ lhs, rhs }, output);

    try std.testing.expectEqualSlices(
        f32,
        &.{ 58, 64, 139, 154 },
        &output_data,
    );
}

test "matmul handles a native SIMD chunk followed by a column tail" {
    const vector_len = std.simd.suggestVectorLength(f32) orelse 1;
    const m = 2;
    const k_len = 3;
    const n = vector_len + 1;

    var lhs_data = [_]f32{
        1, 2, 3,
        4, 5, 6,
    };
    var rhs_data: [k_len * n]f32 = undefined;
    var output_data: [m * n]f32 = @splat(std.math.nan(f32));
    var expected: [m * n]f32 = undefined;

    for (&rhs_data, 0..) |*value, index| {
        value.* = @floatFromInt(index + 1);
    }

    for (0..m) |row| {
        for (0..n) |col| {
            var accumulator: f32 = 0;
            for (0..k_len) |k| {
                accumulator += lhs_data[row * k_len + k] * rhs_data[k * n + col];
            }
            expected[row * n + col] = accumulator;
        }
    }

    const lhs: zgc.Tensor.View(f32, 2) = .{
        .data = &lhs_data,
        .shape = .{ m, k_len },
    };
    const rhs: zgc.Tensor.View(f32, 2) = .{
        .data = &rhs_data,
        .shape = .{ k_len, n },
    };
    const output: zgc.Tensor.View(f32, 2) = .{
        .data = &output_data,
        .shape = .{ m, n },
    };

    const op: zgc.Op = .matmul;
    op.execute(.{ lhs, rhs }, output);

    try std.testing.expectEqualSlices(f32, &expected, &output_data);
}
