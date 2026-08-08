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
        .storage = &lhs_data,
        .shape = .{ 2, 3 },
        .strides = .{ 3, 1 },
        .offset = 0,
    };
    const rhs: zgc.Tensor.View(f32, 2) = .{
        .storage = &rhs_data,
        .shape = .{ 3, 2 },
        .strides = .{ 2, 1 },
        .offset = 0,
    };
    const output: zgc.Tensor.View(f32, 2) = .{
        .storage = &output_data,
        .shape = .{ 2, 2 },
        .strides = .{ 2, 1 },
        .offset = 0,
    };

    const op: zgc.Op = .{ .compute = .matmul };
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
        .storage = &lhs_data,
        .shape = .{ m, k_len },
        .strides = .{ k_len, 1 },
        .offset = 0,
    };
    const rhs: zgc.Tensor.View(f32, 2) = .{
        .storage = &rhs_data,
        .shape = .{ k_len, n },
        .strides = .{ n, 1 },
        .offset = 0,
    };
    const output: zgc.Tensor.View(f32, 2) = .{
        .storage = &output_data,
        .shape = .{ m, n },
        .strides = .{ n, 1 },
        .offset = 0,
    };

    const op: zgc.Op = .{ .compute = .matmul };
    op.execute(.{ lhs, rhs }, output);

    try std.testing.expectEqualSlices(f32, &expected, &output_data);
}

test "matmul respects contiguous view offsets" {
    var lhs_storage = [_]f32{ 99, 1, 2, 3, 4, 99 };
    var rhs_storage = [_]f32{ 99, 99, 5, 6, 7, 8 };
    var output_storage: [7]f32 = @splat(99);

    const lhs: zgc.Tensor.View(f32, 2) = .{
        .storage = &lhs_storage,
        .shape = .{ 2, 2 },
        .strides = .{ 2, 1 },
        .offset = 1,
    };
    const rhs: zgc.Tensor.View(f32, 2) = .{
        .storage = &rhs_storage,
        .shape = .{ 2, 2 },
        .strides = .{ 2, 1 },
        .offset = 2,
    };
    const output: zgc.Tensor.View(f32, 2) = .{
        .storage = &output_storage,
        .shape = .{ 2, 2 },
        .strides = .{ 2, 1 },
        .offset = 1,
    };

    const op: zgc.Op = .{ .compute = .matmul };
    op.execute(.{ lhs, rhs }, output);

    try std.testing.expectEqualSlices(
        f32,
        &.{ 99, 19, 22, 43, 50, 99, 99 },
        &output_storage,
    );
}
