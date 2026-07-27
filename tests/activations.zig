const std = @import("std");
const testing = std.testing;

const zffnn = @import("zffnn");
const expect_mat_approx_equal = @import("test_helpers.zig").expect_mat_approx_equal;

test "none leaves values unchanged" {
    var actual = zffnn.Mat(2, 3).create(0);
    actual.load(.{
        .{ -1, 2, -3 },
        .{ 4, -5, 6 },
    });
    const expected = actual.dupe_like(.clone);

    zffnn.Activation.apply(.none, &actual, false);

    try testing.expectEqualDeep(expected.data, actual.data);
}

test "relu clamps negative values and preserves non-negative values" {
    var actual = zffnn.Mat(2, 3).create(0);
    actual.load(.{
        .{ -1, 2, -3 },
        .{ 4, 0, 6 },
    });
    var expected = zffnn.Mat(2, 3).create(0);
    expected.load(.{
        .{ 0, 2, 0 },
        .{ 4, 0, 6 },
    });

    zffnn.Activation.apply(.relu, &actual, false);

    try testing.expectEqualDeep(expected.data, actual.data);
}

test "sigmoid matches known values and remains finite at extremes" {
    var actual = zffnn.Mat(2, 3).create(0);
    actual.load(.{
        .{ -1, 0, 1 },
        .{ -1000, 2, 1000 },
    });

    zffnn.Activation.apply(.sigmoid, &actual, false);

    try testing.expectApproxEqAbs(@as(f32, 0.26894143), actual.get(0, 0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.5), actual.get(0, 1), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.7310586), actual.get(0, 2), 1e-6);
    try testing.expectEqual(@as(f32, 0), actual.get(1, 0));
    try testing.expectApproxEqAbs(@as(f32, 0.880797), actual.get(1, 1), 1e-6);
    try testing.expectEqual(@as(f32, 1), actual.get(1, 2));

    for (0..actual.rows()) |row| {
        for (0..actual.cols()) |col| {
            try testing.expect(std.math.isFinite(actual.get(row, col)));
        }
    }
}

test "softmax branches agree and remain stable for extreme negative logits" {
    var single = zffnn.Mat(3, 2).create(0);
    single.load(.{
        .{ -1000, -3 },
        .{ -1001, -2 },
        .{ -1002, -1 },
    });
    var batched = single.dupe_like(.clone);

    zffnn.Activation.apply(.softmax, &single, false);
    zffnn.Activation.apply(.softmax, &batched, true);

    try expect_mat_approx_equal(single, batched, 1e-6);

    const sums: [2]f32 = batched.sum_cwise();
    for (sums) |sum| {
        try testing.expectApproxEqAbs(@as(f32, 1), sum, 1e-6);
    }

    for (0..batched.rows()) |row| {
        for (0..batched.cols()) |col| {
            const probability = batched.get(row, col);
            try testing.expect(std.math.isFinite(probability));
            try testing.expect(probability >= 0 and probability <= 1);
        }
    }

    try testing.expectApproxEqAbs(@as(f32, 0.66524094), batched.get(0, 0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.09003057), batched.get(2, 0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.09003057), batched.get(0, 1), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.66524094), batched.get(2, 1), 1e-6);
}
