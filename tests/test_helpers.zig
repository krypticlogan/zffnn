const std = @import("std");
const testing = std.testing;

pub fn expect_mat_approx_equal(expected: anytype, actual: anytype, tolerance: f32) !void {
    try testing.expectEqual(expected.rows(), actual.rows());
    try testing.expectEqual(expected.cols(), actual.cols());

    for (0..expected.rows()) |row| {
        for (0..expected.cols()) |col| {
            try testing.expectApproxEqAbs(
                expected.get(row, col),
                actual.get(row, col),
                tolerance,
            );
        }
    }
}
