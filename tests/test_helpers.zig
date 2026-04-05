const std = @import("std");
const testing = std.testing;
const expect = testing.expect;

const zffnn = @import("zffnn");

pub fn mat_equal(a: anytype, b: anytype) bool {
    if (a.rows() != b.rows() or a.cols() != b.cols()) return false;
    for (a.data, b.data) |a_row, b_row| {
        if (!@import("std").meta.eql(a_row, b_row)) return false;
    }
    return true;
}

test "mat_equal" {
    var a = zffnn.Mat(2, 3).create(0);
    a.load([_][3]f32{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    
    var b = zffnn.Mat(2, 3).create(0);
    b.load([_][3]f32{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });

    var c = zffnn.Mat(2, 3).create(0);
    c.load([_][3]f32{
        .{ 1, 2, 3 },
        .{ 4, 5, 7 },
    });

    try expect(mat_equal(a, b));
    try expect(!mat_equal(a, c));
}