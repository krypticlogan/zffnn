const std = @import("std");
const testing = std.testing;
const expect = testing.expect;

const zffnn = @import("zffnn");
const mat_equal = @import("tests.zig").mat_equal;

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