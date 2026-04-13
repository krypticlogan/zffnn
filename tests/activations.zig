const std = @import("std");
const testing = std.testing;
const expect = testing.expect;

const zffnn = @import("zffnn");
const mat_equal = @import("test_helpers.zig").mat_equal;

test "relu" {
    var a = zffnn.Mat(2, 3).create(0);
    a.load([_][3]f32{
        .{ -1, 2, -3 },
        .{ 4, -5, 6 },
    });
    var expected = zffnn.Mat(2, 3).create(0);
    expected.load([_][3]f32{
        .{ 0, 2, 0 },
        .{ 4, 0, 6 },
    });
    // var relu_a = a.
    zffnn.Activation.apply(.relu, &a, false);
    try expect(mat_equal(a, expected));
}

test "sigmoid" {
    var a = zffnn.Mat(2, 3).create(0);
    a.load([_][3]f32{
        .{ -1, 2, -3 },
        .{ 4, -5, 6 },
    });

    zffnn.Activation.apply(.sigmoid, &a, false);
    try expect(@reduce(.Min, a.data[0]) == a.data[0][2]);
    try expect(@reduce(.Max, a.data[1]) == a.data[1][2]);

    // numerical stability
    try expect(a.data[0][0] == 0.2689414213699951);
    try expect(a.data[1][0] == 0.9820137900379085);
}

test "softmax" {
    var a = zffnn.Mat(2, 3).create(0);
    a.load([_][3]f32{
        .{ -1, 2, -3 },
        .{ 4, -5, 6 },
    });

    zffnn.Activation.apply(.softmax, &a, false);
    for (a.data) |row| { // each row should sum to 1
        const sum = @reduce(.Add, row);
        try expect(1 - sum < 0.01); // accounting for floating point precision
    }

    try expect(@reduce(.Max, a.data[0]) == a.data[0][1]);
    try expect(@reduce(.Min, a.data[1]) == a.data[1][1]);
}
