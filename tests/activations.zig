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
    const relu_a = zffnn.Activation.apply(.relu, a, false);
    try expect(mat_equal(relu_a, expected));
}

test "sigmoid" {
    var a = zffnn.Mat(2, 3).create(0);
    a.load([_][3]f32{
        .{ -1, 2, -3 },
        .{ 4, -5, 6 },
    });

    const sigmoid_a = zffnn.Activation.apply(.sigmoid, a, false);
    try expect(@reduce(.Min, sigmoid_a.data[0]) == sigmoid_a.data[0][2]);
    try expect(@reduce(.Max, sigmoid_a.data[1]) == sigmoid_a.data[1][2]);

    // numerical stability
    try expect(sigmoid_a.data[0][0] == 0.2689414213699951);
    try expect(sigmoid_a.data[1][0] == 0.9820137900379085);
}

test "softmax" {
    var a = zffnn.Mat(2, 3).create(0);
    a.load([_][3]f32{
        .{ -1, 2, -3 },
        .{ 4, -5, 6 },
    });

    const softmax_a = zffnn.Activation.apply(.softmax, a, false);
    for (softmax_a.data) |row| { // each row should sum to 1
        const sum = @reduce(.Add, row);
        try expect(1 - sum < 0.01); // accounting for floating point precision
    }

    try expect(@reduce(.Max, softmax_a.data[0]) == softmax_a.data[0][1]);
    try expect(@reduce(.Min, softmax_a.data[1]) == softmax_a.data[1][1]);
}
