//! Matrix operations validation tests.
const std = @import("std");
const testing = std.testing;
const expect = testing.expect;
const approxEqAbs = testing.expectApproxEqAbs;

const zffnn = @import("zffnn");

test "mat mul" {
    var a = zffnn.Mat(2, 3).create(0);
    a.load([_][3]f32{ .{ 1, 2, 3 }, .{ 4, 5, 6 } });
    var b = zffnn.Mat(3, 2).create(0);
    b.load([_][2]f32{
        .{ 7, 8 },
        .{ 9, 10 },
        .{ 11, 12 },
    });

    // mat mul like:
    //
    // {1, 2, 3}      {7, 8}        {1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12}
    // {4, 5, 6}    * {9, 10} =     {4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12}
    //                {11, 12}

    const c = a.mul(b, false); // result is a 2x2 matrix
    try expect(c.data[0][0] == 58);
    try expect(c.data[0][1] == 64);
    try expect(c.data[1][0] == 139);
    try expect(c.data[1][1] == 154);
}

test "full add" {
    var a = zffnn.Mat(2, 3).create(0);
    a.load([_][3]f32{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    var b = zffnn.Mat(2, 3).create(0);
    b.load([_][3]f32{
        .{ 7, 8, 9 },
        .{ 10, 11, 12 },
    });

    // mat add like:
    //
    // {1, 2, 3}      {7, 8, 9}      {1+7, 2+8, 3+9}
    // {4, 5, 6}    + {10, 11, 12} = {4+10, 5+11, 6+12}

    const c = a.add(b); // result is a 2x3 matrix
    try expect(c.data[0][0] == 8);
    try expect(c.data[0][1] == 10);
    try expect(c.data[0][2] == 12);
    try expect(c.data[1][0] == 14);
    try expect(c.data[1][1] == 16);
    try expect(c.data[1][2] == 18);
}

test "col-wise add" {
    var a = zffnn.Mat(2, 3).create(0);
    a.load([_][3]f32{
        .{ -4, -3, -2 },
        .{ -2, -1, 0 },
    });
    var b = zffnn.Mat(2, 1).create(0);
    b.load([_][1]f32{
        .{5},
        .{6},
    });

    // row-wise add like:
    //
    // {-4, -3, -2}    + {5}     {-4+5, -3+5, -2+5}
    // {-2, -1,  0}    + {6} =   {-2+6, -1+6,  0+6}

    const c = a.add(b); // result is a 2x3 matrix
    try expect(c.data[0][0] == 1);
    try expect(c.data[0][1] == 2);
    try expect(c.data[0][2] == 3);
    try expect(c.data[1][0] == 4);
    try expect(c.data[1][1] == 5);
    try expect(c.data[1][2] == 6);
}

test "full sub" {
    var a = zffnn.Mat(2, 3).create(0);
    a.load([_][3]f32{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    var b = zffnn.Mat(2, 3).create(0);
    b.load([_][3]f32{
        .{ 7, 8, 9 },
        .{ 10, 11, 12 },
    });

    // mat sub like:
    //
    // {1, 2, 3}      {7, 8, 9}      {1-7, 2-8, 3-9}
    // {4, 5, 6}    - {10, 11, 12} = {4-10, 5-11, 6-12}

    const c = a.sub(b); // result is a 2x3 matrix
    try expect(c.data[0][0] == -6);
    try expect(c.data[0][1] == -6);
    try expect(c.data[0][2] == -6);
    try expect(c.data[1][0] == -6);
    try expect(c.data[1][1] == -6);
    try expect(c.data[1][2] == -6);
}

test "col-wise sub" {
    var a = zffnn.Mat(2, 3).create(0);
    a.load([_][3]f32{
        .{ 4, 3, 2 },
        .{ 2, 1, 0 },
    });
    var b = zffnn.Mat(2, 1).create(0);
    b.load([_][1]f32{
        .{5},
        .{6},
    });

    // row-wise sub like:
    //
    // {4, 3, 2}      {5}      {4-5, 3-5, 2-5}
    // {2, 1, 0}    - {6} =   {2-6, 1-6, 0-6}

    const c = a.sub(b); // result is a 2x3 matrix
    try expect(c.data[0][0] == -1);
    try expect(c.data[0][1] == -2);
    try expect(c.data[0][2] == -3);
    try expect(c.data[1][0] == -4);
    try expect(c.data[1][1] == -5);
    try expect(c.data[1][2] == -6);
}

test "max_rwise" {
    var a = zffnn.Mat(2, 3).create(0);
    a.load([_][3]f32{
        .{ 4, 3, 2 },
        .{ 2, 1, 0 },
    });
    const c = a.max_rwise();
    try expect(c[0] == 4);
    try expect(c[1] == 2);
}

test "max_cwise" {
    var a = zffnn.Mat(2, 3).create(0);
    a.load([_][3]f32{
        .{ 4, 3, 2 },
        .{ 2, 1, 0 },
    });
    const c = a.max_cwise();
    try expect(c[0] == 4);
    try expect(c[1] == 3);
    try expect(c[2] == 2);
}

test "exp" {
    var a = zffnn.Mat(2, 3).create(0);
    a.load([_][3]f32{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    const c = a.exp();

    try approxEqAbs(@exp(@as(f32, 1.0)), c.data[0][0], 1e-5);
    try approxEqAbs(@exp(@as(f32, 2.0)), c.data[0][1], 1e-5);
    try approxEqAbs(@exp(@as(f32, 3.0)), c.data[0][2], 1e-5);
    try approxEqAbs(@exp(@as(f32, 4.0)), c.data[1][0], 1e-5);
    try approxEqAbs(@exp(@as(f32, 5.0)), c.data[1][1], 1e-5);
    try approxEqAbs(@exp(@as(f32, 6.0)), c.data[1][2], 1e-5);
}

test "transpose" {
    var a = zffnn.Mat(2, 3).create(0);
    a.load([_][3]f32{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    const c = a.t();
    try expect(c.data[0][0] == 1);
    try expect(c.data[0][1] == 4);
    try expect(c.data[1][0] == 2);
    try expect(c.data[1][1] == 5);
    try expect(c.data[2][0] == 3);
    try expect(c.data[2][1] == 6);
}
