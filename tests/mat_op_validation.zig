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

    const c = a.mul(&b, false); // result is a 2x2 matrix
    try expect(c.data[0][0] == 58);
    try expect(c.data[0][1] == 64);
    try expect(c.data[1][0] == 139);
    try expect(c.data[1][1] == 154);
}

test "mat mul batched" {
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

    const c = a.mul(&b, true); // result is a 2x2 matrix
    try expect(c.data[0][0] == 58);
    try expect(c.data[0][1] == 64);
    try expect(c.data[1][0] == 139);
    try expect(c.data[1][1] == 154);
}

test "in-place single mat mul matches returning mat mul" {
    var a = zffnn.Mat(2, 3).create(0);
    a.load([_][3]f32{ .{ 1, 2, 3 }, .{ 4, 5, 6 } });
    var b = zffnn.Mat(3, 2).create(0);
    b.load([_][2]f32{
        .{ 7, 8 },
        .{ 9, 10 },
        .{ 11, 12 },
    });
    const expected = a.mul(&b, false);

    var actual = zffnn.Mat(2, 2).create(std.math.nan(f32));
    a.mul_(&b, &actual, false);

    try testing.expectEqualDeep(expected.data, actual.data);
}

test "in-place batched mat mul matches returning mat mul" {
    var a = zffnn.Mat(2, 3).create(0);
    a.load([_][3]f32{ .{ 1, 2, 3 }, .{ 4, 5, 6 } });
    var b = zffnn.Mat(3, 2).create(0);
    b.load([_][2]f32{
        .{ 7, 8 },
        .{ 9, 10 },
        .{ 11, 12 },
    });
    const expected = a.mul(&b, true);

    var actual = zffnn.Mat(2, 2).create(std.math.nan(f32));
    a.mul_(&b, &actual, true);

    try testing.expectEqualDeep(expected.data, actual.data);
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

test "in-place elementwise operations match returning operations" {
    var a = zffnn.Mat(2, 3).create(0);
    a.load(.{
        .{ -4, -3, -2 },
        .{ -2, -1, 0 },
    });
    var full = zffnn.Mat(2, 3).create(0);
    full.load(.{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    var broadcast = zffnn.Mat(2, 1).create(0);
    broadcast.load(.{
        .{5},
        .{6},
    });

    var actual = a.dupe_like(.clone);
    const expected_full_add = a.add(full);
    actual.add_(full);
    try testing.expectEqualDeep(expected_full_add.data, actual.data);

    actual = a.dupe_like(.clone);
    const expected_broadcast_add = a.add(broadcast);
    actual.add_(broadcast);
    try testing.expectEqualDeep(expected_broadcast_add.data, actual.data);

    actual = a.dupe_like(.clone);
    const expected_full_sub = a.sub(full);
    actual.sub_(full);
    try testing.expectEqualDeep(expected_full_sub.data, actual.data);

    actual = a.dupe_like(.clone);
    const expected_broadcast_sub = a.sub(broadcast);
    actual.sub_(broadcast);
    try testing.expectEqualDeep(expected_broadcast_sub.data, actual.data);
}

test "row and column reductions handle negative values" {
    var a = zffnn.Mat(2, 3).create(0);
    a.load(.{
        .{ -4, -8, -2 },
        .{ -5, -3, -9 },
    });

    const row_max: [2]f32 = a.max_rwise();
    try testing.expectEqualSlices(f32, &.{ -2, -3 }, &row_max);

    const col_max: [3]f32 = a.max_cwise();
    try testing.expectEqualSlices(f32, &.{ -4, -3, -2 }, &col_max);

    const row_sum: [2]f32 = a.sum_rwise();
    try testing.expectEqualSlices(f32, &.{ -14, -17 }, &row_sum);

    const col_sum: [3]f32 = a.sum_cwise();
    try testing.expectEqualSlices(f32, &.{ -9, -11, -11 }, &col_sum);
}

test "runtime scalar access reads and writes vector-backed rows" {
    var a = zffnn.Mat(2, 3).create(0);

    var row: usize = 0;
    var col: usize = 1;
    row += 1;
    col += 1;

    a.set(row, col, 42);
    try testing.expectEqual(@as(f32, 42), a.get(row, col));
}
