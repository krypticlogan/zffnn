const std = @import("std");
const print = std.debug.print;
const testing = std.testing;
const expect = testing.expect;
const expectEqual = testing.expectEqual;


const Mat = @import("matrix.zig").Mat;
const NN = @import("network.zig").NN;

pub fn is_matrix(comptime maybe_mat_type: anytype) bool {
    // @compileLog(maybe_mat_type);
    const fields = std.meta.fields(maybe_mat_type);
    // todo : gate on comptime n and m too
    return std.mem.eql(u8, fields[0].name, "data");
}
test "test_isMatrix" {
    var test_mat = Mat(2, 4).create(0);
    const test_mat_ptr = &test_mat;
    try std.testing.expect(is_matrix(Mat(3, 4)) == true);
    try std.testing.expect(is_matrix(@TypeOf(test_mat)) == true);
    try std.testing.expect(is_matrix(@TypeOf(test_mat_ptr.*)) == true);
    try std.testing.expect(is_matrix(struct { not_data: f32 }) == false);
}


pub fn elemwise_is_defined(comptime a_type: anytype, comptime b_type: anytype) union(enum) { none, full, per_row } {
    if (a_type.n == b_type.n and a_type.m == b_type.m) return .full;
    if (a_type.n == b_type.n and b_type.m == 1) return .per_row;
    return .none;
}
test "elemwiseIsDefined" {
    try std.testing.expect(elemwise_is_defined(Mat(2, 4), Mat(2, 4)) == .full);
    try std.testing.expect(elemwise_is_defined(Mat(15, 25), Mat(15, 1)) == .per_row);
    try std.testing.expect(elemwise_is_defined(Mat(15, 25), Mat(15, 2)) == .none);
    try std.testing.expect(elemwise_is_defined(Mat(1, 2), Mat(3, 4)) == .none);
}

pub fn mul_is_defined(comptime a_type: anytype, comptime b_type: anytype) bool {
    return a_type.m == b_type.n;
}
test "mulIsDefined" {
    try std.testing.expect(mul_is_defined(Mat(4, 2), Mat(2, 5)) == true);
    try std.testing.expect(mul_is_defined(Mat(1, 2), Mat(3, 4)) == false);
}