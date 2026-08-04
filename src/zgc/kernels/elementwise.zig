const std = @import("std");
const Dtype = @import("../dtype.zig").Dtype;

/// Apply an operator to a contiguous tensor one native SIMD-width chunk at a
/// time, then finish any remainder with the same operator's scalar form.
fn unary(input: anytype, output: anytype, comptime Operator: type) void {
    const Input = @TypeOf(input);
    const Output = @TypeOf(output);

    comptime {
        if (Input.scalar_type != Output.scalar_type) {
            @compileError("elementwise input and output dtypes must match");
        }
        if (Input.rank != Output.rank) {
            @compileError("elementwise input and output ranks must match");
        }
    }

    std.debug.assert(input.data.len == output.data.len);
    std.debug.assert(std.mem.eql(usize, &input.shape, &output.shape));

    const dtype = Input.dtype;
    const vector_len = std.simd.suggestVectorLength(Input.scalar_type) orelse 1;
    const Vector = dtype.Vector(vector_len);

    var index: usize = 0;
    while (index + vector_len <= input.data.len) : (index += vector_len) {
        const values: Vector = input.data[index..][0..vector_len].*;
        output.data[index..][0..vector_len].* = Operator.vector(dtype, vector_len, values);
    }
    while (index < input.data.len) : (index += 1) {
        output.data[index] = Operator.scalar(dtype, input.data[index]);
    }
}

/// Apply an operator on two contiguous tensors one native SIMD-width chunk at a
/// time, then finish any remainder with the same operator's scalar form.
fn binary(a: anytype, b: anytype, output: anytype, comptime Operator: type) void {
    const A = @TypeOf(a);
    const B = @TypeOf(b);
    const Output = @TypeOf(output);

    comptime {
        if (A.scalar_type != B.scalar_type) {
            @compileError("binary elementwise input dtypes must match");
        }
        if (A.scalar_type != Output.scalar_type) {
            @compileError("elementwise input and output dtypes must match");
        }
        if (A.rank != Output.rank) {
            @compileError("elementwise input and output ranks must match");
        }
    }

    std.debug.assert(a.data.len == b.data.len); // extra check for binary operations
    std.debug.assert(a.data.len == output.data.len);
    std.debug.assert(std.mem.eql(usize, &a.shape, &b.shape)); // this may change depending on how broadcasting is handled
    std.debug.assert(std.mem.eql(usize, &a.shape, &output.shape));

    const dtype = Output.dtype;
    const vector_len = std.simd.suggestVectorLength(Output.scalar_type) orelse 1;
    const Vector = dtype.Vector(vector_len);

    var index: usize = 0;
    while (index + vector_len <= a.data.len) : (index += vector_len) {
        const a_values: Vector = a.data[index..][0..vector_len].*;
        const b_values: Vector = b.data[index..][0..vector_len].*;
        output.data[index..][0..vector_len].* = Operator.vector(dtype, vector_len, a_values, b_values);
    }
    while (index < a.data.len) : (index += 1) {
        output.data[index] = Operator.scalar(dtype, a.data[index], b.data[index]);
    }
}

pub fn relu(input: anytype, output: anytype) void {
    unary(input, output, struct {
        fn scalar(comptime dtype: Dtype, value: dtype.Scalar()) dtype.Scalar() {
            return @max(value, dtype.zero());
        }

        fn vector(
            comptime dtype: Dtype,
            comptime len: usize,
            values: dtype.Vector(len),
        ) dtype.Vector(len) {
            return @max(values, dtype.vectorZero(len));
        }
    });
}

pub fn add(a: anytype, b: anytype, output: anytype) void {
    binary(a, b, output, struct {
        fn scalar(
            comptime dtype: Dtype, 
            a_value: dtype.Scalar(), 
            b_value: dtype.Scalar()
        ) dtype.Scalar() {
            return a_value + b_value;
        }

        fn vector(
            comptime dtype: Dtype,
            comptime len: usize,
            a_vec: dtype.Vector(len), 
            b_vec: dtype.Vector(len)
        ) dtype.Vector(len) {
            return a_vec + b_vec;
        }
    });
}
