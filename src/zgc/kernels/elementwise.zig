const std = @import("std");
const Dtype = @import("../storage.zig").Dtype;

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
