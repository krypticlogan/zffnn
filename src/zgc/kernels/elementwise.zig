const std = @import("std");
const Dtype = @import("../dtype.zig").Dtype;

/// Apply an element-wise operator through a contiguous SIMD fast path or a generic strided
/// traversal when any participating view is non-contiguous.
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

    std.debug.assert(std.mem.eql(usize, &input.shape, &output.shape));

    const dtype = Input.dtype;
    if (input.contiguousSlice()) |input_storage| {
        if (output.contiguousSlice()) |output_storage| {
            std.debug.assert(input_storage.len == output_storage.len);
            const vector_len = std.simd.suggestVectorLength(Input.scalar_type) orelse 1;
            const Vector = dtype.Vector(vector_len);

            var index: usize = 0;
            while (index + vector_len <= input_storage.len) : (index += vector_len) {
                const values: Vector = input_storage[index..][0..vector_len].*;
                output_storage[index..][0..vector_len].* = Operator.vector(dtype, vector_len, values);
            }
            while (index < input_storage.len) : (index += 1) {
                output_storage[index] = Operator.scalar(dtype, input_storage[index]);
            }
            return;
        }
    }

    for (0..input.len()) |linear_index| {
        const input_index = input.elementOffsetFromLinear(linear_index);
        const output_index = output.elementOffsetFromLinear(linear_index);
        output.storage[output_index] = Operator.scalar(
            dtype,
            input.storage[input_index],
        );
    }
}

/// Apply a element-wise binary operator through a contiguous SIMD fast path or a generic
/// strided traversal.
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
    }

    const a_view = a.broadcastTo(Output.rank, output.shape);
    const b_view = b.broadcastTo(Output.rank, output.shape);

    const dtype = Output.dtype;
    if (a_view.contiguousSlice()) |a_storage| {
        if (b_view.contiguousSlice()) |b_storage| {
            if (output.contiguousSlice()) |output_storage| {
                const vector_len = std.simd.suggestVectorLength(Output.scalar_type) orelse 1;
                const Vector = dtype.Vector(vector_len);

                var index: usize = 0;
                while (index + vector_len <= a_storage.len) : (index += vector_len) {
                    const a_values: Vector = a_storage[index..][0..vector_len].*;
                    const b_values: Vector = b_storage[index..][0..vector_len].*;
                    output_storage[index..][0..vector_len].* = Operator.vector(dtype, vector_len, a_values, b_values);
                }
                while (index < a_storage.len) : (index += 1) {
                    output_storage[index] = Operator.scalar(dtype, a_storage[index], b_storage[index]);
                }
                return;
            }
        }
    }

    for (0..output.len()) |linear_index| {
        const a_index = a_view.elementOffsetFromLinear(linear_index);
        const b_index = b_view.elementOffsetFromLinear(linear_index);
        const output_index = output.elementOffsetFromLinear(linear_index);
        output.storage[output_index] = Operator.scalar(
            dtype,
            a_view.storage[a_index],
            b_view.storage[b_index],
        );
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

pub fn exp(input: anytype, output: anytype) void {
    comptime {
        if (@TypeOf(input).dtype.kind() != .float) {
            @compileError("exp supports only floating-point tensors");
        }
    }

    unary(input, output, struct {
        fn scalar(comptime dtype: Dtype, value: dtype.Scalar()) dtype.Scalar() {
            return @exp(value);
        }

        fn vector(
            comptime dtype: Dtype,
            comptime len: usize,
            values: dtype.Vector(len),
        ) dtype.Vector(len) {
            return @exp(values);
        }
    });
}

pub fn add(a: anytype, b: anytype, output: anytype) void {
    binary(a, b, output, struct {
        fn scalar(comptime dtype: Dtype, a_value: dtype.Scalar(), b_value: dtype.Scalar()) dtype.Scalar() {
            return a_value + b_value;
        }

        fn vector(comptime dtype: Dtype, comptime len: usize, a_vec: dtype.Vector(len), b_vec: dtype.Vector(len)) dtype.Vector(len) {
            return a_vec + b_vec;
        }
    });
}

pub fn sub(a: anytype, b: anytype, output: anytype) void {
    binary(a, b, output, struct {
        fn scalar(comptime dtype: Dtype, a_value: dtype.Scalar(), b_value: dtype.Scalar()) dtype.Scalar() {
            return a_value - b_value;
        }

        fn vector(comptime dtype: Dtype, comptime len: usize, a_vec: dtype.Vector(len), b_vec: dtype.Vector(len)) dtype.Vector(len) {
            return a_vec - b_vec;
        }
    });
}
