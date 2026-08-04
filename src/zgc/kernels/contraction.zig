const std = @import("std");
const accumulation = @import("accumulation.zig");

/// Multiply two contiguous row-major rank-2 tensors.
///
/// lhs:    [M, K]
/// rhs:    [K, N]
/// output: [M, N]
///
/// The SIMD lanes span N
pub fn matmul(lhs: anytype, rhs: anytype, output: anytype) void {
    const Lhs = @TypeOf(lhs);
    const Rhs = @TypeOf(rhs);
    const Output = @TypeOf(output);

    comptime {
        if (Lhs.rank != 2 or Rhs.rank != 2 or Output.rank != 2) {
            @compileError("matmul requires rank-2 input and output views");
        }
        if (Lhs.scalar_type != Rhs.scalar_type or
            Lhs.scalar_type != Output.scalar_type)
        {
            @compileError("matmul input and output dtypes must match");
        }
        if (Output.dtype != .f32) {
            @compileError("the row-major matmul kernel currently supports only f32");
        }
    }

    const m = lhs.shape[0];
    const k_len = lhs.shape[1];
    const n = rhs.shape[1];

    std.debug.assert(rhs.shape[0] == k_len); // TODO: this stuff should be enforced at graph-time
    std.debug.assert(output.shape[0] == m);
    std.debug.assert(output.shape[1] == n);
    std.debug.assert(lhs.data.len == m * k_len);
    std.debug.assert(rhs.data.len == k_len * n);
    std.debug.assert(output.data.len == m * n);

    const dtype = Output.dtype;
    const vector_len = std.simd.suggestVectorLength(Output.scalar_type) orelse 1;
    const InputVector = dtype.Vector(vector_len);
    const Accumulator = accumulation.AccumulatorVector(dtype, vector_len);
    const AccumulatorScalar = accumulation.AccumulatorScalar(dtype);

    for (0..m) |row| {
        var col: usize = 0;

        while (col + vector_len <= n) : (col += vector_len) {
            var accumulator: Accumulator = @splat(0);

            for (0..k_len) |k| {
                const lhs_value: Accumulator = @splat(
                    accumulation.widenScalar(dtype, lhs.data[row * k_len + k]),
                );
                const rhs_values: InputVector =
                    rhs.data[k * n + col ..][0..vector_len].*;

                accumulator +=
                    lhs_value * accumulation.widenVector(dtype, vector_len, rhs_values);
            }

            output.data[row * n + col ..][0..vector_len].* = accumulator;
        }

        while (col < n) : (col += 1) {
            var accumulator: AccumulatorScalar = 0;

            for (0..k_len) |k| {
                accumulator +=
                    accumulation.widenScalar(dtype, lhs.data[row * k_len + k]) *
                    accumulation.widenScalar(dtype, rhs.data[k * n + col]);
            }

            output.data[row * n + col] = accumulator;
        }
    }
}
