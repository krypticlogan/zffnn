const std = @import("std");
const line = @import("line.zig");

/// Numerically stable softmax over one logical tensor axis.
pub fn softmax(input: anytype, output: anytype, comptime softmax_axis: i8) void {
    const Input = @TypeOf(input);
    const Output = @TypeOf(output);
    comptime {
        if (softmax_axis < 0 or softmax_axis >= Input.rank) {
            @compileError("softmax axis is outside the input rank");
        }
        if (Input.rank != Output.rank) {
            @compileError("softmax input and output ranks must match");
        }
        if (Input.scalar_type != Output.scalar_type) {
            @compileError("softmax input and output dtypes must match");
        }
        if (Input.dtype.kind() != .float) {
            @compileError("softmax supports only floating-point tensors");
        }
    }

    std.debug.assert(std.mem.eql(usize, &input.shape, &output.shape));
    const axis: usize = @intCast(softmax_axis);
    const extent = input.shape[axis];
    if (extent == 0) return;

    const slice_count = input.len() / extent;

    for (0..slice_count) |slice_index| {
        const input_line = input.axisSlice(axis, slice_index);
        const output_line = output.axisSlice(axis, slice_index);
        const maximum = line.max(input_line);
        const denominator = line.shiftedExpSum(input_line, maximum);
        line.normalizedExp(input_line, output_line, maximum, denominator);
    }
}
