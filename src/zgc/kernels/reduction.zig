const std = @import("std");
const accumulation = @import("accumulation.zig");
const line = @import("line.zig");

pub fn sum(input: anytype, output: anytype, comptime reduction_axis: i8) void {
    const Input = @TypeOf(input);
    const axis: usize = @intCast(reduction_axis);

    const dtype = Input.dtype;
    for (0..output.len()) |slice_index| {
        const input_line = input.axisSlice(axis, slice_index);
        const result = line.sum(input_line);
        const output_offset = output.elementOffsetFromLinear(slice_index);
        output.storage[output_offset] = accumulation.narrowScalar(dtype, result);
    }
}
