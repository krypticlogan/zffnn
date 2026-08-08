const std = @import("std");
const accumulation = @import("accumulation.zig");
const line = @import("line.zig");

pub fn sum(input: anytype, output: anytype, comptime reduction_axis: i8) void {
    const Input = @TypeOf(input);
    const Output = @TypeOf(output);
    comptime {
        if (reduction_axis < 0 or reduction_axis >= Input.rank) {
            @compileError("sum axis is outside the input rank");
        }
        if (Output.rank + 1 != Input.rank) {
            @compileError("sum output rank must be one less than its input rank");
        }
        if (Input.scalar_type != Output.scalar_type) {
            @compileError("sum input and output dtypes must match");
        }
    }

    const axis: usize = @intCast(reduction_axis);
    assertReducedShape(input, output, axis);

    const dtype = Input.dtype;
    for (0..output.len()) |slice_index| {
        const input_line = input.axisSlice(axis, slice_index);
        const result = line.sum(input_line);
        const output_offset = output.elementOffsetFromLinear(slice_index);
        output.storage[output_offset] = accumulation.narrowScalar(dtype, result);
    }
}

fn assertReducedShape(input: anytype, output: anytype, axis: usize) void {
    if (comptime @TypeOf(output).rank == 0) return;

    var output_axis: usize = 0;
    for (input.shape, 0..) |extent, input_axis| {
        if (input_axis == axis) continue;
        std.debug.assert(output.shape[output_axis] == extent);
        output_axis += 1;
    }
}
