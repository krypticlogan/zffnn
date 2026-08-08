const std = @import("std");
const Op = @import("../op.zig").Op;
const Tensor = @import("../tensor.zig");
const validation = @import("../validation.zig");

pub fn Result(comptime max_rank: usize) type {
    return struct {
        shape: Tensor.Shape(max_rank),
        layout: Tensor.Layout(max_rank),
        storage_tensor: Tensor.Id,
    };
}

pub fn inferRank(comptime op: Op.View, inputs: anytype) usize {
    return switch (op) {
        .transpose => blk: {
            validation.requireInputCount("transpose", inputs, 1);
            break :blk validation.rankOf(inputs[0]);
        },
    };
}

/// Infer the graph metadata produced by a view operation. This switch is
/// intentionally exhaustive so every new `Op.View` requires an implementation.
pub fn infer(
    comptime op: Op.View,
    comptime inputs: anytype,
    comptime max_rank: usize,
) Result(max_rank) {
    return switch (op) {
        .transpose => |attrs| transpose(inputs, attrs, max_rank),
    };
}

fn transpose(
    comptime inputs: anytype,
    comptime attrs: Op.View.TransposeAttrs,
    comptime max_rank: usize,
) Result(max_rank) {
    validation.requireInputCount("transpose", inputs, 1);
    validation.requireAxis("transpose", inputs[0], attrs.axis_a);
    validation.requireAxis("transpose", inputs[0], attrs.axis_b);

    const axis_a: usize = @intCast(attrs.axis_a);
    const axis_b: usize = @intCast(attrs.axis_b);
    var shape = inputs[0].shape;
    var tensor_layout = inputs[0].layout;

    std.mem.swap(usize, &shape.dims[axis_a], &shape.dims[axis_b]);
    std.mem.swap(
        isize,
        &tensor_layout.strides[axis_a],
        &tensor_layout.strides[axis_b],
    );

    return .{
        .shape = shape,
        .layout = tensor_layout,
        .storage_tensor = inputs[0].storage_tensor,
    };
}
