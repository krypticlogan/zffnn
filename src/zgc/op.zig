const std = @import("std");
const Tensor = @import("tensor.zig");
const kernels = @import("kernels.zig");
/// Tensor Operations
pub const Op = union(enum) {
    relu,
    add,
    matmul,
    softmax: SoftmaxAttrs,
    transpose: TransposeAttrs,

    pub const SoftmaxAttrs = struct { axis: i8 };
    pub const TransposeAttrs = struct { axis_a: i8, axis_b: i8 };

    /// Execute a tensor operation
    /// `inputs` should be a tuple of TensorViews
    /// Invalid inputs/state will not compile
    pub fn execute(comptime op: Op, inputs: anytype, output: anytype) void {
        kernels.execute(op, inputs, output);
    }

    pub fn inferRank(op: Op, inputs: anytype) usize {
        return switch (op) {
            .relu, .softmax, .transpose => inputs[0].rank,
            .add => @max(inputs[0].rank, inputs[1].rank),
            .matmul => 2,
        };
    }

    pub fn inferShape(
        comptime op: Op,
        comptime inputs: anytype,
        comptime max_rank: usize,
    ) Tensor.Shape(max_rank) {
        return switch (op) {
            .relu, .softmax => inputs[0].shape,
            .add => if (inputs[0].shape.rank >= inputs[1].shape.rank)
                inputs[0].shape
            else
                inputs[1].shape,
            .matmul => Tensor.Shape(max_rank).init(&.{
                inputs[0].shape.at(0),
                inputs[1].shape.at(1),
            }),
            .transpose => |attrs| blk: {
                var shape = inputs[0].shape;
                const axis_a: usize = @intCast(attrs.axis_a);
                const axis_b: usize = @intCast(attrs.axis_b);
                std.mem.swap(
                    usize,
                    &shape.dims[axis_a],
                    &shape.dims[axis_b],
                );
                break :blk shape;
            },
        };
    }

    pub fn debugPrint(op: Op) void {
        switch (op) {
            .relu => std.debug.print("relu", .{}),
            .add => std.debug.print("add", .{}),
            .matmul => std.debug.print("matmul", .{}),
            .softmax => |attrs| {
                std.debug.print("softmax(axis={d})", .{attrs.axis});
            },
            .transpose => |attrs| {
                std.debug.print(
                    "transpose(axes={d},{d})",
                    .{ attrs.axis_a, attrs.axis_b },
                );
            },
        }
    }
};
