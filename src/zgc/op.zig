const std = @import("std");
const Tensor = @import("tensor.zig");
const kernels = @import("kernels.zig");
const validation = @import("validation.zig");
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
            .relu => inferUnaryRank("relu", inputs),
            .add => inferAddRank(inputs),
            .matmul => inferMatmulRank(inputs),
            .softmax => inferUnaryRank("softmax", inputs),
            .transpose => inferUnaryRank("transpose", inputs),
        };
    }

    pub fn inferShape(
        comptime op: Op,
        comptime inputs: anytype,
        comptime max_rank: usize,
    ) Tensor.Shape(max_rank) {
        return switch (op) {
            .relu => inferUnaryShape("relu", inputs, max_rank),
            .add => inferAddShape(inputs, max_rank),
            .matmul => inferMatmulShape(inputs, max_rank),
            .softmax => |attrs| blk: {
                const shape = inferUnaryShape("softmax", inputs, max_rank);
                validation.requireAxis("softmax", inputs[0], attrs.axis);
                break :blk shape;
            },
            .transpose => |attrs| blk: {
                var shape = inferUnaryShape("transpose", inputs, max_rank);
                validation.requireAxis("transpose", inputs[0], attrs.axis_a);
                validation.requireAxis("transpose", inputs[0], attrs.axis_b);
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

fn inferUnaryRank(comptime operation: []const u8, inputs: anytype) usize {
    validation.requireInputCount(operation, inputs, 1);
    return validation.rankOf(inputs[0]);
}

fn inferUnaryShape(
    comptime operation: []const u8,
    comptime inputs: anytype,
    comptime max_rank: usize,
) Tensor.Shape(max_rank) {
    validation.requireInputCount(operation, inputs, 1);
    return inputs[0].shape;
}

fn inferAddRank(inputs: anytype) usize {
    validation.requireInputCount("add", inputs, 2);
    validation.requireMatchingRanks("add", inputs);
    validation.requireMatchingDtypes("add", inputs);
    return validation.rankOf(inputs[0]);
}

fn inferAddShape(
    comptime inputs: anytype,
    comptime max_rank: usize,
) Tensor.Shape(max_rank) {
    _ = inferAddRank(inputs);
    validation.requireMatchingShapes("add", inputs[0], inputs[1]);
    return inputs[0].shape;
}

fn inferMatmulRank(inputs: anytype) usize {
    validation.requireInputCount("matmul", inputs, 2);
    validation.requireRanks("matmul", inputs, &.{ 2, 2 });
    validation.requireMatchingDtypes("matmul", inputs);
    validation.requireDtype("matmul", inputs[0], .f32);
    return 2;
}

fn inferMatmulShape(
    comptime inputs: anytype,
    comptime max_rank: usize,
) Tensor.Shape(max_rank) {
    _ = inferMatmulRank(inputs);
    validation.requireMatchingExtents("matmul", inputs[0], 1, inputs[1], 0);

    return Tensor.Shape(max_rank).init(&.{
        inputs[0].shape.at(0),
        inputs[1].shape.at(1),
    });
}
