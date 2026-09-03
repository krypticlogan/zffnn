const std = @import("std");
const Tensor = @import("tensor.zig");
const Matmul = @import("matmul.zig");
const kernels = @import("kernels.zig");
const validation = @import("validation.zig");

/// Tensor operation classified by whether it computes new storage or creates
/// another view of existing storage.
pub const Op = union(enum) {
    compute: Compute,
    view: View,

    pub const Kind = enum { compute, view };

    pub const Compute = union(enum) {
        relu,
        exp,
        add,
        sub,
        matmul: MatmulAttrs,
        sum: SumAttrs,
        softmax: SoftmaxAttrs,

        pub const SumAttrs = struct { axis: i8 };
        pub const SoftmaxAttrs = struct { axis: i8 };
        pub const MatmulAttrs = Matmul.Plan;

        pub fn execute(
            comptime op: Compute,
            inputs: anytype,
            output: anytype,
        ) void {
            kernels.execute(op, inputs, output);
        }

        pub fn inferRank(op: Compute, inputs: anytype) usize {
            return switch (op) {
                .relu => inferUnaryRank("relu", inputs),
                .exp => inferFloatUnaryRank("exp", inputs),
                .add => inferAddRank(inputs),
                .sub => inferBinaryElementwiseRank("sub", inputs),
                .matmul => inferMatmulRank(inputs),
                .sum => |attrs| inferReductionRank("sum", inputs, attrs.axis),
                .softmax => |attrs| blk: {
                    const rank = inferFloatUnaryRank("softmax", inputs);
                    validation.requireAxis("softmax", inputs[0], attrs.axis);
                    break :blk rank;
                },
            };
        }

        pub fn inferShape(
            comptime op: Compute,
            comptime inputs: anytype,
            comptime max_rank: usize,
        ) Tensor.Shape(max_rank) {
            return switch (op) {
                .relu => inferUnaryShape("relu", inputs, max_rank),
                .exp => blk: {
                    _ = inferFloatUnaryRank("exp", inputs);
                    break :blk inferUnaryShape("exp", inputs, max_rank);
                },
                .add => inferAddShape(inputs, max_rank),
                .sub => inferBinaryElementwiseShape("sub", inputs, max_rank),
                .matmul => inferMatmulShape(inputs, max_rank),
                .sum => |attrs| inferReductionShape(
                    "sum",
                    inputs,
                    attrs.axis,
                    max_rank,
                ),
                .softmax => |attrs| blk: {
                    _ = inferFloatUnaryRank("softmax", inputs);
                    const shape = inferUnaryShape("softmax", inputs, max_rank);
                    validation.requireAxis("softmax", inputs[0], attrs.axis);
                    break :blk shape;
                },
            };
        }
    };

    pub const View = union(enum) {
        transpose: TransposeAttrs,

        pub const TransposeAttrs = struct { axis_a: i8, axis_b: i8 };
    };

    pub fn kind(op: Op) Kind {
        return switch (op) {
            .compute => .compute,
            .view => .view,
        };
    }

    pub fn execute(comptime op: Op, inputs: anytype, output: anytype) void {
        switch (op) {
            .compute => |compute| compute.execute(inputs, output),
            .view => @compileError("view operations do not execute a runtime kernel"),
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

fn inferFloatUnaryRank(comptime operation: []const u8, inputs: anytype) usize {
    const rank = inferUnaryRank(operation, inputs);
    validation.requireDtypeKind(operation, inputs[0], .float);
    return rank;
}

fn inferReductionRank(
    comptime operation: []const u8,
    inputs: anytype,
    comptime axis: i8,
) usize {
    const rank = inferUnaryRank(operation, inputs);
    validation.requireAxis(operation, inputs[0], axis);
    return rank - 1;
}

fn inferReductionShape(
    comptime operation: []const u8,
    comptime inputs: anytype,
    comptime axis: i8,
    comptime max_rank: usize,
) Tensor.Shape(max_rank) {
    _ = inferReductionRank(operation, inputs, axis);
    const removed_axis: usize = @intCast(axis);
    var shape = inputs[0].shape;
    var current_axis = removed_axis;
    while (current_axis + 1 < shape.rank) : (current_axis += 1) {
        shape.dims[current_axis] = shape.dims[current_axis + 1];
    }
    shape.rank -= 1;
    shape.dims[shape.rank] = 0;
    return shape;
}

fn inferBinaryElementwiseRank(
    comptime operation: []const u8,
    inputs: anytype,
) usize {
    validation.requireInputCount(operation, inputs, 2);
    validation.requireMatchingDtypes(operation, inputs);
    return @max(validation.rankOf(inputs[0]), validation.rankOf(inputs[1]));
}

fn inferBinaryElementwiseShape(
    comptime operation: []const u8,
    comptime inputs: anytype,
    comptime max_rank: usize,
) Tensor.Shape(max_rank) {
    const result_rank = inferBinaryElementwiseRank(operation, inputs);
    const lhs_shape = inputs[0].shape.slice();
    const rhs_shape = inputs[1].shape.slice();
    var result = Tensor.Shape(max_rank){
        .rank = result_rank,
        .dims = @splat(0),
    };

    for (0..result_rank) |axis_from_end| {
        const lhs_extent = if (axis_from_end < lhs_shape.len)
            lhs_shape[lhs_shape.len - 1 - axis_from_end]
        else
            1;
        const rhs_extent = if (axis_from_end < rhs_shape.len)
            rhs_shape[rhs_shape.len - 1 - axis_from_end]
        else
            1;

        if (!validation.extentsBroadcast(lhs_extent, rhs_extent)) {
            @compileError(std.fmt.comptimePrint(
                "{s} cannot broadcast extents {d} and {d} at aligned axis {d}",
                .{ operation, lhs_extent, rhs_extent, result_rank - 1 - axis_from_end },
            ));
        }

        result.dims[result_rank - 1 - axis_from_end] =
            if (lhs_extent == 1) rhs_extent else lhs_extent;
    }
    return result;
}

fn inferAddRank(inputs: anytype) usize {
    return inferBinaryElementwiseRank("add", inputs);
}

fn inferAddShape(
    comptime inputs: anytype,
    comptime max_rank: usize,
) Tensor.Shape(max_rank) {
    return inferBinaryElementwiseShape("add", inputs, max_rank);
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
