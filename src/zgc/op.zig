const std = @import("std");
const Tensor = @import("tensor.zig");
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
        matmul,
        softmax: SoftmaxAttrs,

        pub const SoftmaxAttrs = struct { axis: i8 };

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
                .exp => inferUnaryRank("exp", inputs),
                .add => inferAddRank(inputs),
                .sub => inferBinaryElementwiseRank("sub", inputs),
                .matmul => inferMatmulRank(inputs),
                .softmax => inferUnaryRank("softmax", inputs),
            };
        }

        pub fn inferShape(
            comptime op: Compute,
            comptime inputs: anytype,
            comptime max_rank: usize,
        ) Tensor.Shape(max_rank) {
            return switch (op) {
                .relu => inferUnaryShape("relu", inputs, max_rank),
                .exp => inferUnaryShape("exp", inputs, max_rank),
                .add => inferAddShape(inputs, max_rank),
                .sub => inferBinaryElementwiseShape("sub", inputs, max_rank),
                .matmul => inferMatmulShape(inputs, max_rank),
                .softmax => |attrs| blk: {
                    const shape = inferUnaryShape("softmax", inputs, max_rank);
                    validation.requireAxis("softmax", inputs[0], attrs.axis);
                    break :blk shape;
                },
            };
        }

        pub fn debugPrint(op: Compute) void {
            switch (op) {
                .relu => std.debug.print("relu", .{}),
                .exp => std.debug.print("exp", .{}),
                .add => std.debug.print("add", .{}),
                .sub => std.debug.print("sub", .{}),
                .matmul => std.debug.print("matmul", .{}),
                .softmax => |attrs| {
                    std.debug.print("softmax(axis={d})", .{attrs.axis});
                },
            }
        }
    };

    pub const View = union(enum) {
        transpose: TransposeAttrs,

        pub const TransposeAttrs = struct { axis_a: i8, axis_b: i8 };

        pub fn debugPrint(op: View) void {
            switch (op) {
                .transpose => |attrs| {
                    std.debug.print(
                        "transpose(axes={d},{d})",
                        .{ attrs.axis_a, attrs.axis_b },
                    );
                },
            }
        }
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

    pub fn debugPrint(op: Op) void {
        switch (op) {
            .compute => |compute| compute.debugPrint(),
            .view => |view| view.debugPrint(),
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

fn inferBinaryElementwiseRank(
    comptime operation: []const u8,
    inputs: anytype,
) usize {
    validation.requireInputCount(operation, inputs, 2);
    validation.requireMatchingRanks(operation, inputs);
    validation.requireMatchingDtypes(operation, inputs);
    return validation.rankOf(inputs[0]);
}

fn inferBinaryElementwiseShape(
    comptime operation: []const u8,
    comptime inputs: anytype,
    comptime max_rank: usize,
) Tensor.Shape(max_rank) {
    _ = inferBinaryElementwiseRank(operation, inputs);
    validation.requireMatchingShapes(operation, inputs[0], inputs[1]);
    return inputs[0].shape;
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
