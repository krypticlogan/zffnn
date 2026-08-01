const std = @import("std");
const Dtype = @import("storage.zig").Dtype;
const Node = @import("builder.zig").Node;

pub const Id = usize;
pub const Shape_T = []const usize;

/// Owned shape metadata with a graph-configurable maximum rank.
pub fn Shape(comptime max_rank: usize) type {
    return struct {
        pub const rank_capacity = max_rank;
        const Self = @This();

        rank: usize,
        dims: [max_rank]usize,

        pub fn init(extents: Shape_T) Self {
            if (extents.len > max_rank) {
                @panic("shape rank exceeds graph capacity");
            }

            var shape = Self{
                .rank = extents.len,
                .dims = @splat(0),
            };
            @memcpy(shape.dims[0..extents.len], extents);
            return shape;
        }

        pub fn elementCount(shape: *const Self) usize {
            var count: usize = 1;
            for (shape.slice()) |extent| count *= extent;
            return count;
        }

        pub fn slice(shape: *const Self) []const usize {
            return shape.dims[0..shape.rank];
        }

        pub fn at(shape: Self, axis: usize) usize {
            return shape.dims[axis];
        }
    };
}

fn elementCount(comptime shape: Shape_T) usize {
    var count: usize = 1;
    for (shape) |extent| count *= extent;
    return count;
}

pub fn Tensor(comptime dtype: Dtype, comptime tensor_shape: Shape_T) type {
    const T = dtype.Scalar();
    const len = elementCount(tensor_shape);

    return struct {
        pub const shape = tensor_shape;
        pub const scalar_type = T;
        pub const element_count = len;

        data: [len]T,
    };
}

pub const Source = struct {
    kind: Kind,
    tensor: Id,
    // binding: Binding,

    pub const Kind = enum { input, parameter, constant, state };

    pub const Binding = enum {
        embed,
    };
};

/// Originators (producers) of tensors.
pub const Origin = union(enum) {
    source: Id,
    node: Node.Id,
};

/// Tensor metadata specialized for one graph's maximum rank.
pub fn Info(comptime max_rank: usize) type {
    return struct {
        dtype: Dtype,
        shape: Shape(max_rank),
        origin: Origin,

        pub fn debugPrint(info: @This(), id: Id) void {
            std.debug.print("  t{d}: {s} shape=", .{ id, @tagName(info.dtype) });
            debugPrintShape(&info.shape);
            switch (info.origin) {
                .node => |node| std.debug.print(" producer=n{d}\n", .{node}),
                .source => |source| std.debug.print(
                    " source={d}\n",
                    .{source},
                ),
            }
        }
    };
}

pub const Ref = struct { id: Id };

pub fn debugPrintShape(shape: anytype) void {
    std.debug.print("[", .{});
    for (shape.slice(), 0..) |extent, axis| {
        if (axis != 0) std.debug.print(", ", .{});
        std.debug.print("{d}", .{extent});
    }
    std.debug.print("]", .{});
}

/// View into memory for a given tensor.
pub fn View(comptime T: type, comptime tensor_rank: usize) type {
    return struct {
        pub const scalar_type = T;
        pub const dtype = Dtype.fromScalar(T);
        pub const rank = tensor_rank;

        data: []T,
        shape: [tensor_rank]usize,
        // stride: [tensor_rank]usize,
        // offset: usize,
    };
}

/// Read-only view into memory for a given tensor.
pub fn ConstView(comptime T: type, comptime tensor_rank: usize) type {
    return struct {
        pub const scalar_type = T;
        pub const dtype = Dtype.fromScalar(T);
        pub const rank = tensor_rank;

        data: []const T,
        shape: [tensor_rank]usize,
    };
}
