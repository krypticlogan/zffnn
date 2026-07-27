const std = @import("std");
const Dtype = @import("dtype.zig").Dtype;
const NodeId = @import("builder.zig").Node.Id;

pub const Id = usize;
pub const Shape = struct {
    pub const max_rank = 8;

    rank: u8,
    dims: [max_rank]usize,

    pub fn init(extents: []const usize) Shape {
        if (extents.len > max_rank) {
            @panic("tensor rank exceeds Shape.max_rank");
        }

        var shape = Shape{
            .rank = @intCast(extents.len),
            .dims = @splat(0),
        };
        @memcpy(shape.dims[0..extents.len], extents);
        return shape;
    }

    pub fn slice(shape: *const Shape) []const usize {
        return shape.dims[0..shape.rank];
    }

    pub fn at(shape: Shape, axis: usize) usize {
        return shape.dims[axis];
    }
};

fn elementCount(comptime shape: []const usize) usize {
    var count: usize = 1;
    for (shape) |extent| count *= extent;
    return count;
}
pub fn Tensor(comptime dtype: Dtype, comptime t_shape: []const usize) type {
    const T = dtype.Scalar();
    const len = elementCount(t_shape);
    return struct {
        pub const scalar_type = T;
        pub const shape = t_shape;
        pub const element_count = len;
        data: [len]T,
    };
}

pub const Info = struct {
    dtype: Dtype,
    shape: Shape,
    producer: ?NodeId = null,

    pub fn debugPrint(info: Info, id: Id) void {
        std.debug.print("  t{d}: {s} shape=", .{ id, @tagName(info.dtype) });
        debugPrintShape(&info.shape);

        if (info.producer) |producer| {
            std.debug.print(" producer=n{d}\n", .{producer});
        } else {
            std.debug.print(" source\n", .{});
        }
    }
};

pub const Ref = struct { id: Id };

pub fn debugPrintShape(shape: *const Shape) void {
    std.debug.print("[", .{});
    for (shape.slice(), 0..) |extent, axis| {
        if (axis != 0) std.debug.print(", ", .{});
        std.debug.print("{d}", .{extent});
    }
    std.debug.print("]", .{});
}

pub fn View(T: type, rank: usize) type {
    return struct {
        ptr: [*]T,
        shape: [rank]usize, // length per tensor dimension; indices correlate directly with dimension.
        stride: [rank]usize, // stride per tensor dimension; indices correlate directly with dimension.
        offset: usize,
    };
}
