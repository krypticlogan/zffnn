const std = @import("std");
const Dtype = @import("dtype.zig").Dtype;

pub const Id = usize;
pub const Shape_T = []const usize;
pub fn Tensor(comptime dtype: Dtype, comptime tensor_shape: Shape_T) type {
    const T = dtype.Scalar();
    const len = Shape(tensor_shape.len).init(tensor_shape).elementCount();

    return struct {
        pub const shape = tensor_shape;
        pub const scalar_type = T;
        pub const element_count = len;

        storage: [len]T,
    };
}

/// View into memory for a given tensor.
pub fn View(comptime T: type, comptime tensor_rank: usize) type {
    return struct {
        const Self = @This();
        pub const scalar_type = T;
        pub const dtype = Dtype.fromScalar(T);
        pub const rank = tensor_rank;
        // pub const len = Shape(tensor_rank).init().elementCount();

        storage: []T,
        shape: [tensor_rank]usize,
        strides: [tensor_rank]isize,
        offset: usize,

        pub fn len(self: *const Self) usize {
            return Shape(rank).init(&self.shape).elementCount();
        }
        pub fn elementOffset(
            self: *const Self,
            indices: [rank]usize,
        ) usize {
            var offset: isize = @intCast(self.offset);
            for (self.strides, indices) |stride, index| {
                offset += @as(isize, @intCast(index)) * stride;
            }
            std.debug.assert(offset >= 0);
            return @intCast(offset);
        }

        pub fn get(
            self: *const Self,
            indices: [rank]usize,
        ) T {
            const elem_index = self.elementOffset(indices);
            return self.storage[elem_index];
        }

        pub fn set(
            self: *Self,
            indices: [rank]usize,
            value: T,
        ) void {
            const elem_index = self.elementOffset(indices);
            self.storage[elem_index] = value;
        }

        pub fn isContiguous(self: *const Self) bool {
            var expected: isize = 1;
            var axis = rank;
            while (axis > 0) {
                axis -= 1;
                if (self.shape[axis] > 1 and self.strides[axis] != expected)
                    return false;
                expected *= @intCast(self.shape[axis]);
            }
            return true;
        }

        pub fn contiguousSlice(self: *const Self) ?[]T {
            if (!self.isContiguous()) return null;
            return self.storage[self.offset..][0..self.len()];
        }
    };
}

/// Read-only view into memory for a given tensor.
pub fn ConstView(comptime T: type, comptime tensor_rank: usize) type {
    return struct {
        const Self = @This();
        pub const scalar_type = T;
        pub const dtype = Dtype.fromScalar(T);
        pub const rank = tensor_rank;

        storage: []const T,
        shape: [tensor_rank]usize,
        strides: [tensor_rank]isize,
        offset: usize,

        pub fn len(self: *const Self) usize {
            return Shape(rank).init(&self.shape).elementCount();
        }
        pub fn elementOffset(
            self: *const Self,
            indices: [rank]usize,
        ) usize {
            var offset: isize = @intCast(self.offset);
            for (self.strides, indices) |stride, index| {
                offset += @as(isize, @intCast(index)) * stride;
            }
            std.debug.assert(offset >= 0);
            return @intCast(offset);
        }

        pub fn get(
            self: *const Self,
            indices: [rank]usize,
        ) T {
            const elem_index = self.elementOffset(indices);
            return self.storage[elem_index];
        }

        pub fn isContiguous(self: *const Self) bool {
            var expected: isize = 1;
            var axis = rank;
            while (axis > 0) {
                axis -= 1;
                if (self.shape[axis] > 1 and self.strides[axis] != expected)
                    return false;
                expected *= @intCast(self.shape[axis]);
            }
            return true;
        }

        pub fn contiguousSlice(self: *const Self) ?[]const T {
            if (!self.isContiguous()) return null;
            return self.storage[self.offset..][0..self.len()];
        }
    };
}

pub fn Layout(comptime max_rank: usize) type {
    return struct {
        const This = @This();
        offset: usize,
        strides: [max_rank]isize,

        pub fn contiguous(shape: Shape(max_rank)) This {
            var result: This = .{
                .offset = 0,
                .strides = @splat(0),
            };

            var stride: isize = 1;
            var axis = shape.rank;

            while (axis > 0) {
                axis -= 1;
                result.strides[axis] = stride;
                stride *= @intCast(shape.at(axis));
            }

            return result;
        }
    };
}

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

// fn elementCount(comptime shape: Shape_T) usize {
//     var count: usize = 1;
//     for (shape) |extent| count *= extent;
//     return count;
// }

// Represents an input source
pub const Source = struct {
    kind: Kind,
    tensor: Id,

    pub const Kind = enum { input, parameter, constant, state };

    pub const Binding = enum {
        embed,
    };
};

/// Originators (producers) of tensors.
pub const Origin = union(enum) {
    source: Id,
    node: Id,
};

/// Tensor metadata specialized for one graph's maximum rank.
pub fn Info(comptime max_rank: usize) type {
    return struct {
        dtype: Dtype,
        shape: Shape(max_rank),
        layout: Layout(max_rank),

        // root tensor whose allocation backs this value *multiple values may refer to one tensor*
        storage_tensor: Id,

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
