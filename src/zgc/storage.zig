const std = @import("std");
const Builder = @import("builder.zig");
const Tensor = @import("tensor.zig");
const Graph = @import("builder.zig").Graph;
pub const Dtype = enum {
    f32,
    f16,
    i8,

    pub fn Scalar(comptime self: Dtype) type {
        return switch (self) {
            .f32 => f32,
            .f16 => f16,
            .i8 => i8,
        };
    }

    pub fn byteSize(comptime dtype: Dtype) usize {
        return switch (dtype) {
            .f32 => @sizeOf(f32),
            .f16 => @sizeOf(f16),
            .i8 => @sizeOf(i8),
        };
    }

    pub fn alignment(comptime dtype: Dtype) usize {
        return switch (dtype) {
            .f32 => @alignOf(f32),
            .f16 => @alignOf(f16),
            .i8 => @alignOf(i8),
        };
    }
};

pub const StorageRegion = struct { offset: usize, len_bytes: usize, alignment: usize };

pub fn MemoryPlan(comptime g: anytype) type {
    var memory_plan: [g.tensor_ct]StorageRegion = undefined;

    var max_alignment: usize = 1;
    var offset: usize = 0;
    for (0..g.tensor_ct) |tensor_id| {
        const tensor_info = g.tensors[tensor_id].?;
        const dtype = tensor_info.dtype;
        const alignment = dtype.alignment();
        const len_bytes = tensor_info.shape.elementCount() * dtype.byteSize();

        offset = std.mem.alignForward(usize, offset, alignment);

        const region: StorageRegion = .{
            .offset = offset,
            .len_bytes = len_bytes,
            .alignment = alignment,
        };
        memory_plan[tensor_id] = region;

        offset += len_bytes;
        max_alignment = @max(max_alignment, alignment);
    }

    const regions = memory_plan;
    const total_bytes = offset;
    const storage_alignment = max_alignment;

    return struct {
        pub const tensor_regions = regions;
        pub const byte_count = total_bytes;
        pub const alignment = storage_alignment;
    };
}

// pub fn memoryPlan() MemoryPlan {

// }
