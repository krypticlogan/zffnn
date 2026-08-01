const std = @import("std");
const Builder = @import("builder.zig");
const Tensor = @import("tensor.zig");
const Graph = @import("builder.zig").Graph;
pub const Dtype = enum {
    f32,
    f16,
    i8,

    pub const Kind = enum {
        float,
        signed_integer,
    };

    pub fn Scalar(comptime self: Dtype) type {
        return switch (self) {
            .f32 => f32,
            .f16 => f16,
            .i8 => i8,
        };
    }

    pub fn fromScalar(comptime T: type) Dtype {
        return if (T == f32)
            .f32
        else if (T == f16)
            .f16
        else if (T == i8)
            .i8
        else
            @compileError("unsupported tensor scalar type: " ++ @typeName(T));
    }

    pub fn kind(comptime dtype: Dtype) Kind {
        return switch (dtype) {
            .f32, .f16 => .float,
            .i8 => .signed_integer,
        };
    }

    pub fn Vector(comptime dtype: Dtype, comptime len: usize) type {
        return @Vector(len, dtype.Scalar());
    }

    pub fn zero(comptime dtype: Dtype) dtype.Scalar() {
        return 0;
    }

    pub fn vectorZero(comptime dtype: Dtype, comptime len: usize) dtype.Vector(len) {
        return @splat(dtype.zero());
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

pub fn MemoryPlan(comptime capacities: Builder.GraphCapacity, comptime g: Graph(capacities)) type {
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
        pub const tensor_regions: [g.tensor_ct]StorageRegion = regions;
        pub const byte_count: usize = total_bytes;
        pub const alignment: usize = storage_alignment;

        pub fn debugPrint() void {
            std.debug.print(
                "MemoryPlan(bytes={d}, alignment={d}, tensors={d})\n",
                .{ byte_count, alignment, tensor_regions.len },
            );

            var previous_end: usize = 0;
            for (tensor_regions, 0..) |region, tensor_id| {
                const info = g.tensors[tensor_id].?;
                const end = region.offset + region.len_bytes;
                std.debug.print(
                    "  t{d}: [{d}..{d}) bytes={d} align={d} padding={d} dtype={s} shape=",
                    .{
                        tensor_id,
                        region.offset,
                        end,
                        region.len_bytes,
                        region.alignment,
                        region.offset - previous_end,
                        @tagName(info.dtype),
                    },
                );
                Tensor.debugPrintShape(&info.shape);
                std.debug.print("\n", .{});
                previous_end = end;
            }
        }
    };
}

// pub fn memoryPlan() MemoryPlan {

// }
