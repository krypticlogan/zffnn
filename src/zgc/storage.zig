const std = @import("std");
const Tensor = @import("tensor.zig");
const Graph = @import("graph.zig");
const Dtype = @import("dtype.zig").Dtype;

pub const StorageRegion = struct { offset: usize, len_bytes: usize, alignment: usize };

pub fn MemoryPlan(comptime capacities: Graph.Capacity, comptime g: Graph.Graph(capacities)) type {
    var memory_plan: [g.tensor_ct]StorageRegion = undefined;

    var max_alignment: usize = 1;
    var offset: usize = 0;
    for (0..g.tensor_ct) |tensor_id| {
        const tensor_info = g.tensors[tensor_id].?;
        if (tensor_info.storage_tensor != tensor_id) continue;

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

    for (0..g.tensor_ct) |tensor_id| {
        const storage_tensor = g.tensors[tensor_id].?.storage_tensor;
        if (storage_tensor != tensor_id) {
            memory_plan[tensor_id] = memory_plan[storage_tensor];
        }
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
                const owns_storage = info.storage_tensor == tensor_id;
                const padding = if (owns_storage)
                    region.offset - previous_end
                else
                    0;
                std.debug.print(
                    "  t{d}: [{d}..{d}) bytes={d} align={d} padding={d} storage=t{d} dtype={s} shape=",
                    .{
                        tensor_id,
                        region.offset,
                        end,
                        region.len_bytes,
                        region.alignment,
                        padding,
                        info.storage_tensor,
                        @tagName(info.dtype),
                    },
                );
                Tensor.debugPrintShape(&info.shape);
                std.debug.print("\n", .{});
                if (owns_storage) previous_end = end;
            }
        }
    };
}

// pub fn memoryPlan() MemoryPlan {

// }
