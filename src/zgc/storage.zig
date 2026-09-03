const std = @import("std");
const Graph = @import("graph.zig");

pub const StorageRegion = struct {
    offset: usize,
    len_bytes: usize,
    alignment: usize,
};

pub fn MemoryPlan(
    comptime capacities: Graph.Capacity,
    comptime g: Graph.Graph(capacities),
    comptime SourcePlan: type,
) type {
    var memory_plan: [g.tensor_ct]?StorageRegion = @splat(null);

    var max_alignment: usize = 1;
    var offset: usize = 0;
    for (0..g.tensor_ct) |tensor_id| {
        const tensor_info = g.tensors[tensor_id].?;
        if (tensor_info.storage_tensor != tensor_id) continue;
        if (!SourcePlan.isOwned(tensor_info)) continue;

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
        pub const tensor_regions: [g.tensor_ct]?StorageRegion = regions;
        pub const byte_count: usize = total_bytes;
        pub const alignment: usize = storage_alignment;
    };
}
