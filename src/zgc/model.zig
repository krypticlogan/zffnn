const std = @import("std");
const Builder = @import("builder.zig");
const Graph = @import("graph.zig");
const Storage = @import("storage.zig");
const Tensor = @import("tensor.zig");

pub fn Model(
    comptime capacities: Graph.Capacity,
    comptime graph: Graph.Graph(capacities),
) type {
    // generate storage plan
    const plan = Storage.MemoryPlan(capacities, graph);
    return struct {
        const Self = @This();
        pub const build_graph = graph;
        pub const memory_plan = plan;
        pub const internal_capacity = capacities;

        memory: [plan.byte_count]u8 align(plan.alignment) = undefined,

        pub fn init() Self {
            return .{ .memory = @splat(0) };
        }

        pub fn run(model: *Self) void {
            inline for (0..graph.node_ct) |node_id| {
                model.executeNode(node_id);
            }
        }

        pub fn outputView(model: *const Self, comptime output_index: usize) blk: {
            if (output_index >= graph.output_ct) {
                @compileError("output index is outside the graph's outputs");
            }
            const tensor_id = graph.outputs[output_index].?;
            const info = graph.tensors[tensor_id].?;
            break :blk Tensor.ConstView(info.dtype.Scalar(), info.shape.rank);
        } {
            const tensor_id = graph.outputs[output_index].?;
            return model.constTensorView(tensor_id);
        }

        pub fn Source(
            model: *Self,
            comptime source_key: anytype,
            bytes: []const u8,
        ) void {
            const source_id: usize = switch (@typeInfo(@TypeOf(source_key))) {
                .@"enum" => @intFromEnum(source_key),
                else => @compileError("source keys must be enum values"),
            };
            if (source_id >= graph.sources.len) @compileError("The graph does not contain a source with the provided ID");
            const source = graph.sources[source_id] orelse @compileError("The graph does not contain a source with the provided ID");
            const source_region = model.tensorBytes(source.tensor);
            @memcpy(source_region, bytes);
        }

        fn executeNode(model: *Self, comptime node_id: Graph.Node.Id) void {
            const node = graph.nodes[node_id].?;
            const compute = switch (node.op) {
                .compute => |op| op,
                .view => return,
            };
            const InputViews = comptime blk: {
                var input_types: [node.input_count]type = undefined;
                for (0..node.input_count) |input_index| {
                    const tensor_id = graph.input_refs[node.input_start + input_index].?;
                    const info = graph.tensors[tensor_id].?;
                    input_types[input_index] = Tensor.View(
                        info.dtype.Scalar(),
                        info.shape.rank,
                    );
                }
                break :blk std.meta.Tuple(&input_types);
            };

            var inputs: InputViews = undefined;
            inline for (0..node.input_count) |input_index| {
                const tensor_id = comptime graph.input_refs[node.input_start + input_index].?;
                inputs[input_index] = model.tensorView(tensor_id);
            }

            const output = model.tensorView(node.result);
            compute.execute(inputs, output);
        }

        fn tensorView(model: *Self, comptime tensor_id: Tensor.Id) blk: {
            const info = graph.tensors[tensor_id].?;
            break :blk // return type
            Tensor.View(info.dtype.Scalar(), info.shape.rank);
        } {
            const info = graph.tensors[tensor_id].?;
            const T = info.dtype.Scalar();
            const bytes = model.tensorBytes(info.storage_tensor);
            const aligned_bytes: []align(@alignOf(T)) u8 = @alignCast(bytes);

            return .{
                .storage = std.mem.bytesAsSlice(T, aligned_bytes),
                .shape = info.shape.dims[0..info.shape.rank].*,
                .strides = info.layout.strides[0..info.shape.rank].*,
                .offset = info.layout.offset,
            };
        }

        fn constTensorView(model: *const Self, comptime tensor_id: Tensor.Id) blk: {
            const info = graph.tensors[tensor_id].?;
            break :blk Tensor.ConstView(info.dtype.Scalar(), info.shape.rank);
        } {
            const info = graph.tensors[tensor_id].?;
            const T = info.dtype.Scalar();
            const region = plan.tensor_regions[info.storage_tensor];
            const bytes = model.memory[region.offset .. region.offset + region.len_bytes];
            const aligned_bytes: []align(@alignOf(T)) const u8 = @alignCast(bytes);

            return .{
                .storage = std.mem.bytesAsSlice(T, aligned_bytes),
                .shape = info.shape.dims[0..info.shape.rank].*,
                .strides = info.layout.strides[0..info.shape.rank].*,
                .offset = info.layout.offset,
            };
        }

        fn tensorBytes(model: *Self, comptime tensor_id: Tensor.Id) []u8 {
            const region = plan.tensor_regions[tensor_id];
            return model.memory[region.offset .. region.offset + region.len_bytes];
        }

        pub fn debugPrintMemory(model: *const Self, comptime byte_limit: usize) void {
            const displayed_bytes = @min(byte_limit, plan.byte_count);
            std.debug.print(
                "ModelMemory(bytes={d}, alignment={d}, showing={d})\n",
                .{ plan.byte_count, plan.alignment, displayed_bytes },
            );

            for (model.memory[0..displayed_bytes], 0..) |byte, offset| {
                if (offset % 16 == 0) {
                    std.debug.print("  {x:0>6}: ", .{offset});
                }
                std.debug.print("{x:0>2} ", .{byte});
                if (offset % 16 == 15 or offset + 1 == displayed_bytes) {
                    std.debug.print("\n", .{});
                }
            }

            if (displayed_bytes < plan.byte_count) {
                std.debug.print(
                    "  ... {d} bytes omitted\n",
                    .{plan.byte_count - displayed_bytes},
                );
            }
        }
    };
}
