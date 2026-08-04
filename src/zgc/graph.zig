const std = @import("std");
const Tensor = @import("tensor.zig");
const Dtype = @import("storage.zig").Dtype;
const Shape_T = Tensor.Shape_T;
const Op = @import("op.zig").Op;

pub const Node = struct {
    pub const Id = usize;
    op: Op,
    input_start: usize,
    input_count: usize,
    result: Tensor.Id,
};

pub fn Graph(comptime capacity: Capacity) type {
    return struct {
        const Self = @This();
        pub const max_rank = capacity.max_rank;
        pub const TensorInfo = Tensor.Info(max_rank);

        nodes: [capacity.max_nodes]?Node = .{null} ** capacity.max_nodes,
        tensors: [capacity.max_tensors]?TensorInfo = .{null} ** capacity.max_tensors,
        input_refs: [capacity.max_input_refs]?Tensor.Id = .{null} ** capacity.max_input_refs,
        outputs: [capacity.max_outputs]?Tensor.Id = .{null} ** capacity.max_outputs,
        sources: [capacity.max_sources]?Tensor.Source = .{null} ** capacity.max_sources,

        node_ct: usize = 0,
        input_ref_ct: usize = 0,
        tensor_ct: usize = 0,
        output_ct: usize = 0,
        source_ct: usize = 0,

        pub fn init() Self {
            return Self{};
        }


        pub fn insertSource(g: *Self, comptime source_index: usize, source: Tensor.Source) void {
            g.sources[source_index] = source;
            g.source_ct += 1;
        }

        pub fn insertNode(g: *Self, node: Node) void {
            g.nodes[g.node_ct] = node;
            g.node_ct += 1;
        }

        pub fn insertTensor(g: *Self, info: TensorInfo) Tensor.Id {
            const id = g.tensor_ct;
            g.tensors[id] = info;
            g.tensor_ct += 1;
            return id;
        }

        pub fn insertRef(g: *Self, tensor_id: Tensor.Id) void {
            g.input_refs[g.input_ref_ct] = tensor_id;
            g.input_ref_ct += 1;
        }

        pub fn insertOutput(g: *Self, tensor_id: Tensor.Id) void {
            g.outputs[g.output_ct] = tensor_id;
            g.output_ct += 1;
        }

        pub fn debugPrint(g: *const Self) void {
            std.debug.print(
                "Graph(nodes={d}/{d}, input_refs={d}/{d}, tensors={d}/{d}, outputs={d}/{d})\n",
                .{
                    g.node_ct,
                    capacity.max_nodes,
                    g.input_ref_ct,
                    capacity.max_input_refs,
                    g.tensor_ct,
                    capacity.max_tensors,
                    g.output_ct,
                    capacity.max_outputs,
                },
            );

            std.debug.print("Tensors:\n", .{});
            for (g.tensors[0..g.tensor_ct], 0..) |maybe_info, id| {
                maybe_info.?.debugPrint(id);
            }

            std.debug.print("Nodes:\n", .{});
            for (g.nodes[0..g.node_ct], 0..) |maybe_node, id| {
                const node = maybe_node.?;
                std.debug.print("  n{d}: ", .{id});
                node.op.debugPrint();
                std.debug.print("(", .{});

                for (0..node.input_count) |input_index| {
                    if (input_index != 0) std.debug.print(", ", .{});
                    const ref = g.input_refs[node.input_start + input_index].?;
                    std.debug.print("t{d}", .{ref});
                }

                std.debug.print(") -> t{d}\n", .{node.result});
            }

            std.debug.print("Outputs: [", .{});
            for (g.outputs[0..g.output_ct], 0..) |maybe_output, index| {
                if (index != 0) std.debug.print(", ", .{});
                std.debug.print("t{d}", .{maybe_output.?});
            }
            std.debug.print("]\n", .{});
        }

        pub fn debugPrintStructure(g: *const Self) void {
            std.debug.print("Graph structure:\n", .{});

            if (g.output_ct == 0) {
                std.debug.print("  (no graph outputs)\n", .{});
                return;
            }

            for (g.outputs[0..g.output_ct], 0..) |maybe_output, output_index| {
                std.debug.print("output[{d}]\n", .{output_index});

                var expanded_nodes: [capacity.max_nodes]bool =
                    @splat(false);
                var ancestor_is_last: [capacity.max_nodes * 2 + 2]bool =
                    @splat(true);

                g.debugPrintTensorTree(
                    maybe_output.?,
                    0,
                    true,
                    &ancestor_is_last,
                    &expanded_nodes,
                );
            }
        }

        fn debugPrintTensorTree(
            g: *const Self,
            tensor_id: Tensor.Id,
            depth: usize,
            is_last: bool,
            ancestor_is_last: []bool,
            expanded_nodes: *[capacity.max_nodes]bool,
        ) void {
            debugPrintTreePrefix(depth, is_last, ancestor_is_last);

            if (tensor_id >= g.tensor_ct or g.tensors[tensor_id] == null) {
                std.debug.print("t{d} (missing tensor metadata)\n", .{tensor_id});
                return;
            }

            const info = g.tensors[tensor_id].?;
            std.debug.print("t{d} [{s} ", .{ tensor_id, @tagName(info.dtype) });
            Tensor.debugPrintShape(&info.shape);
            std.debug.print("]", .{});

            // const producer_id = info.origin.node else {
            //
            //     return;
            // };
            //
            const producer_id = switch (info.origin) {
                .node => info.origin.node,
                .source => |source_index| {
                    const source = g.sources[source_index].?;
                    std.debug.print(" (source[{d}])={s}\n", .{ source_index, @tagName(source.kind) });
                    return;
                },
            };

            if (producer_id >= g.node_ct or g.nodes[producer_id] == null) {
                std.debug.print(" (missing producer n{d})\n", .{producer_id});
                return;
            }

            if (expanded_nodes[producer_id]) {
                std.debug.print(" (from n{d}, already shown)\n", .{producer_id});
                return;
            }

            std.debug.print("\n", .{});
            expanded_nodes[producer_id] = true;
            ancestor_is_last[depth] = is_last;

            const node = g.nodes[producer_id].?;
            debugPrintTreePrefix(depth + 1, true, ancestor_is_last);
            std.debug.print("n{d} ", .{producer_id});
            node.op.debugPrint();
            std.debug.print("\n", .{});

            ancestor_is_last[depth + 1] = true;
            for (0..node.input_count) |input_index| {
                const input_id =
                    g.input_refs[node.input_start + input_index].?;
                g.debugPrintTensorTree(
                    input_id,
                    depth + 2,
                    input_index == node.input_count - 1,
                    ancestor_is_last,
                    expanded_nodes,
                );
            }
        }

        fn debugPrintTreePrefix(
            depth: usize,
            is_last: bool,
            ancestor_is_last: []const bool,
        ) void {
            for (0..depth) |ancestor_depth| {
                if (ancestor_is_last[ancestor_depth]) {
                    std.debug.print("   ", .{});
                } else {
                    std.debug.print("│  ", .{});
                }
            }

            if (is_last) {
                std.debug.print("└─ ", .{});
            } else {
                std.debug.print("├─ ", .{});
            }
        }
    };
}

pub const Capacity = struct {
    max_nodes: usize = 0,
    max_input_refs: usize = 0,
    max_tensors: usize = 0,
    max_outputs: usize = 0,
    max_sources: usize = 0,
    max_rank: usize = 0,

    pub fn debugPrint(capacity: Capacity) void {
        std.debug.print(
            "Capacity(nodes={d}, input_refs={d}, tensors={d}, outputs={d}, rank={d}, sources={d})\n",
            .{
                capacity.max_nodes,
                capacity.max_input_refs,
                capacity.max_tensors,
                capacity.max_outputs,
                capacity.max_rank,
                capacity.max_sources,
            },
        );
    }
};