const std = @import("std");
const Tensor = @import("tensor.zig");
const Dtype = @import("dtype.zig").Dtype;
const Shape = Tensor.Shape;

pub const NodeType = enum { compute, view, source, output };

pub const Node = struct {
    pub const Id = usize;
    op: Op,
    input_start: usize,
    input_count: usize,
    result: Tensor.Id,
};

pub const Op = union(enum) {
    relu,
    add,
    matmul,
    softmax: SoftmaxAttrs,
    transpose: TransposeAttrs,

    pub const SoftmaxAttrs = struct { axis: i8 };
    pub const TransposeAttrs = struct { axis_a: i8, axis_b: i8 };

    pub fn debugPrint(op: Op) void {
        switch (op) {
            .relu => std.debug.print("relu", .{}),
            .add => std.debug.print("add", .{}),
            .matmul => std.debug.print("matmul", .{}),
            .softmax => |attrs| {
                std.debug.print("softmax(axis={d})", .{attrs.axis});
            },
            .transpose => |attrs| {
                std.debug.print(
                    "transpose(axes={d},{d})",
                    .{ attrs.axis_a, attrs.axis_b },
                );
            },
        }
    }
};

pub const Value = struct {
    id: Tensor.Id,
    dtype: Dtype,
    shape: Shape,

    pub fn debugPrint(value: Value) void {
        std.debug.print(
            "t{d}: {s} shape=",
            .{ value.id, @tagName(value.dtype) },
        );
        Tensor.debugPrintShape(&value.shape);
        std.debug.print("\n", .{});
    }
};

pub fn Builder(comptime Backend: type) type {
    return struct {
        backend: *Backend,
        const Self = @This();

        pub fn input(self: *Self, dtype: Dtype, shape: []const usize) Value {
            return (self.backend.addSource(.{
                .dtype = dtype,
                .shape = .init(shape),
            }));
        }

        pub fn parameter(self: *Self, dtype: Dtype, shape: []const usize) Value {
            return (self.backend.addSource(.{
                .dtype = dtype,
                .shape = .init(shape),
            }));
        }

        pub fn constant(self: *Self, dtype: Dtype, shape: []const usize) Value {
            return (self.backend.addSource(.{
                .dtype = dtype,
                .shape = .init(shape),
            }));
        }

        pub fn matmul(self: *Self, lhs: Value, rhs: Value) Value {
            // const output_info = inferMatmul(lhs, rhs); // tensor info
            const output_info = Tensor.Info{
                .dtype = lhs.dtype,
                .producer = self.backend.graph.node_ct,
                .shape = .init(&.{ lhs.shape.at(0), rhs.shape.at(1) }),
            };
            return self.backend.addNode(.matmul, &.{ lhs, rhs }, output_info);
        }

        pub fn relu(self: *Self, input_ref: Value) Value {
            const output_info = Tensor.Info{
                .dtype = input_ref.dtype,
                .shape = input_ref.shape,
            };

            return self.backend.addNode(.relu, &.{input_ref}, output_info);
        }

        pub fn output(self: *Self, value: Value) void {
            self.backend.markOutput(value);
        }
    };
}

pub const GraphCapacity = struct {
    max_nodes: usize = 0,
    max_input_refs: usize = 0,
    max_tensors: usize = 0,
    max_outputs: usize = 0,

    pub fn debugPrint(capacity: GraphCapacity) void {
        std.debug.print(
            "GraphCapacity(nodes={d}, input_refs={d}, tensors={d}, outputs={d})\n",
            .{
                capacity.max_nodes,
                capacity.max_input_refs,
                capacity.max_tensors,
                capacity.max_outputs,
            },
        );
    }
};
pub fn Graph(capacity: GraphCapacity) type {
    return struct {
        const Self = @This();

        nodes: [capacity.max_nodes]?Node = .{null} ** capacity.max_nodes,
        tensors: [capacity.max_tensors]?Tensor.Info = .{null} ** capacity.max_tensors,
        input_refs: [capacity.max_input_refs]?Tensor.Id = .{null} ** capacity.max_input_refs,
        outputs: [capacity.max_outputs]?Tensor.Id = .{null} ** capacity.max_outputs,

        node_ct: usize = 0,
        input_ref_ct: usize = 0,
        tensor_ct: usize = 0,
        output_ct: usize = 0,

        pub fn init() Self {
            return Self{};
        }

        pub fn insertNode(g: *Self, node: Node) void {
            g.nodes[g.node_ct] = node;
            g.node_ct += 1;
        }

        pub fn insertTensor(g: *Self, info: Tensor.Info) Tensor.Id {
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

            const producer_id = info.producer orelse {
                std.debug.print(" (source)\n", .{});
                return;
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

pub fn GraphBackend(comptime capacities: GraphCapacity) type {
    return struct {
        const Self = @This();
        graph: Graph(capacities) = .init(),

        pub fn init() Self {
            return Self{};
        }

        fn nextNodeID(self: *Self) usize {
            return self.graph.node_ct;
        }

        fn nextTensorID(self: *Self) usize {
            return self.graph.tensor_ct;
        }

        pub fn addSource( // add tensor info to graph.inputs
            self: *Self,
            info: Tensor.Info,
        ) Value {
            const id = self.graph.insertTensor(info);
            return .{
                .id = @intCast(id),
                .dtype = info.dtype,
                .shape = info.shape,
            };
        }

        pub fn addNode(
            self: *@This(),
            op: Op,
            inputs: []const Value,
            output: Tensor.Info,
        ) Value {
            const node_id = self.nextNodeID();
            var result_info = output;
            result_info.producer = node_id;
            const result_id = self.graph.insertTensor(result_info);

            for (inputs) |input| {
                self.graph.insertRef(input.id);
            }

            self.graph.insertNode(.{
                .op = op,
                .input_count = inputs.len,
                .input_start = self.graph.input_ref_ct - inputs.len,
                .result = result_id,
            });

            return .{
                .id = @intCast(result_id),
                .dtype = output.dtype,
                .shape = output.shape,
            };
        }

        pub fn markOutput(self: *@This(), value: Value) void {
            self.graph.insertOutput(value.id);
        }

        pub fn finish(self: *@This()) Graph(capacities) {
            return self.graph;
        }
    };
}

pub const CapacityCountingBackend = struct {
    counts: GraphCapacity = .{},
    graph: struct { node_ct: usize } = .{ .node_ct = 0 },

    pub fn addSource(
        self: *@This(),
        info: Tensor.Info,
    ) Value {
        const id = self.counts.max_tensors;
        self.counts.max_tensors += 1;

        return .{
            .id = @intCast(id),
            .dtype = info.dtype,
            .shape = info.shape,
        };
    }

    pub fn addNode(
        self: *@This(),
        op: Op,
        inputs: []const Value,
        output: Tensor.Info,
    ) Value {
        _ = op;

        self.counts.max_nodes += 1;
        self.graph.node_ct += 1;
        self.counts.max_input_refs += inputs.len;

        const id = self.counts.max_tensors;
        self.counts.max_tensors += 1;

        return .{
            .id = @intCast(id),
            .dtype = output.dtype,
            .shape = output.shape,
        };
    }

    pub fn markOutput(self: *@This(), value: Value) void {
        _ = value;
        self.counts.max_outputs += 1;
    }
};
