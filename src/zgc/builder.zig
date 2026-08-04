const std = @import("std");
const Graph = @import("graph.zig");
const Tensor = @import("tensor.zig");
const Dtype = @import("dtype.zig").Dtype;
const Shape_T = Tensor.Shape_T;
const Op = @import("op.zig").Op;

/// Builder that recieves a backend (Graph / Counting) supports building a model.
/// Either begin with the `GraphBackend` and supply maximum counts yourself, or use the `CountingBackend` to compute maximums to be supplied to the `GraphBackend`.
pub fn Builder(comptime Backend: type) type {
    return struct {
        const ValueType = Backend.ValueType;
        backend: *Backend,
        const Self = @This();

        pub fn input(
            self: *Self,
            comptime source_key: anytype,
            comptime dtype: Dtype,
            comptime shape: Shape_T,
        ) ValueType {
            return self.backend.addSource(sourceIndex(source_key), .input, dtype, shape);
        }

        pub fn parameter(
            self: *Self,
            comptime source_key: anytype,
            comptime dtype: Dtype,
            comptime shape: Shape_T,
        ) ValueType {
            return self.backend.addSource(sourceIndex(source_key), .parameter, dtype, shape);
        }

        pub fn constant(
            self: *Self,
            comptime source_key: anytype,
            comptime dtype: Dtype,
            comptime shape: Shape_T,
        ) ValueType {
            return self.backend.addSource(sourceIndex(source_key), .constant, dtype, shape);
        }

        pub fn relu(self: *Self, comptime tensor: ValueType) ValueType {
            return self.backend.addNode(.relu, &.{tensor});
        }

        pub fn exp(self: *Self, comptime tensor: ValueType) ValueType {
            return self.backend.addNode(.exp, &.{tensor});
        }
        
        pub fn add(self: *Self, comptime lhs: ValueType, comptime rhs: ValueType) ValueType {
            return self.backend.addNode(.add, &.{lhs, rhs});
        }

        pub fn sub(self: *Self, comptime lhs: ValueType, comptime rhs: ValueType) ValueType {
            return self.backend.addNode(.sub, &.{lhs, rhs});
        }

        pub fn matmul(self: *Self, comptime lhs: ValueType, comptime rhs: ValueType) ValueType {
            return self.backend.addNode(.matmul, &.{ lhs, rhs });
        }
        
        pub fn softmax(
              self: *Self,
              comptime tensor: ValueType,
              comptime axis: i8,
          ) ValueType {
              return self.backend.addNode(.{ .softmax = .{ .axis = axis }}, &.{tensor});
          }
        
        pub fn output(self: *Self, comptime value: ValueType) void {
            self.backend.markOutput(value);
        }
    };
}

fn sourceIndex(comptime source_key: anytype) usize {
    return switch (@typeInfo(@TypeOf(source_key))) {
        .@"enum" => @intFromEnum(source_key),
        else => @compileError("source keys must be enum values"),
    };
}

/// Intermediate / Internal Graph type
pub fn Value(comptime max_rank: usize) type {
    return struct {
        id: Tensor.Id,
        dtype: Dtype,
        shape: Tensor.Shape(max_rank),

        pub fn debugPrint(value: @This()) void {
            std.debug.print(
                "t{d}: {s} shape=",
                .{ value.id, @tagName(value.dtype) },
            );
            Tensor.debugPrintShape(&value.shape);
            std.debug.print("\n", .{});
        }
    };
}

const CountingValue = struct {
    id: Tensor.Id,
    dtype: Dtype,
    rank: usize,
};

pub fn GraphBackend(comptime capacities: Graph.Capacity) type {
    return struct {
        const Self = @This();
        pub const max_rank = capacities.max_rank;
        const TensorInfo = Tensor.Info(max_rank);
        pub const ValueType = Value(max_rank);
        graph: Graph.Graph(capacities) = .init(),

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
            comptime source_index: usize,
            comptime kind: Tensor.Source.Kind,
            comptime dtype: Dtype,
            comptime shape: Shape_T,
        ) ValueType {
            const info = TensorInfo {
                .dtype = dtype,
                .shape = .init(shape),
                .origin = .{ .source = source_index },
            };
            const source = Tensor.Source {
                .kind = kind,
                .tensor = self.nextTensorID(),
            };
            self.graph.insertSource(source_index, source);
            const id = self.graph.insertTensor(info);
            return .{
                .id = @intCast(id),
                .dtype = info.dtype,
                .shape = info.shape,
            };
        }

        pub fn addNode(
            self: *@This(),
            comptime op: Op,
            comptime inputs: []const ValueType,
        ) ValueType {
            const node_id = self.nextNodeID();
            const result_shape = op.inferShape(inputs, max_rank);
            const result_info = TensorInfo{
                .dtype = inputs[0].dtype,
                .shape = result_shape,
                .origin = .{ .node = node_id },
            };
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
                .dtype = result_info.dtype,
                .shape = result_info.shape,
            };
        }

        pub fn markOutput(self: *@This(), value: ValueType) void {
            self.graph.insertOutput(value.id);
        }

        pub fn finish(self: *@This()) Graph.Graph(capacities) {
            return self.graph;
        }
    };
}

pub const CapacityCountingBackend = struct {
    const Self = @This();
    pub const ValueType = CountingValue;

    counts: Graph.Capacity = .{},
    graph: struct { node_ct: usize } = .{ .node_ct = 0 },

    pub fn addSource(
        self: *Self,
        comptime source_index: usize,
        comptime kind: Tensor.Source.Kind,
        comptime dtype: Dtype,
        comptime shape: Shape_T,
    ) ValueType {
        _ = kind;

        self.counts.max_sources = @max(self.counts.max_sources, source_index + 1);

        const id = self.counts.max_tensors;
        self.counts.max_tensors += 1;
        self.counts.max_rank = @max(self.counts.max_rank, shape.len);

        return .{
            .id = @intCast(id),
            .dtype = dtype,
            .rank = shape.len,
        };
    }

    pub fn addNode(
        self: *Self,
        op: Op,
        inputs: []const ValueType,
    ) ValueType {
        const output_rank = op.inferRank(inputs);

        self.counts.max_nodes += 1;
        self.graph.node_ct += 1;
        self.counts.max_input_refs += inputs.len;
        self.counts.max_rank = @max(self.counts.max_rank, output_rank);

        const id = self.counts.max_tensors;
        self.counts.max_tensors += 1;

        return .{
            .id = @intCast(id),
            .dtype = inputs[0].dtype,
            .rank = output_rank,
        };
    }

    pub fn markOutput(self: *Self, value: ValueType) void {
        _ = value;
        self.counts.max_outputs += 1;
    }
};
