const std = @import("std");
const Dtype = @import("../dtype.zig").Dtype;
const Op = @import("../op.zig").Op;
const Tensor = @import("../tensor.zig");

pub const Limits = struct {
    max_rank: usize = 8,
    max_nodes: usize = 64,
    max_tensors: usize = 128,
    max_input_refs: usize = 192,
    max_outputs: usize = 8,
};

pub fn Value(comptime max_rank: usize) type {
    return struct {
        id: Tensor.Id,
        dtype: Dtype,
        shape: Tensor.Shape(max_rank),
    };
}

pub fn Definition(comptime SourceKey: type, comptime limits: Limits) type {
    return struct {
        const Self = @This();
        pub const max_rank = limits.max_rank;
        pub const Source = SourceKey;
        pub const ValueType = Value(max_rank);

        const Node = struct {
            op: Op,
            input_start: usize,
            input_count: usize,
            result: Tensor.Id,
        };

        const TensorRecord = struct {
            value: ValueType,
            origin: Tensor.Origin,
            source_kind: ?Tensor.Source.Kind = null,
        };

        nodes: [limits.max_nodes]Node = undefined,
        tensors: [limits.max_tensors]TensorRecord = undefined,
        input_refs: [limits.max_input_refs]Tensor.Id = undefined,
        outputs: [limits.max_outputs]Tensor.Id = undefined,
        node_count: usize = 0,
        tensor_count: usize = 0,
        input_ref_count: usize = 0,
        output_count: usize = 0,

        /// Run capacity counting, graph lowering, memory planning, and model
        /// generation for this completed definition.
        pub fn model(comptime definition: Self) type {
            return @import("pipeline.zig").model(Self, definition, .{});
        }

        /// Compile this definition with named source-storage overrides. Fields
        /// use SourceKey tag names and values from `zgc.Source`.
        pub fn modelWith(comptime definition: Self, comptime sources: anytype) type {
            return @import("pipeline.zig").model(Self, definition, sources);
        }
    };
}

/// The typed, front-facing model-definition backend.
pub fn DefinitionBackend(comptime SourceKey: type, comptime limits: Limits) type {
    const source_capacity = enumCapacity(SourceKey);
    const DefinitionType = Definition(SourceKey, limits);
    const ValueType = DefinitionType.ValueType;

    return struct {
        const Self = @This();
        pub const Source = SourceKey;
        pub const definition_limits = limits;
        pub const DefinitionOutput = DefinitionType;
        pub const TensorValue = ValueType;

        definition: DefinitionType = .{},
        used_sources: [source_capacity]bool = @splat(false),

        pub fn init() Self {
            return .{};
        }

        pub fn input(self: *Self, comptime source_key: SourceKey, comptime dtype: Dtype, comptime shape: []const usize) ValueType {
            return self.addSource(source_key, .input, dtype, shape);
        }

        pub fn parameter(
            self: *Self,
            comptime source_key: SourceKey,
            comptime dtype: Dtype,
            comptime shape: []const usize,
        ) ValueType {
            return self.addSource(source_key, .parameter, dtype, shape);
        }

        pub fn constant(
            self: *Self,
            comptime source_key: SourceKey,
            comptime dtype: Dtype,
            comptime shape: []const usize,
        ) ValueType {
            return self.addSource(source_key, .constant, dtype, shape);
        }

        pub fn relu(self: *Self, comptime tensor: ValueType) ValueType {
            return self.addCompute(.relu, &.{tensor});
        }

        pub fn exp(self: *Self, comptime tensor: ValueType) ValueType {
            return self.addCompute(.exp, &.{tensor});
        }

        pub fn add(self: *Self, comptime lhs: ValueType, comptime rhs: ValueType) ValueType {
            return self.addCompute(.add, &.{ lhs, rhs });
        }

        pub fn sub(self: *Self, comptime lhs: ValueType, comptime rhs: ValueType) ValueType {
            return self.addCompute(.sub, &.{ lhs, rhs });
        }

        pub fn matmul(self: *Self, comptime lhs: ValueType, comptime rhs: ValueType) ValueType {
            return self.addCompute(.{ .matmul = .{ .strategy = .scalar } }, &.{ lhs, rhs });
        }

        pub fn sum(self: *Self, comptime tensor: ValueType, comptime axis: i8) ValueType {
            return self.addCompute(.{ .sum = .{ .axis = axis } }, &.{tensor});
        }

        pub fn softmax(self: *Self, comptime tensor: ValueType, comptime axis: i8) ValueType {
            return self.addCompute(.{ .softmax = .{ .axis = axis } }, &.{tensor});
        }

        pub fn transpose(
            self: *Self,
            comptime tensor: ValueType,
            comptime axis_a: i8,
            comptime axis_b: i8,
        ) ValueType {
            if (axis_a < 0 or axis_b < 0 or axis_a >= tensor.shape.rank or axis_b >= tensor.shape.rank) {
                @compileError("transpose axis is outside the tensor rank");
            }
            var shape = tensor.shape;
            std.mem.swap(usize, &shape.dims[@intCast(axis_a)], &shape.dims[@intCast(axis_b)]);
            return self.addNode(
                .{ .view = .{ .transpose = .{ .axis_a = axis_a, .axis_b = axis_b } } },
                &.{tensor},
                tensor.dtype,
                shape,
            );
        }

        pub fn output(self: *Self, comptime value: ValueType) void {
            if (self.definition.output_count == limits.max_outputs) {
                @compileError("definition exceeds max_outputs");
            }
            self.definition.outputs[self.definition.output_count] = value.id;
            self.definition.output_count += 1;
        }

        pub fn finish(self: *const Self) DefinitionType {
            return self.definition;
        }

        fn addSource(
            self: *Self,
            comptime source_key: SourceKey,
            comptime kind: Tensor.Source.Kind,
            comptime dtype: Dtype,
            comptime shape_extents: []const usize,
        ) ValueType {
            if (shape_extents.len > limits.max_rank) @compileError("source shape exceeds definition max_rank");
            if (self.definition.tensor_count == limits.max_tensors) @compileError("definition exceeds max_tensors");

            const source_index: usize = @intCast(@intFromEnum(source_key));
            if (self.used_sources[source_index]) @compileError("a source key may only be defined once");
            self.used_sources[source_index] = true;

            const id = self.definition.tensor_count;
            const value: ValueType = .{
                .id = id,
                .dtype = dtype,
                .shape = .init(shape_extents),
            };
            self.definition.tensors[id] = .{
                .value = value,
                .origin = .{ .source = source_index },
                .source_kind = kind,
            };
            self.definition.tensor_count += 1;
            return value;
        }

        fn addCompute(
            self: *Self,
            comptime compute: Op.Compute,
            comptime inputs: []const ValueType,
        ) ValueType {
            const shape = compute.inferShape(inputs, limits.max_rank);
            return self.addNode(.{ .compute = compute }, inputs, inputs[0].dtype, shape);
        }

        fn addNode(
            self: *Self,
            comptime op: Op,
            comptime inputs: []const ValueType,
            comptime dtype: Dtype,
            comptime shape: Tensor.Shape(limits.max_rank),
        ) ValueType {
            if (self.definition.node_count == limits.max_nodes) @compileError("definition exceeds max_nodes");
            if (self.definition.tensor_count == limits.max_tensors) @compileError("definition exceeds max_tensors");
            if (self.definition.input_ref_count + inputs.len > limits.max_input_refs) @compileError("definition exceeds max_input_refs");

            const node_id = self.definition.node_count;
            const tensor_id = self.definition.tensor_count;
            const input_start = self.definition.input_ref_count;
            for (inputs) |input_value| {
                self.definition.input_refs[self.definition.input_ref_count] = input_value.id;
                self.definition.input_ref_count += 1;
            }
            self.definition.nodes[node_id] = .{
                .op = op,
                .input_start = input_start,
                .input_count = inputs.len,
                .result = tensor_id,
            };
            const value: ValueType = .{ .id = tensor_id, .dtype = dtype, .shape = shape };
            self.definition.tensors[tensor_id] = .{
                .value = value,
                .origin = .{ .node = node_id },
            };
            self.definition.node_count += 1;
            self.definition.tensor_count += 1;
            return value;
        }
    };
}

fn enumCapacity(comptime Enum: type) usize {
    const info = @typeInfo(Enum);
    if (info != .@"enum") @compileError("DefinitionBackend source keys must be an enum type");
    var capacity: usize = 0;
    for (info.@"enum".fields) |field| {
        if (field.value < 0) @compileError("source enum values must be non-negative");
        capacity = @max(capacity, @as(usize, @intCast(field.value)) + 1);
    }
    return capacity;
}
