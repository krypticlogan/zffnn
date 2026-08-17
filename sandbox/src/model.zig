const std = @import("std");
const zgc = @import("zgc");

pub const batch_size = 2;
pub const input_size = 3;
pub const hidden_size = 4;
pub const output_size = 2;

pub const Sources = enum(usize) {
    input,
    hidden_weights,
    hidden_bias,
    output_weights,
    output_bias,
};

pub const Definition = zgc.DefinitionBackend(Sources, .{
    .max_rank = 2,
    .max_nodes = 8,
    .max_tensors = 16,
    .max_input_refs = 16,
    .max_outputs = 2,
});

fn defineGraph(builder: *Definition) void {
    const input = builder.input(
        Sources.input,
        .f32,
        &.{ batch_size, input_size },
    );
    const hidden_weights = builder.parameter(
        Sources.hidden_weights,
        .f32,
        &.{ input_size, hidden_size },
    );
    const hidden_bias = builder.parameter(
        Sources.hidden_bias,
        .f32,
        &.{hidden_size},
    );
    const output_weights = builder.parameter(
        Sources.output_weights,
        .f32,
        &.{ hidden_size, output_size },
    );
    const output_bias = builder.parameter(
        Sources.output_bias,
        .f32,
        &.{output_size},
    );

    const hidden_linear = builder.add(
        builder.matmul(input, hidden_weights),
        hidden_bias,
    );
    const hidden = builder.relu(hidden_linear);
    const logits = builder.add(
        builder.matmul(hidden, output_weights),
        output_bias,
    );
    builder.output(builder.softmax(logits, 1));
}

pub const definition = blk: {
    var builder = Definition.init();
    defineGraph(&builder);
    break :blk builder.finish();
};

pub const Model = definition.model();
pub const capacity = Model.internal_capacity;
pub const graph = Model.build_graph;

pub const input_values = [_]f32{
    1, 2, 3,
    4, 5, 6,
};

pub const hidden_weight_values = [_]f32{
    1, 0, -1, 0.5,
    0, 1, 0,  -0.5,
    1, 1, 1,  0,
};

pub const hidden_bias_values = [_]f32{ 0.5, -1, 0, 1 };

pub const output_weight_values = [_]f32{
    1,   -1,
    0,   1,
    0.5, 0,
    -1,  0.5,
};

pub const output_bias_values = [_]f32{ 0.25, -0.25 };

pub fn loadInput(model: *Model) void {
    const input_bytes: [@sizeOf(@TypeOf(input_values))]u8 =
        @bitCast(input_values);
    const hidden_weight_bytes: [@sizeOf(@TypeOf(hidden_weight_values))]u8 =
        @bitCast(hidden_weight_values);
    const hidden_bias_bytes: [@sizeOf(@TypeOf(hidden_bias_values))]u8 =
        @bitCast(hidden_bias_values);
    const output_weight_bytes: [@sizeOf(@TypeOf(output_weight_values))]u8 =
        @bitCast(output_weight_values);
    const output_bias_bytes: [@sizeOf(@TypeOf(output_bias_values))]u8 =
        @bitCast(output_bias_values);

    model.Source(Sources.input, &input_bytes);
    model.Source(Sources.hidden_weights, &hidden_weight_bytes);
    model.Source(Sources.hidden_bias, &hidden_bias_bytes);
    model.Source(Sources.output_weights, &output_weight_bytes);
    model.Source(Sources.output_bias, &output_bias_bytes);
}

pub fn keepOutputAlive(model: *const Model) void {
    std.mem.doNotOptimizeAway(model.outputView(0).storage);
}
