const std = @import("std");
const zgc = @import("zgc");
const params = @import("model_params");

pub const batch_size = 1;
pub const input_size = 784;
pub const hidden_size_1 = 128;
pub const hidden_size_2 = 64;
pub const output_size = 10;

pub const Sources = enum(usize) {
    input,
    w1,
    b1,
    w2,
    b2,
    w3,
    b3,
};

pub const Definition = zgc.DefinitionBackend(Sources, .{
    .max_rank = 2,
    .max_nodes = 12,
    .max_tensors = 19,
    .max_input_refs = 18,
    .max_outputs = 1,
});

fn defineGraph(builder: *Definition) void {
    const input = builder.input(.input, .f32, &.{ batch_size, input_size });
    // The binary inputs serialize weights as [output, input]. Transpose views
    // adapt them to this graph's [input, output] matmul convention without a
    // copy or another memory-plan region.
    const w1 = builder.parameter(.w1, .f32, &.{ hidden_size_1, input_size });
    const b1 = builder.parameter(.b1, .f32, &.{hidden_size_1});
    const w2 = builder.parameter(.w2, .f32, &.{ hidden_size_2, hidden_size_1 });
    const b2 = builder.parameter(.b2, .f32, &.{hidden_size_2});
    const w3 = builder.parameter(.w3, .f32, &.{ output_size, hidden_size_2 });
    const b3 = builder.parameter(.b3, .f32, &.{output_size});

    const w1_for_matmul = builder.transpose(w1, 0, 1);
    const w2_for_matmul = builder.transpose(w2, 0, 1);
    const w3_for_matmul = builder.transpose(w3, 0, 1);

    const hidden_1 = builder.relu(builder.add(builder.matmul(input, w1_for_matmul), b1));
    const hidden_2 = builder.relu(builder.add(builder.matmul(hidden_1, w2_for_matmul), b2));
    const logits = builder.add(builder.matmul(hidden_2, w3_for_matmul), b3);
    builder.output(builder.softmax(logits, 1));
}

pub const definition = blk: {
    var builder = Definition.init();
    defineGraph(&builder);
    break :blk builder.finish();
};

pub const Model = definition.modelWith(.{
    .input = zgc.Source.bound,
    .w1 = zgc.Source.embed(params.w1),
    .b1 = zgc.Source.embed(params.b1),
    .w2 = zgc.Source.embed(params.w2),
    .b2 = zgc.Source.embed(params.b2),
    .w3 = zgc.Source.embed(params.w3),
    .b3 = zgc.Source.embed(params.b3),
});
pub const capacity = Model.internal_capacity;
pub const graph = Model.build_graph;

/// Stable placeholder data for the CLI and disassembly examples.
pub fn exampleInput() [input_size]f32 {
    return @splat(0);
}

pub fn bindInput(model: *Model, input: *const [input_size]f32) void {
    model.bindInput(.input, input) catch unreachable;
}

pub fn keepOutputAlive(model: *const Model) void {
    std.mem.doNotOptimizeAway(model.outputView(0).storage);
}
