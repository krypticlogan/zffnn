const std = @import("std");
const zgc = @import("zgc");

const Sources = enum(usize) { input };
const Definition = zgc.DefinitionBackend(Sources, .{
    .max_rank = 2,
    .max_nodes = 2,
    .max_tensors = 3,
    .max_input_refs = 2,
    .max_outputs = 1,
});

const Model = model: {
    var builder = Definition.init();
    const input = builder.input(.input, .f32, &.{ 2, 3 });
    builder.output(builder.relu(builder.transpose(input, 0, 1)));
    break :model builder.finish().model();
};

test "inspection renders a generated model through a writer" {
    var buffer: [8192]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buffer);
    try zgc.Inspect.writeModel(Model, &writer, .{});
    const output = writer.buffered();

    try std.testing.expect(std.mem.indexOf(u8, output, "Capacity(nodes=2") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "transpose(axes=0,1)") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "relu(t1) -> t2") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "Graph structure:") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "MemoryPlan(bytes=48") != null);
}

test "inspection CLI selects individual representations" {
    var buffer: [2048]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buffer);
    try std.testing.expect(try zgc.Inspect.runCli(Model, &.{"summary"}, &writer));
    const output = writer.buffered();
    try std.testing.expect(std.mem.indexOf(u8, output, "== Capacity ==") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "== Graph ==") == null);
}

test "inspection CLI reports invalid commands" {
    var buffer: [2048]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buffer);
    try std.testing.expect(!try zgc.Inspect.runCli(Model, &.{"unknown"}, &writer));
    try std.testing.expect(std.mem.indexOf(u8, writer.buffered(), "usage: zgc-inspect") != null);
}

test "inspection renders bounded model memory" {
    var model = Model.init();
    var buffer: [512]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buffer);
    try zgc.Inspect.writeModelMemory(&model, &writer, 8);
    const output = writer.buffered();
    try std.testing.expect(std.mem.indexOf(u8, output, "showing=8") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "40 bytes omitted") != null);
}

test "inspection renders reused regions in tensor order" {
    const ReuseDefinition = zgc.DefinitionBackend(Sources, .{
        .max_rank = 1,
        .max_nodes = 3,
        .max_tensors = 4,
        .max_input_refs = 3,
        .max_outputs = 1,
    });
    const definition = comptime blk: {
        var builder = ReuseDefinition.init();
        const input = builder.input(.input, .f32, &.{4});
        const first = builder.relu(input);
        const second = builder.relu(first);
        builder.output(builder.relu(second));
        break :blk builder.finish();
    };
    const ReuseModel = definition.modelWith(.{ .input = zgc.Source.bound });
    var buffer: [2048]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buffer);

    try zgc.Inspect.writeMemoryPlan(ReuseModel, &writer);
    const output = writer.buffered();
    try std.testing.expect(std.mem.indexOf(u8, output, "MemoryPlan(bytes=32") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "t1: [0..16)") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "t3: [0..16)") != null);
}
