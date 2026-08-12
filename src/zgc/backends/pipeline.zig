const CountingBackend = @import("counting.zig").CountingBackend;
const GraphBackend = @import("graph.zig").GraphBackend;
const Model = @import("../model.zig").Model;

pub fn model(
    comptime Definition: type,
    comptime definition: Definition,
) type {
    const Counting = CountingBackend(Definition);
    const capacity = Counting.count(definition);
    const Lowering = GraphBackend(Definition, capacity);
    const graph = Lowering.build(definition);
    return Model(capacity, graph);
}
