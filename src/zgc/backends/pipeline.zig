const CountingBackend = @import("counting.zig").CountingBackend;
const GraphBackend = @import("graph.zig").GraphBackend;
const Model = @import("../model.zig").Model;
const Source = @import("../source.zig");

pub fn model(
    comptime Definition: type,
    comptime definition: Definition,
    comptime source_configuration: anytype,
) type {
    const Counting = CountingBackend(Definition);
    const capacity = Counting.count(definition);
    const Lowering = GraphBackend(Definition, capacity);
    const graph = Lowering.build(definition);
    const SourcePlan = Source.Plan(
        Definition.Source,
        capacity,
        graph,
        source_configuration,
    );
    return Model(Definition.Source, capacity, graph, SourcePlan);
}
