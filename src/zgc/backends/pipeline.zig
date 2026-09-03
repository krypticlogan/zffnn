const CountingBackend = @import("counting.zig").CountingBackend;
const GraphBackend = @import("graph.zig").GraphBackend;
const ValidationBackend = @import("validation.zig").ValidationBackend;
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
    const lowered_graph = Lowering.build(definition);
    const Validator = ValidationBackend(capacity);
    const Validated = Validator.validate(lowered_graph);
    const graph = Validated.graph;
    const SourcePlan = Source.Plan(
        Definition.Source,
        capacity,
        graph,
        source_configuration,
    );
    return Model(Definition.Source, capacity, Validated, SourcePlan);
}
