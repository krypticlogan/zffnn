const std = @import("std");
const Graph = @import("graph.zig");

/// Compile-time storage selection for a graph source.
pub const Binding = union(enum) {
    /// Reserve storage in the model's inline memory.
    owned,
    /// Borrow storage supplied to a model instance at runtime.
    bound,
    /// Read immutable bytes directly from the executable's read-only data.
    embedded: []const u8,
};

pub const owned: Binding = .owned;
pub const bound: Binding = .bound;

pub fn embed(comptime bytes: []const u8) Binding {
    return .{ .embedded = bytes };
}

/// Normalize a named source configuration into the graph's enum-indexed source
/// table. Unspecified sources remain model-owned.
pub fn Plan(
    comptime SourceKey: type,
    comptime capacity: Graph.Capacity,
    comptime graph: Graph.Graph(capacity),
    comptime configuration: anytype,
) type {
    var bindings: [capacity.max_sources]Binding = @splat(.owned);

    for (std.meta.fields(@TypeOf(configuration))) |field| {
        const key: SourceKey = key: {
            for (std.meta.fields(SourceKey)) |source_field| {
                if (std.mem.eql(u8, source_field.name, field.name)) {
                    break :key @enumFromInt(source_field.value);
                }
            }
            @compileError("source configuration contains an unknown field: " ++ field.name);
        };
        const source_index: usize = @intCast(@intFromEnum(key));
        if (source_index >= graph.sources.len or graph.sources[source_index] == null) {
            @compileError("source configuration refers to a source that is not used by the graph: " ++ field.name);
        }

        const binding: Binding = @field(configuration, field.name);
        const source = graph.sources[source_index].?;
        const tensor = graph.tensors[source.tensor].?;
        const expected_bytes = tensor.shape.elementCount() * tensor.dtype.byteSize();

        switch (binding) {
            .owned => {},
            .bound => {
                if (source.kind != .input) {
                    @compileError("only input sources may use runtime-bound storage");
                }
            },
            .embedded => |bytes| {
                if (source.kind != .parameter and source.kind != .constant) {
                    @compileError("only parameter and constant sources may be embedded");
                }
                if (bytes.len != expected_bytes) {
                    @compileError(std.fmt.comptimePrint(
                        "embedded source '{s}' requires {d} bytes, received {d}",
                        .{ field.name, expected_bytes, bytes.len },
                    ));
                }
            },
        }
        bindings[source_index] = binding;
    }

    const normalized = bindings;
    return struct {
        pub const source_bindings = normalized;

        pub fn bindingForTensor(tensor_info: anytype) Binding {
            return switch (tensor_info.origin) {
                .source => |source_index| source_bindings[source_index],
                .node => .owned,
            };
        }

        pub fn isOwned(comptime tensor_info: anytype) bool {
            return bindingForTensor(tensor_info) == .owned;
        }
    };
}
