/// Compile-time matmul traversal selected by graph lowering. `automatic` is
/// retained for direct kernel use where concrete layouts are not supplied by a
/// compiled graph.
pub const Strategy = enum {
    automatic,
    output_columns,
    contracted_axis,
    output_rows,
    scalar,
};

/// Lowered matmul configuration. Blocking and specialized small-batch
/// parameters can be added here without changing the operation dispatch
/// boundary.
pub const Plan = struct {
    strategy: Strategy = .automatic,
};
