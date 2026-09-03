const Op = @import("op.zig").Op;
const contraction = @import("kernels/contraction.zig");
const elementwise = @import("kernels/elementwise.zig");
const reduction = @import("kernels/reduction.zig");
const special = @import("kernels/special.zig");

/// Route graph operations to a kernel family.
pub fn execute(comptime op: Op.Compute, inputs: anytype, output: anytype) void {
    switch (op) {
        .relu => elementwise.relu(inputs[0], output),
        .exp => elementwise.exp(inputs[0], output),
        .add => elementwise.add(inputs[0], inputs[1], output),
        .sub => elementwise.sub(inputs[0], inputs[1], output),
        .matmul => |plan| contraction.matmulWithPlan(
            plan.strategy,
            inputs[0],
            inputs[1],
            output,
        ),
        .sum => |attrs| reduction.sum(inputs[0], output, attrs.axis),
        .softmax => |attrs| special.softmax(inputs[0], output, attrs.axis),
    }
}
