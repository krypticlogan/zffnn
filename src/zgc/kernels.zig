const Op = @import("op.zig").Op;
const contraction = @import("kernels/contraction.zig");
const elementwise = @import("kernels/elementwise.zig");

/// Route graph operations to a kernel family.
pub fn execute(comptime op: Op.Compute, inputs: anytype, output: anytype) void {
    switch (op) {
        .relu => elementwise.relu(inputs[0], output),
        .exp => @compileError("exp kernel is not implemented"),
        .add => elementwise.add(inputs[0], inputs[1], output),
        .sub => @compileError("sub kernel is not implemented"),
        .matmul => contraction.matmul(inputs[0], inputs[1], output),
        .softmax => @compileError("softmax kernel is not implemented"),
    }
}
