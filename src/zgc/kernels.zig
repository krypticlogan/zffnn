const Op = @import("op.zig").Op;
const elementwise = @import("kernels/elementwise.zig");

/// Route graph operations to a kernel family.
pub fn execute(comptime op: Op, inputs: anytype, output: anytype) void {
    switch (op) {
        .relu => elementwise.relu(inputs[0], output),
        .add => @compileError("add kernel is not implemented"),
        .matmul => @compileError("matmul kernel is not implemented"),
        .softmax => @compileError("softmax kernel is not implemented"),
        .transpose => @compileError("transpose kernel is not implemented"),
    }
}
