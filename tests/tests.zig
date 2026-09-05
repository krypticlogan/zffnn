test {
    _ = @import("test_helpers.zig");
    _ = @import("mat_op_validation.zig");
    _ = @import("activations.zig");
    _ = @import("network.zig");
    _ = @import("network_validation.zig");
    _ = @import("meta.zig");
    _ = @import("definition.zig");
    _ = @import("model.zig");
    _ = @import("inspect.zig");
    _ = @import("lifetime_analysis.zig");
    _ = @import("memory_reuse.zig");
    _ = @import("tensor_view.zig");
    _ = @import("tensor_ops/tests.zig");
    _ = @import("validation.zig");
}
