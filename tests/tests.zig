test {
    _ = @import("mat_op_validation.zig");
    _ = @import("activations.zig");
    _ = @import("test_helpers.zig");
}

pub fn mat_equal(a: anytype, b: anytype) bool {
    if (a.rows() != b.rows() or a.cols() != b.cols()) return false;
    for (a.data, b.data) |a_row, b_row| {
        if (@reduce(.Add, a_row) != @reduce(.Add, b_row)) return false;
    }
    return true;
}
