//! Root module exporting the public API.
pub const Activation = @import("activations.zig").Activation;
pub const Mat = @import("matrix.zig").Mat;
pub const NN = @import("network.zig").NN;
pub const network_validation = @import("network_validation.zig");
