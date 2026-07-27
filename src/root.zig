//! Root module exporting the public API.
pub const Activation = @import("activations.zig").Activation;
pub const Mat = @import("matrix.zig").Mat;

pub const Builder = @import("builder.zig").Builder;
pub const Graph = @import("builder.zig").Graph;
pub const GraphCapacity = @import("builder.zig").GraphCapacity;
pub const Dtype = @import("dtype.zig").Dtype;
pub const CountingBackend = @import("builder.zig").CapacityCountingBackend;
pub const GraphBackend = @import("builder.zig").GraphBackend;
pub const NN = @import("network.zig").NN;
pub const network_validation = @import("network_validation.zig");
