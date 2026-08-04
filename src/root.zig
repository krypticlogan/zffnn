//! Root module exporting the public API.
pub const Builder = @import("zgc/builder.zig").Builder;
pub const Graph = @import("zgc/builder.zig").Graph;
pub const GraphCapacity = @import("zgc/graph.zig").Capacity;
pub const Dtype = @import("zgc/dtype.zig").Dtype;
pub const Op = @import("zgc/op.zig").Op;
pub const Tensor = @import("zgc/tensor.zig");
pub const Validation = @import("zgc/validation.zig");
pub const CountingBackend = @import("zgc/builder.zig").CapacityCountingBackend;
pub const GraphBackend = @import("zgc/builder.zig").GraphBackend;
pub const Model = @import("zgc/model.zig").Model;
pub const Storage = @import("zgc/storage.zig");
pub const Extensions = @import("extensions/ext.zig");

// pub const network_validation = @import("network_validation.zig");
