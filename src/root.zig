//! Root module exporting the public API.
pub const Builder = @import("zgc/builder.zig").Builder;
pub const Graph = @import("zgc/builder.zig").Graph;
pub const GraphCapacity = @import("zgc/builder.zig").GraphCapacity;
pub const Dtype = @import("zgc/storage.zig").Dtype;
pub const CountingBackend = @import("zgc/builder.zig").CapacityCountingBackend;
pub const GraphBackend = @import("zgc/builder.zig").GraphBackend;
pub const Storage =  @import("zgc/storage.zig").Storage;
pub const Extensions = @import("extensions/ext.zig");

// pub const network_validation = @import("network_validation.zig");
