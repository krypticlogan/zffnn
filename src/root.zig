//! Root module exporting the public API.
pub const GraphCapacity = @import("zgc/graph.zig").Capacity;
pub const Dtype = @import("zgc/dtype.zig").Dtype;
pub const Op = @import("zgc/op.zig").Op;
pub const Tensor = @import("zgc/tensor.zig");
pub const Validation = @import("zgc/validation.zig");
pub const DefinitionBackend = @import("zgc/backends/definition.zig").DefinitionBackend;
pub const DefinitionLimits = @import("zgc/backends/definition.zig").Limits;
pub const Model = @import("zgc/model.zig").Model;
pub const Storage = @import("zgc/storage.zig");
pub const Extensions = @import("extensions/ext.zig");