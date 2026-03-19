const std = @import("std");

pub const EmbedParamsOptions = struct {
    /// - Path to the directory containing the layer weight and bias files.
    /// - The files should be named 'w{layer_index}.bin' and 'b{layer_index}.bin' for each layer.
    dir: std.Build.LazyPath,
    /// Number of trainable layers.
    layer_count: usize,
    /// Name of the import for the embedding parameters.
    import_name: []const u8 = "embed_params",
};

pub fn addEmbeddedParams(
    b: *std.Build,
    target_mod: *std.Build.Module,
    options: EmbedParamsOptions,
) void {
    const wf = b.addWriteFiles();

    var zig_src: std.ArrayList(u8) = .empty;
    defer zig_src.deinit();
    
    zig_src.writer(b.allocator).writeAll("pub const weights = [_][]const u8{\n") catch unreachable;
    for (1..options.layer_count + 1) |i| {
        const w_src = options.dir.path(b, b.fmt("w{d}.bin", .{i}));
        _ = wf.addCopyFile(w_src, b.fmt("w{d}.bin", .{i}));
        zig_src.writer(b.allocator).print("    @embedFile(\"w{d}.bin\"),\n", .{i}) catch unreachable;
    }
    zig_src.writer(b.allocator).writeAll("};\n\n") catch unreachable;

    zig_src.writer(b.allocator).writeAll("pub const biases = [_][]const u8{\n") catch unreachable;
    for (1..options.layer_count + 1) |i| {
        const b_src = options.dir.path(b, b.fmt("b{d}.bin", .{i}));
        _ = wf.addCopyFile(b_src, b.fmt("b{d}.bin", .{i}));
        zig_src.writer(b.allocator).print("    @embedFile(\"b{d}.bin\"),\n", .{i}) catch unreachable;
    }
    zig_src.writer(b.allocator).writeAll("};\n") catch unreachable;

    const embeds = wf.add("zffnn_embeds.zig", zig_src.items);
    target_mod.addAnonymousImport("embed_params", .{
        .root_source_file = embeds,
    });
}