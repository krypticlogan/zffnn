const std = @import("std");
const codegen = @import("embedding.zig");

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    var args = try init.minimal.args.iterateAllocator(gpa);
    _ = args.skip(); // argv[0]

    const layer_count_str = args.next() orelse return error.MissingLayerCount;
    const embed_file_name = args.next() orelse return error.MissingEmbedFileName;
    const input_dir = args.next() orelse return error.MissingInputDir;
    const output_dir = args.next() orelse return error.MissingOutputDir;

    const layer_count = try std.fmt.parseInt(usize, layer_count_str, 10);

    try codegen.writeEmbeddedParamsBundle(
        init.io,
        gpa,
        embed_file_name,
        input_dir,
        output_dir,
        layer_count,
    );
}
