const std = @import("std");

pub fn writeEmbeddedParamsBundle(
    allocator: std.mem.Allocator,
    embed_file_name: []const u8,
    input_dir_path: []const u8,
    output_dir_path: []const u8,
    layer_count: usize,
) !void {
    const cwd = std.fs.cwd();

    try cwd.makePath(output_dir_path);

    var input_dir = try cwd.openDir(input_dir_path, .{});
    defer input_dir.close();

    var output_dir = try cwd.openDir(output_dir_path, .{});
    defer output_dir.close();

    var zig_src: std.ArrayList(u8) = .empty;
    defer zig_src.deinit(allocator);

    const w = zig_src.writer(allocator);

    try w.writeAll("pub const weights = [_][]const u8{\n");
    for (1..layer_count + 1) |i| {
        const src_name = try std.fmt.allocPrint(allocator, "w{d}.bin", .{i});
        defer allocator.free(src_name);

        try copyFileIntoDir(input_dir, output_dir, src_name);
        try w.print("    @embedFile(\"w{d}.bin\"),\n", .{i});
    }
    try w.writeAll("};\n\n");

    try w.writeAll("pub const biases = [_][]const u8{\n");
    for (1..layer_count + 1) |i| {
        const src_name = try std.fmt.allocPrint(allocator, "b{d}.bin", .{i});
        defer allocator.free(src_name);

        try copyFileIntoDir(input_dir, output_dir, src_name);
        try w.print("    @embedFile(\"b{d}.bin\"),\n", .{i});
    }
    try w.writeAll("};\n");

    try output_dir.writeFile(.{
        .sub_path = embed_file_name,
        .data = zig_src.items,
    });
}

fn copyFileIntoDir(
    src_dir: std.fs.Dir,
    dst_dir: std.fs.Dir,
    name: []const u8,
) !void {
    var src_file = try src_dir.openFile(name, .{});
    defer src_file.close();

    var dst_file = try dst_dir.createFile(name, .{ .truncate = true });
    defer dst_file.close();

    var rbuf: [4096]u8 = undefined;
    var wbuf: [4096]u8 = undefined;

    var r = src_file.reader(&rbuf);
    var w = dst_file.writer(&wbuf);

    _ = try r.interface.streamRemaining(&w.interface);
    try w.interface.flush();
}