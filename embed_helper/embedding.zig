const std = @import("std");

pub fn writeEmbeddedParamsBundle(
    io: std.Io,
    allocator: std.mem.Allocator,
    embed_file_name: []const u8,
    input_dir_path: []const u8,
    output_dir_path: []const u8,
    layer_count: usize,
) !void {
    const cwd = std.Io.Dir.cwd();

    try cwd.createDirPath(io, output_dir_path);

    var input_dir = try cwd.openDir(io, input_dir_path, .{});
    defer input_dir.close(io);

    var output_dir = try cwd.openDir(io, output_dir_path, .{});
    defer output_dir.close(io);

    var zig_src: std.ArrayList(u8) = .empty;
    defer zig_src.deinit(allocator);

    try zig_src.appendSlice(allocator, "pub const weights = [_][]const u8{\n");
    for (1..layer_count + 1) |i| {
        const src_name = try std.fmt.allocPrint(allocator, "w{d}.bin", .{i});
        defer allocator.free(src_name);

        const line = try std.fmt.allocPrint(allocator, "    @embedFile(\"w{d}.bin\"),\n", .{i});
        defer allocator.free(line);
        try copyFileIntoDir(io, input_dir, output_dir, src_name);
        try zig_src.appendSlice(allocator, line);
    }
    try zig_src.appendSlice(allocator, "};\n\n");

    try zig_src.appendSlice(allocator, "pub const biases = [_][]const u8{\n");
    for (1..layer_count + 1) |i| {
        const src_name = try std.fmt.allocPrint(allocator, "b{d}.bin", .{i});
        defer allocator.free(src_name);

        const line = try std.fmt.allocPrint(allocator, "    @embedFile(\"b{d}.bin\"),\n", .{i});
        defer allocator.free(line);
        try copyFileIntoDir(io, input_dir, output_dir, src_name);
        try zig_src.appendSlice(allocator, line);
    }
    try zig_src.appendSlice(allocator, "};\n");

    try output_dir.writeFile(io, .{
        .sub_path = embed_file_name,
        .data = zig_src.items,
    });
}

fn copyFileIntoDir(
    io: std.Io,
    src_dir: std.Io.Dir,
    dst_dir: std.Io.Dir,
    name: []const u8,
) !void {
    var src_file = try src_dir.openFile(io, name, .{});
    defer src_file.close(io);

    var dst_file = try dst_dir.createFile(io, name, .{ .truncate = true });
    defer dst_file.close(io);

    var rbuf: [4096]u8 = undefined;
    var wbuf: [4096]u8 = undefined;

    var r = src_file.reader(io, &rbuf);
    var w = dst_file.writer(io, &wbuf);

    _ = try r.interface.streamRemaining(&w.interface);
    try w.interface.flush();
}
