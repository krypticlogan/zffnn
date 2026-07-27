const std = @import("std");
const testing = std.testing;

const embedding_codegen = @import("embedding_codegen");
const zffnn = @import("zffnn");

test "matrix and network types retain their compile-time shape metadata" {
    const Matrix = zffnn.Mat(3, 5);
    try testing.expectEqual(@as(usize, 3), Matrix.n);
    try testing.expectEqual(@as(usize, 5), Matrix.m);
    try testing.expectEqual([5]f32, Matrix.Row);
    try testing.expectEqual([3]f32, Matrix.Col);
    try testing.expectEqual(@Vector(5, f32), Matrix.RowVec);
    try testing.expectEqual(@Vector(3, f32), Matrix.ColVec);
    try testing.expectEqual([3][5]f32, @TypeOf(@as(Matrix, undefined).data));

    const definition: []const struct { usize, zffnn.Activation } = &.{
        .{ 2, .none },
        .{ 3, .relu },
        .{ 1, .softmax },
    };
    var network = zffnn.NN(definition, 4).new();

    try testing.expectEqual(@as(usize, 6), network.num_nodes);
    try testing.expectEqual(@as(usize, 3), network.layers.len);
    try testing.expectEqual(@as(usize, 3), network.layers[1].weights.rows());
    try testing.expectEqual(@as(usize, 2), network.layers[1].weights.cols());
    try testing.expectEqual(@as(usize, 3), network.layers[1].bias.rows());
    try testing.expectEqual(@as(usize, 1), network.layers[1].bias.cols());
    try testing.expectEqual(@as(usize, 4), network.layers[1].a.cols());
}

test "load_from_embeds reconstructs parameters and inference output" {
    const definition: []const struct { usize, zffnn.Activation } = &.{
        .{ 2, .none },
        .{ 2, .none },
    };
    const Net = zffnn.NN(definition, 1);
    var network = comptime Net.load_from_embeds();

    var expected_weights = zffnn.Mat(2, 2).create(0);
    expected_weights.load(.{
        .{ 2, -1 },
        .{ 0.5, 3 },
    });
    var expected_bias = zffnn.Mat(2, 1).create(0);
    expected_bias.load(.{
        .{1},
        .{-2},
    });

    try testing.expectEqualDeep(expected_weights.data, network.layers[1].weights.data);
    try testing.expectEqualDeep(expected_bias.data, network.layers[1].bias.data);

    var input = zffnn.Mat(1, 2).create(0);
    input.load(.{
        .{ 4, 5 },
    });
    var expected_output = zffnn.Mat(2, 1).create(0);
    expected_output.load(.{
        .{4},
        .{15},
    });

    const actual_output = network.forward(input);
    try testing.expectEqualDeep(expected_output.data, actual_output.data);
}

test "embedding generator copies parameters and emits the expected module" {
    const io = testing.io;
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.createDir(io, "input", .default_dir);
    var input_dir = try tmp.dir.openDir(io, "input", .{});
    defer input_dir.close(io);

    const weights = [_]f32{ 2, -1, 0.5, 3 };
    const biases = [_]f32{ 1, -2 };
    try input_dir.writeFile(io, .{
        .sub_path = "w1.bin",
        .data = std.mem.asBytes(&weights),
    });
    try input_dir.writeFile(io, .{
        .sub_path = "b1.bin",
        .data = std.mem.asBytes(&biases),
    });

    const root_path = try std.fmt.allocPrint(
        allocator,
        ".zig-cache/tmp/{s}",
        .{tmp.sub_path},
    );
    defer allocator.free(root_path);
    const input_path = try std.fs.path.join(allocator, &.{ root_path, "input" });
    defer allocator.free(input_path);
    const output_path = try std.fs.path.join(allocator, &.{ root_path, "output" });
    defer allocator.free(output_path);

    try embedding_codegen.writeEmbeddedParamsBundle(
        io,
        allocator,
        "embeds.zig",
        input_path,
        output_path,
        1,
    );

    var output_dir = try tmp.dir.openDir(io, "output", .{});
    defer output_dir.close(io);

    const generated = try output_dir.readFileAlloc(
        io,
        "embeds.zig",
        allocator,
        .limited(1024),
    );
    defer allocator.free(generated);
    try testing.expectEqualStrings(
        \\pub const weights = [_][]const u8{
        \\    @embedFile("w1.bin"),
        \\};
        \\
        \\pub const biases = [_][]const u8{
        \\    @embedFile("b1.bin"),
        \\};
        \\
    , generated);

    const copied_weights = try output_dir.readFileAlloc(
        io,
        "w1.bin",
        allocator,
        .limited(1024),
    );
    defer allocator.free(copied_weights);
    try testing.expectEqualSlices(u8, std.mem.asBytes(&weights), copied_weights);

    const copied_biases = try output_dir.readFileAlloc(
        io,
        "b1.bin",
        allocator,
        .limited(1024),
    );
    defer allocator.free(copied_biases);
    try testing.expectEqualSlices(u8, std.mem.asBytes(&biases), copied_biases);
}
