const std = @import("std");

pub fn build(b: *std.Build) void {

    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const nn_dep = b.dependency("zffnn", .{
        .target = target,
        .optimize = optimize,
    });
    const zffnn = nn_dep.module("zffnn");
    addEmbeddedParams(
        b, 
        zffnn, 
        b.path("model_params"), 
        3
    );

    const raylib_dep = b.dependency("raylib_zig", .{
        .target = target,
        .optimize = optimize,
    });

    const raylib = raylib_dep.module("raylib"); // main raylib module
    const raygui = raylib_dep.module("raygui"); // raygui module
    const raylib_artifact = raylib_dep.artifact("raylib"); // raylib C library

    const exe = b.addExecutable(.{
        .name = "demo",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "raylib", .module = raylib },
                .{ .name = "raygui", .module = raygui },
                .{ .name = "zffnn", .module = zffnn },
            },
            .link_libc = true,
        }),
    });
    exe.root_module.linkLibrary(raylib_artifact);
    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");

    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
}

pub fn addEmbeddedParams(
    b: *std.Build,
    target_mod: *std.Build.Module,
    /// - The files should be named 'w{layer_index}.bin' and 'b{layer_index}.bin' for each layer.
    dir: std.Build.LazyPath,
    /// Number of trainable layers.
    layer_count: usize,
) void {
    const wf = b.addWriteFiles();

    var zig_src: std.ArrayList(u8) = .empty;
    defer zig_src.deinit(b.allocator);
    
    zig_src.writer(b.allocator).writeAll("pub const weights = [_][]const u8{\n") catch unreachable;
    for (1..layer_count + 1) |i| {
        const w_src = dir.path(b, b.fmt("w{d}.bin", .{i}));
        _ = wf.addCopyFile(w_src, b.fmt("w{d}.bin", .{i}));
        zig_src.writer(b.allocator).print("    @embedFile(\"w{d}.bin\"),\n", .{i}) catch unreachable;
    }
    zig_src.writer(b.allocator).writeAll("};\n\n") catch unreachable;

    zig_src.writer(b.allocator).writeAll("pub const biases = [_][]const u8{\n") catch unreachable;
    for (1..layer_count + 1) |i| {
        const b_src = dir.path(b, b.fmt("b{d}.bin", .{i}));
        _ = wf.addCopyFile(b_src, b.fmt("b{d}.bin", .{i}));
        zig_src.writer(b.allocator).print("    @embedFile(\"b{d}.bin\"),\n", .{i}) catch unreachable;
    }
    zig_src.writer(b.allocator).writeAll("};\n") catch unreachable;

    const embeds = wf.add("zffnn_embeds.zig", zig_src.items);
    target_mod.addAnonymousImport("embed_params", .{
        .root_source_file = embeds,
    });
}
