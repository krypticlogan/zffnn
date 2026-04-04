const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const nn_dep = b.dependency("zffnn", .{
        .target = target,
        .optimize = optimize,
    });
    const zffnn = nn_dep.module("zffnn");
    
    { // Run the embed helper to generate the embeds.zig file
        const params_dir = b.path("model_params");
        const embed_file_name = "embeds.zig";
        
        const zffn_embed_gen = nn_dep.artifact("embed_helper");
        const run_gen = b.addRunArtifact(zffn_embed_gen);
        run_gen.addArg("3"); // number of trainable layers
        run_gen.addArg(embed_file_name);
        run_gen.addDirectoryArg(params_dir); // directory containing model parameters
        
        const out_dir = run_gen.addOutputDirectoryArg("zffnn_embeds"); // output directory for the generated embeds.zig file

        const embed_mod = b.createModule(.{
            .root_source_file = out_dir.path(b, embed_file_name),
            .target = target,
            .optimize = optimize,
        });
        // add the embed module to the zffnn import, the name "embed_params" must be used
        zffnn.addImport("embed_params", embed_mod);
    }

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

// pub fn addEmbeddedParams(
//     b: *std.Build,
//     target_mod: *std.Build.Module,
//     /// - The files should be named 'w{layer_index}.bin' and 'b{layer_index}.bin' for each layer.
//     dir: std.Build.LazyPath,
//     /// Number of trainable layers.
//     layer_count: usize,
// ) void {
//     const wf = b.addWriteFiles();

//     var zig_src: std.ArrayList(u8) = .empty;
//     defer zig_src.deinit(b.allocator);

//     zig_src.writer(b.allocator).writeAll("pub const weights = [_][]const u8{\n") catch unreachable;
//     for (1..layer_count + 1) |i| {
//         const w_src = dir.path(b, b.fmt("w{d}.bin", .{i}));
//         _ = wf.addCopyFile(w_src, b.fmt("w{d}.bin", .{i}));
//         zig_src.writer(b.allocator).print("    @embedFile(\"w{d}.bin\"),\n", .{i}) catch unreachable;
//     }
//     zig_src.writer(b.allocator).writeAll("};\n\n") catch unreachable;

//     zig_src.writer(b.allocator).writeAll("pub const biases = [_][]const u8{\n") catch unreachable;
//     for (1..layer_count + 1) |i| {
//         const b_src = dir.path(b, b.fmt("b{d}.bin", .{i}));
//         _ = wf.addCopyFile(b_src, b.fmt("b{d}.bin", .{i}));
//         zig_src.writer(b.allocator).print("    @embedFile(\"b{d}.bin\"),\n", .{i}) catch unreachable;
//     }
//     zig_src.writer(b.allocator).writeAll("};\n") catch unreachable;

//     const embeds = wf.add("zffnn_embeds.zig", zig_src.items);
//     target_mod.addAnonymousImport("embed_params", .{
//         .root_source_file = embeds,
//     });
// }
