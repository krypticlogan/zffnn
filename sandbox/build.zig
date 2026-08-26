const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zgc_dep = b.dependency("zgc", .{
        .target = target,
        .optimize = optimize,
    });
    const zgc_mod = zgc_dep.module("zgc");
    const model_params_mod = b.createModule(.{
        .root_source_file = b.path("model_params/params.zig"),
        .target = target,
        .optimize = optimize,
    });
    const raylib_dep = b.dependency("raylib_zig", .{
        .target = target,
        .optimize = optimize,
    });
    const raylib_mod = raylib_dep.module("raylib");
    const raylib_artifact = raylib_dep.artifact("raylib");

    const exe = b.addExecutable(.{
        .name = "zgc-sandbox",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zgc", .module = zgc_mod },
                .{ .name = "model_params", .module = model_params_mod },
            },
        }),
    });
    b.installArtifact(exe);

    const run_step = b.step("run", "Build and run the zgc sandbox");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const demo_exe = b.addExecutable(.{
        .name = "demo",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/demo.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zgc", .module = zgc_mod },
                .{ .name = "model_params", .module = model_params_mod },
                .{ .name = "raylib", .module = raylib_mod },
            },
            .link_libc = true,
        }),
    });
    demo_exe.root_module.linkLibrary(raylib_artifact);
    b.installArtifact(demo_exe);

    const demo_step = b.step("demo", "Run the interactive digit-classification demo");
    const demo_cmd = b.addRunArtifact(demo_exe);
    demo_step.dependOn(&demo_cmd.step);

    const inspect_exe = b.addExecutable(.{
        .name = "zgc-model",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/inspect.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zgc", .module = zgc_mod },
                .{ .name = "model_params", .module = model_params_mod },
            },
        }),
    });
    inspect_exe.forceUndefinedSymbol(if (target.result.os.tag == .macos)
        "_zgc_run_model"
    else
        "zgc_run_model");

    const install_inspect = b.addInstallArtifact(inspect_exe, .{});
    const build_model_step = b.step("build-model", "Build the lean model inspection binary");
    build_model_step.dependOn(&install_inspect.step);

    const run_model_cmd = b.addRunArtifact(inspect_exe);
    const run_model_step = b.step("run-model", "Run the lean model binary without instrumentation");
    run_model_step.dependOn(&run_model_cmd.step);

    const disassembler = if (target.result.os.tag == .macos)
        b.addSystemCommand(&.{ "xcrun", "llvm-objdump", "--disassemble-symbols=_zgc_run_model" })
    else
        b.addSystemCommand(&.{ "objdump", "--disassemble=zgc_run_model" });
    disassembler.addArtifactArg(inspect_exe);
    const inspect_model_step = b.step("inspect-model", "Disassemble the lean model execution symbol");
    inspect_model_step.dependOn(&disassembler.step);
}
