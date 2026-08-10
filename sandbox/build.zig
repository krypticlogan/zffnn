const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zgc_dep = b.dependency("zgc", .{
        .target = target,
        .optimize = optimize,
    });
    const zgc_mod = zgc_dep.module("zgc");

    const exe = b.addExecutable(.{
        .name = "zgc-sandbox",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zgc", .module = zgc_mod },
            },
        }),
    });
    b.installArtifact(exe);

    const run_step = b.step("run", "Build and run the zgc sandbox");
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_step.dependOn(&run_cmd.step);

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const inspect_exe = b.addExecutable(.{
        .name = "zgc-model",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/inspect.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zgc", .module = zgc_mod },
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
