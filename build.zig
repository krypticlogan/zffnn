const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});

    const optimize = b.standardOptimizeOption(.{});

    // const zopengl = b.dependency("zopengl", .{});
    // const zgui = b.dependency("zgui", .{
    //     .shared = false,
    //     .with_implot = true,
    //     .backend = .glfw_opengl3
    // });
    // const zglfw = b.dependency("zglfw", .{});

    const mod = b.addModule("zig_nn", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .imports = &.{
            // .{ .name = "zopengl", .module = zopengl.module("root")},
            // .{ .name = "zgui", .module = zgui.module("root")},
            // .{ .name = "zglfw", .module = zglfw.module("root")}
        }
    });
    // mod.addImport("zopengl", zopengl.module("root"));
    // if (target.result.os.tag != .emscripten) {
    //     mod.linkLibrary(zglfw.artifact("glfw"));
    // }
    // mod.linkLibrary(zgui.artifact("imgui"));

    const exe = b.addExecutable(.{
        .name = "ffnn",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "ffnn", .module = mod },
            },
        }),
    });
    b.installArtifact(exe);
    const run_step = b.step("run", "Run the app");

    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const mod_tests = b.addTest(.{
        .root_module = mod,
    });

    // A run step that will run the test executable.
    const run_mod_tests = b.addRunArtifact(mod_tests);
    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });
    const run_exe_tests = b.addRunArtifact(exe_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);
}
