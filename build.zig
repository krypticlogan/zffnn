const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseSmall });
    const options = b.addOptions();
    options.addOption(?std.Build.LazyPath, "params", null);
    const mod = b.addModule("zffnn", .{
        .root_source_file = b.path("src/ffnn.zig"),
        .target = target,
        .optimize= optimize
    });
    mod.addOptions("build_options", options);

    const mod_tests = b.addTest(.{
        .root_module = mod,
    });

    const run_mod_tests = b.addRunArtifact(mod_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
}
