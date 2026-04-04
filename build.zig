const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    // library
    const mod = b.addModule("zffnn", .{ 
        .root_source_file = b.path("src/root.zig"), 
        .target = target,
        .optimize = optimize,
    });
    
    // embed params generator
    const gen = b.addExecutable(.{
        .name = "embed_helper",
        .root_module = b.createModule(.{
            .root_source_file = b.path("embed_helper/embeddings_gen.zig"),
            .target = b.graph.host,
            .optimize = optimize,
        }),
    });
    b.installArtifact(gen);

    // tests
    const mod_tests = b.addTest(.{
        .root_module = mod,
    });
    const run_mod_tests = b.addRunArtifact(mod_tests);
    const test_step = b.step("test", "Run tests");

    test_step.dependOn(&run_mod_tests.step);
}
