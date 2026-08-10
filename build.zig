const std = @import("std");
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
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

    // library
    const zgc_mod = b.addModule("zgc", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // tests
    const test_embed_params = b.createModule(.{
        .root_source_file = b.path("tests/fixtures/embed_params.zig"),
        .target = target,
        .optimize = optimize,
    });
    const test_root_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_root_mod.addImport("embed_params", test_embed_params);

    const test_mod = b.addModule("zgc_tests", .{
        .root_source_file = b.path("tests/tests.zig"),
        .target = target,
        .optimize = optimize,
    });

    const embedding_codegen_mod = b.createModule(.{
        .root_source_file = b.path("embed_helper/embedding.zig"),
        .target = target,
        .optimize = optimize,
    });

    test_mod.addImport("zgc", test_root_mod);
    test_mod.addImport("embedding_codegen", embedding_codegen_mod);

    const tests = b.addTest(.{
        .root_module = test_mod,
    });

    const check_step = b.step("check", "Compile tests without running them");
    check_step.dependOn(&tests.step);

    const run_mod_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);

    // benchmarks
    const benchmark_op = b.option([]const u8, "op", "Tensor operation to benchmark") orelse "all";
    const benchmark_iterations = b.option(usize, "iterations", "Operation invocations per timed run (0 selects the case default)") orelse 0;
    const benchmark_runs = b.option(usize, "runs", "Number of timed runs") orelse 10;
    const benchmark_warmup = b.option(usize, "warmup_iterations", "Untimed warmup invocations (0 selects the case default)") orelse 0;

    const benchmark_options = b.addOptions();
    benchmark_options.addOption([]const u8, "op", benchmark_op);
    benchmark_options.addOption(usize, "iterations", benchmark_iterations);
    benchmark_options.addOption(usize, "runs", benchmark_runs);
    benchmark_options.addOption(usize, "warmup_iterations", benchmark_warmup);

    const benchmark_exe = b.addExecutable(.{
        .name = "zgc-benchmark",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zgc", .module = zgc_mod },
                .{ .name = "build_options", .module = benchmark_options.createModule() },
            },
        }),
    });
    const run_benchmark = b.addRunArtifact(benchmark_exe);
    const benchmark_step = b.step("benchmark", "Benchmark the operation selected by -Dop");
    benchmark_step.dependOn(&run_benchmark.step);
}
