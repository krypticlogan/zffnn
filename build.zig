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
    const root_mod = b.addModule("zffnn", .{
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

    const test_mod = b.addModule("zffnn_tests", .{
        .root_source_file = b.path("tests/tests.zig"),
        .target = target,
        .optimize = optimize,
    });

    const embedding_codegen_mod = b.createModule(.{
        .root_source_file = b.path("embed_helper/embedding.zig"),
        .target = target,
        .optimize = optimize,
    });

    test_mod.addImport("zffnn", test_root_mod);
    test_mod.addImport("embedding_codegen", embedding_codegen_mod);

    const tests = b.addTest(.{
        .root_module = test_mod,
    });
    const run_mod_tests = b.addRunArtifact(tests);

    const test_step = b.step("test", "Run tests");

    test_step.dependOn(&run_mod_tests.step);

    // benchmarks
    //
    const benchmark_cli = b.option([]const u8, "benchmark", "Benchmark to run: inference|batch_sweep|ops") orelse "inference";
    const model_cli = b.option([]const u8, "model", "Model size: small|medium|large") orelse "small";
    const batch_size_cli = b.option(usize, "batch_size", "Batch size") orelse 1;
    const seed_cli = b.option(usize, "seed", "PRNG seed") orelse 1234;
    const write_out_cli = b.option(bool, "write_out", "Write CSV output") orelse false;

    const benchmark_opts = b.addOptions();
    benchmark_opts.addOption([]const u8, "benchmark", benchmark_cli);
    benchmark_opts.addOption([]const u8, "model", model_cli);
    benchmark_opts.addOption(usize, "batch_size", batch_size_cli);
    benchmark_opts.addOption(usize, "seed", seed_cli);
    benchmark_opts.addOption(bool, "write_out", write_out_cli);

    const opts = .{ .target = target, .optimize = optimize };
    const zbench_module = b.dependency("zbench", opts).module("zbench");

    const benchmark_mod = b.addModule("zffnn_benchmarks", .{
        .root_source_file = b.path("benchmarks/benchmark.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zbench", .module = zbench_module },
        },
    });

    benchmark_mod.addImport("zbench", zbench_module);
    benchmark_mod.addImport("zffnn", root_mod);
    benchmark_mod.addImport("build_options", benchmark_opts.createModule());
    const benchmark = b.addExecutable(.{
        .name = "benchmark",
        .root_module = benchmark_mod,
    });
    // benchmark.linkLibrary(ztracy_dep.artifact("tracy"));

    b.installArtifact(benchmark);

    const run_benchmark = b.addRunArtifact(benchmark);
    const benchmark_step = b.step("benchmark", "Run benchmarks");
    benchmark_step.dependOn(&run_benchmark.step);
}
