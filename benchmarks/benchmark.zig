const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

const optimize = builtin.mode;

const zgc = @import("zgc");

const Benchmark = enum {
    inference,
    ops,
};

const which: Benchmark = blk: {
    if (std.mem.eql(u8, build_options.benchmark, "inference")) {
        break :blk .inference;
    } else if (std.mem.eql(u8, build_options.benchmark, "ops")) {
        break :blk .ops;
    } else {
        @compileError("Unknown benchmark" ++ build_options.benchmark);
    }
};

const seed = build_options.seed;
const write_out = build_options.write_out;

const zbench = @import("zbench");
const InferenceBench = struct {
    
    pub fn init() InferenceBench {
       var prng: std.Random.Xoshiro256 = .init(seed);
       _ = &prng;
       return InferenceBench {};
    }
  
    pub fn run(self: *InferenceBench, _: std.mem.Allocator) void {  
        _ = self;
    }
};

const MatMulBench = struct {
    pub fn init(batched: bool) MatMulBench {
        _ = batched;
       var prng: std.Random.Xoshiro256 = .init(seed);
       _ = &prng;
       return .{};
    }
  
    pub fn run(self: *MatMulBench, _: std.mem.Allocator) void {  
        _ = self;
    }
};

pub fn main(init: std.process.Init) !void {
    const io = init.io; 
    const stdout: std.Io.File = .stdout();
    
    var bench = zbench.Benchmark.init(init.gpa, .{});
    defer bench.deinit();

    std.debug.print("Benchmarks | OPTIMIZE={s}\n", .{@tagName(optimize)});

    try bench.addParam("1k inference", &InferenceBench.init(), .{
        .time_budget_ns = 5 * std.time.ns_per_s
    });
    try bench.addParam("1k MatMul (no batch)", &MatMulBench.init(false), .{
        .time_budget_ns = 5 * std.time.ns_per_s
    });
    try bench.addParam("1k MatMul (batched)", &MatMulBench.init(true), .{
        .time_budget_ns = 5 * std.time.ns_per_s
    });
    try bench.run(io, stdout);
}