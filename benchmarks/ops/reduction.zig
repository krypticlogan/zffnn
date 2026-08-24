const std = @import("std");
const zgc = @import("zgc");

const rows = 256;
const columns = 256;
const element_count = rows * columns;
const Layout = enum { contiguous, axis_strided };

fn ReductionBenchmark(
    comptime layout: Layout,
    comptime operation: enum { sum, softmax },
    comptime benchmark_name: []const u8,
) type {
    return struct {
        const Self = @This();

        pub const name = benchmark_name;
        pub const default_iterations = if (operation == .sum) 1_000 else 200;
        pub const default_warmup_iterations = if (operation == .sum) 100 else 20;
        pub const work_items_per_invocation: f64 = element_count;
        pub const work_unit = "elements";
        pub const bytes_per_invocation: f64 = if (operation == .sum)
            @sizeOf(f32) * (element_count + rows)
        else
            @sizeOf(f32) * element_count * 2;

        input_storage: [element_count]f32,
        output_storage: if (operation == .sum) [rows]f32 else [element_count]f32,

        pub fn init() Self {
            var result: Self = undefined;
            for (&result.input_storage, 0..) |*value, index| {
                value.* = @as(f32, @floatFromInt(index % 31)) / 10.0 - 1.5;
            }
            result.output_storage = @splat(0);
            return result;
        }

        pub fn run(self: *Self, iterations: usize) void {
            const input: zgc.Tensor.ConstView(f32, 2) = .{
                .storage = &self.input_storage,
                .shape = .{ rows, columns },
                .strides = if (layout == .contiguous) .{ columns, 1 } else .{ 1, rows },
                .offset = 0,
            };
            if (comptime operation == .sum) {
                const output: zgc.Tensor.View(f32, 1) = .{
                    .storage = &self.output_storage,
                    .shape = .{rows},
                    .strides = .{1},
                    .offset = 0,
                };
                const op: zgc.Op = .{ .compute = .{ .sum = .{ .axis = 1 } } };
                for (0..iterations) |_| {
                    op.execute(.{input}, output);
                    std.mem.doNotOptimizeAway(&self.output_storage);
                }
            } else {
                const output: zgc.Tensor.View(f32, 2) = .{
                    .storage = &self.output_storage,
                    .shape = .{ rows, columns },
                    .strides = .{ columns, 1 },
                    .offset = 0,
                };
                const op: zgc.Op = .{ .compute = .{ .softmax = .{ .axis = 1 } } };
                for (0..iterations) |_| {
                    op.execute(.{input}, output);
                    std.mem.doNotOptimizeAway(&self.output_storage);
                }
            }
        }
    };
}

pub const SumContiguous = ReductionBenchmark(.contiguous, .sum, "sum/f32/256x256/axis1-contiguous");
pub const SumStrided = ReductionBenchmark(.axis_strided, .sum, "sum/f32/256x256/axis1-strided");
pub const SoftmaxContiguous = ReductionBenchmark(.contiguous, .softmax, "softmax/f32/256x256/axis1-contiguous");
pub const SoftmaxStrided = ReductionBenchmark(.axis_strided, .softmax, "softmax/f32/256x256/axis1-strided");
