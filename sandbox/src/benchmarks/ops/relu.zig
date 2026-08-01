const std = @import("std");
const zgc = @import("zgc");

const element_count = 16 * 1024;

pub const Benchmark = struct {
    pub const name = "relu/f32/16k";
    pub const work_items_per_invocation: f64 = element_count;
    pub const bytes_per_invocation: f64 = element_count * @sizeOf(f32) * 2;

    input_data: [element_count]f32,
    output_data: [element_count]f32,

    pub fn init() Benchmark {
        var input_data: [element_count]f32 = undefined;
        for (&input_data, 0..) |*value, index| {
            const magnitude: f32 = @floatFromInt(index % 127);
            value.* = if (index % 2 == 0) magnitude else -magnitude;
        }
        return .{
            .input_data = input_data,
            .output_data = @splat(0),
        };
    }

    pub fn run(self: *Benchmark, iterations: usize) void {
        const input: zgc.Tensor.View(f32, 1) = .{
            .data = &self.input_data,
            .shape = .{element_count},
        };
        const output: zgc.Tensor.View(f32, 1) = .{
            .data = &self.output_data,
            .shape = .{element_count},
        };
        const op: zgc.Op = .relu;
        for (0..iterations) |_| {
            op.execute(.{input}, output);
            std.mem.doNotOptimizeAway(&self.output_data);
        }
    }
};
