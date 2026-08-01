const std = @import("std");
const sandbox_model = @import("model.zig");

pub fn main() void {
    var model = sandbox_model.Model.init();

    std.debug.print("== Counting pass ==\n", .{});
    sandbox_model.capacity.debugPrint();

    std.debug.print("\n== Materialized graph ==\n", .{});
    sandbox_model.graph.debugPrint();
    sandbox_model.graph.debugPrintStructure();

    std.debug.print("\n== Memory plan ==\n", .{});
    sandbox_model.Model.memory_plan.debugPrint();

    sandbox_model.loadInput(&model);
    std.debug.print("\n== Memory before execution ==\n", .{});
    model.debugPrintMemory(sandbox_model.Model.memory_plan.byte_count);

    model.run();

    const output = model.outputView(0);
    std.debug.print("\n== Output ==\nshape={any}\ndata={any}\n", .{
        output.shape,
        output.data,
    });

    std.debug.print("\n== Memory after execution ==\n", .{});
    model.debugPrintMemory(sandbox_model.Model.memory_plan.byte_count);
}
