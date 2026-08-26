const sandbox_model = @import("model.zig");

fn zgc_run_model(model: *sandbox_model.Model) callconv(.c) void {
    model.run();
}

comptime {
    @export(&zgc_run_model, .{ .name = "zgc_run_model" });
}

pub fn main() void {
    var model = sandbox_model.Model.init();
    const input = sandbox_model.exampleInput();
    sandbox_model.bindInput(&model, &input);
    zgc_run_model(&model);
    sandbox_model.keepOutputAlive(&model);
}
