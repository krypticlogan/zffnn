const std = @import("std");
const rl = @import("raylib");
const zffnn = @import("zffnn");

const Mat = zffnn.Mat;
const print = std.debug.print;

const input_ct = 1;
const input_sz = 784;
const def: []const struct { usize, zffnn.Activation } = &.{
    .{ input_sz, .none },
    .{ 128, .relu },
    .{ 64, .relu },
    .{ 10, .softmax },
};
const Net = zffnn.NN(def, input_ct);

pub fn main() void {
    // load our predefined model here under comptime scope
    var nn = comptime Net.load_from_embeds();

    const screenWidth = 1000;
    const screenHeight = 600;

    const canvasSZ = @divFloor(screenWidth * 2, 3); // drawing area
    const res = 28; // resolution 28x28
    const cell_size = canvasSZ / res;

    const dashWidth = screenWidth - canvasSZ + cell_size; // area for predictions and info
    const dashHeight = screenHeight;

    const predictionBarWidth: u16 = @divFloor(dashWidth * 6, 10);
    const predictionBarHeight = 20;
    rl.setTraceLogLevel(.err);
    rl.initWindow(screenWidth, screenHeight, "zig nn demo");
    defer rl.closeWindow();

    // persistents used for the model
    var canvas: [res][res]f32 = undefined;
    for (0..res) |y| {
        @memset(&canvas[y], 0.0);
    }
    var model_input = Mat(1, input_sz).create(0);
    var model_preds = Mat(10, 1).create(0);

    rl.setTargetFPS(120);
    while (!rl.windowShouldClose()) {
        // Update
        if (rl.isKeyPressed(.enter) or rl.isKeyPressed(.kp_enter)) {
            for (0..res) |y| { // clear canvas
                @memset(&canvas[y], 0.0);
                // model_preds.clear()
                model_preds = Mat(10, 1).create(0);
            }
        }

        const mouse_pos = rl.getMousePosition();
        if ((mouse_pos.x > 0 and mouse_pos.y > 0) and (mouse_pos.x < canvasSZ - cell_size and mouse_pos.y < canvasSZ - cell_size)) {
            const cell_x: u8 = @intFromFloat(@divFloor(mouse_pos.x, cell_size));
            const cell_y: u8 = @intFromFloat(@divFloor(mouse_pos.y, cell_size));
            if (rl.isMouseButtonDown(.left)) {
                // fill the canvas where the mouse is pointing
                canvas[cell_y][cell_x] = @min(canvas[cell_y][cell_x] + 0.3, 1);
                if (cell_y > 0) canvas[cell_y - 1][cell_x] = @min(canvas[cell_y - 1][cell_x] + 0.1, 1);
                if (cell_x > 0) canvas[cell_y][cell_x - 1] = @min(canvas[cell_y][cell_x - 1] + 0.1, 1);
                if (cell_y < canvas.len - 1) canvas[cell_y + 1][cell_x] = @min(canvas[cell_y + 1][cell_x] + 0.1, 1);
                if (cell_x < canvas.len - 1) canvas[cell_y][cell_x + 1] = @min(canvas[cell_y][cell_x + 1] + 0.1, 1);

                // make a prediction using the model on what number was drawn
                const model_input_ptr: *[input_sz]f32 = @ptrCast(&canvas);
                model_input.load(.{model_input_ptr.*});
                model_preds = nn.forward(model_input);
                model_preds.show(); // debug

                const certainty = model_preds.max_cwise()[0]; // index 0 since there is only one column
                var guess: u8 = 0;
                while (guess < 10) : (guess += 1) {
                    if (model_preds.get(guess, 0) == certainty)
                        break;
                } else unreachable;
                print("Guess: {d}\n", .{guess});
            }
        }

        // Draw
        rl.beginDrawing();
        defer rl.endDrawing();

        rl.clearBackground(.white);

        for (0..res) |y| {
            for (0..res) |x| {
                const val = canvas[y][x];
                const color = col: {
                    if (val <= 0.2) break :col rl.Color.black;
                    if (val <= 0.4) break :col rl.Color.dark_gray;
                    if (val <= 0.6) break :col rl.Color.gray;
                    if (val <= 0.8) break :col rl.Color.light_gray;
                    break :col rl.Color.white;
                };
                rl.drawRectangle(@intCast(x * cell_size), @intCast(y * cell_size), cell_size, cell_size, color);
            }
        }
        const dashStartX = canvasSZ - cell_size;
        rl.drawRectangle(dashStartX, 0, dashWidth, dashHeight, .dark_brown);

        const predictionBarsTop = 25;
        const predictionBarX = dashStartX + 50;
        var dig_buf: [2]u8 = undefined;
        const certainty_fmt = "{d:0>5.2}%";
        var certainty_buf: [6]u8 = undefined; // may be up to 5 chars long, plus the null termination
        for (0..10) |guess| { // prediction bars
            const predictionBarY: u16 = predictionBarsTop + (predictionBarHeight + 25) * @as(u16, @intCast(guess));
            rl.drawRectangleRounded(
                rl.Rectangle{ .height = predictionBarHeight + 5, .width = predictionBarWidth, .x = predictionBarX, .y = @floatFromInt(predictionBarY) },
                0.8,
                3,
                .dark_gray,
            );

            const certainty = model_preds.get(guess, 0);
            // strings for rendering
            var dig_string = std.fmt.bufPrint(&dig_buf, "{d} ", .{guess}) catch unreachable;
            var certainty_string = std.fmt.bufPrint(&certainty_buf, certainty_fmt, .{certainty}) catch unreachable;
            // set the null terminator
            dig_string[1] = 0;
            certainty_string[5] = 0;
            rl.drawText(dig_string[0..1 :0], predictionBarX - 30, predictionBarY, 32, .white);
            rl.drawText(certainty_string[0..5 :0], predictionBarX + predictionBarWidth + 15, predictionBarY, 28, .white);

            // prediction fill
            const fill_width: f32 = certainty * predictionBarWidth;
            rl.drawRectangleRounded(
                rl.Rectangle{ .height = predictionBarHeight + 5, .width = fill_width, .x = predictionBarX, .y = @floatFromInt(predictionBarY) },
                0.8,
                3,
                .lime,
            );
        }
    }
}
