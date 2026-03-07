const std = @import("std");
const rl = @import("raylib");
const zffnn = @import("zffnn");

const print = std.debug.print;

const input_ct = 1;
const input_sz = 784;
const def: []const struct { usize, zffnn.Activation} = &.{ 
    .{input_sz, .relu}, 
    .{128, .relu}, 
    .{64, .relu}, 
    .{10, .softmax}
};
const Net = zffnn.NN(def, input_ct);

pub fn main() void {
    // load our predefined model
    var nn = comptime Net.load_from_bin("model_params");

    const canvasSZ = 784;
    const dashWidth = 40;
    const dashHeight = canvasSZ; 

    const screenWidth = canvasSZ + dashWidth;
    const screenHeight = canvasSZ;
    const res = 28; // resolution 28x28
    const cell_size = canvasSZ / res;

    rl.initWindow(screenWidth, screenHeight, "zig nn demo");
    defer rl.closeWindow();

    var canvas: [res][res]f32 = undefined;
    for (0..res) |y| {
        @memset(&canvas[y], 0.0);
    }

    rl.setTargetFPS(60); 
    while (!rl.windowShouldClose()) {
    // Update
        if (rl.isKeyPressed(.enter) or rl.isKeyPressed(.kp_enter)) {
            for (0..res) |y| { // clear canvas
                @memset(&canvas[y], 0.0);
            }
        }

        const mouse_pos = rl.getMousePosition();
        if ((mouse_pos.x > 0 and mouse_pos.y > 0) and (mouse_pos.x < canvasSZ and mouse_pos.y < canvasSZ)) {
            const cell_x: u8 = @intFromFloat(@divFloor(mouse_pos.x, cell_size));
            const cell_y: u8 = @intFromFloat(@divFloor(mouse_pos.y, cell_size));
            if (rl.isMouseButtonDown(.left)) {
                // fill the canvas where the mouse is pointing
                canvas[cell_y][cell_x]+=0.5;
                if (cell_y > 0) canvas[cell_y - 1][cell_x]+=0.3;
                if (cell_y < canvas.len - 1) canvas[cell_y + 1][cell_x]+=0.3;

                if (cell_x > 0) canvas[cell_y][cell_x - 1]+=0.3;
                if (cell_x < canvas.len - 1) canvas[cell_y][cell_x + 1]+=0.3;


                // make a prediction using the model on what number was drawn
                const model_input_ptr: *[input_sz]f32 = @ptrCast(&canvas);
                const model_input = model_input_ptr.*;
                const preds = nn.forward(.{model_input});
                preds.show(); // debug
                const preds_row = preds.t();
                const certainty = preds_row.max().get(0,0);
                var guess: u8 = 0;
                while(guess < 10) : (guess+=1) {
                    if (preds_row.get(0, guess) == certainty)
                        break;
                } else unreachable;
                print("Guess: {d}\n", .{guess});
            }
        }

    // Draw
        rl.beginDrawing();
        defer rl.endDrawing();

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
                rl.drawRectangle(@intCast(x * cell_size), @intCast(y * cell_size) , cell_size, cell_size, color);
            }
        }

        rl.drawRectangle(cell_size * canvasSZ, 0, dashWidth, dashHeight, .white);
    }
}

// fn mouseInCell()