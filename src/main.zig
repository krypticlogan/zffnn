const std = @import("std");
const ffnn = @import("ffnn");
const NN = ffnn.NN;
const Mat = ffnn.Mat;
const Activation = ffnn.Activation;
const print = std.debug.print;

pub fn main() void {
    @setEvalBranchQuota(2_000_000_000);

    const csi = "\x1b";
    const img = @embedFile("preprocessed_imgs/3/img.bin");
    const pixels = std.mem.bytesAsSlice(f32, img);

    const entry_ct = 1;
    const feature_ct = 784;
    var data: [entry_ct][feature_ct]f32 = .{ 
        .{0} ** feature_ct,
    };

    for (0..28) |row| {
        for (0..28) |col| {
            const val = pixels[col + row * 28];
            const color = 232 + @as(usize, @intFromFloat(val * 24.0)); // 232 - 255 are grayscale
            data[0][col + row * 28] = @floatCast(val);
            print(csi ++ "[48;5;{d}m  ", .{color});
        }
        print(csi ++ "[0m\n", .{});
    }

    
    

    const def: []const struct { usize, Activation} = &.{ 
        .{feature_ct, .none}, 
        .{128, .relu}, 
        .{64, .relu}, 
        .{10, .softmax}
    };
    const Net = NN(def, entry_ct);
    var nn = comptime Net.load_from_bin("model_params");
    const preds = nn.forward(data);
    preds.show();
    const preds_row = preds.t();
    const certainty = preds_row.max().get(0,0);

    print("Certainty: {any}\n", .{certainty});
    var guess: u8 = 0;
    while(guess < 10) : (guess+=1) {
        if (preds_row.get(0, guess) == certainty)
            break;
    } else unreachable;

    // var guess: u8 = undefined;
    // var certainty: f32 = 0.0;
    // for (0..10) |num| {
    //     if (preds.data[num][0] > certainty) {
    //         certainty = preds.data[num][0];
    //         guess = @intCast(num);
    //     }
    // }
    print("Guess: {d}", .{guess});
}