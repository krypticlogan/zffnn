const std = @import("std");
const ffnn = @import("ffnn");
const NN = ffnn.NN;
const Mat = ffnn.Mat;
const MatOp = ffnn.MatOp;
const print = std.debug.print;
pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer {
            switch (gpa.deinit()) {
                .leak => print("Mem leaked\n", .{}),
                .ok => print("No mem leak detected\n", .{})

            }
        }
    const allocator = gpa.allocator();

    @setEvalBranchQuota(5000);
    // var mat = Mat(.{3, 2}).create(0);
    // // mat.init(0);
    // const mat_template: [3][2]f64 = .{ 
    //     .{1, 2},
    //     .{3, 4},
    //     .{5, 6} 
    // };
    // mat.load(mat_template);
    // mat.show();

    // var mat2 = Mat(.{2, 4}).create(0);
    // // mat2.init(0);
    // const mat_template2: [2][4]f64 = .{
    //     .{6, 5, 4, 3},
    //     .{2, 1, 0, -1},
    // };
    // mat2.load(mat_template2);

    // var vector = Mat(.{1, 5}).create(0);
    // const vector_template: [1][5]f64 = .{
    //     .{57, -23, 3, 45, 10}
    // };
    // vector.load(vector_template);
    // vector.show();
    // var softmax = vector.softmax();
    // softmax.show();

    // var sigmoid = vector.sigmoid();
    // sigmoid.show();

    // var relu = vector.relu();
    // relu.show();

    // var mul_res = mat.mul(mat2);
    // mul_res.show();
    // const e_mul = mul_res.exp();
    // e_mul.show();
    // var mat_t = mat.t();
    // mat_t.show();
    const entry_ct = 2;
    const feature_ct = 3;
    const data: [entry_ct][feature_ct]f64 = .{ 
        .{1, 2, 3},
        .{4, 5, 6},
    };

    const input_type = Mat(.{entry_ct, feature_ct});
    var input = input_type.create(0);
    input.load(data);

    // input.show();
    const def: []const struct { usize, MatOp} = &.{ 
        .{feature_ct, .relu}, 
        .{10, .relu}, 
        .{15, .sigmoid}, 
        .{2, .softmax}
    };
    const Net = NN(def, entry_ct, data);
    var nn = Net.new(allocator);

    // const sig_test: [1][2]f64 = .{
    //     .{ -4.76887372491392, -11.067189709820568 },
    // };

    // var mat = Mat(.{1, 2}).create(0);
    // mat.load(sig_test);
    // print("Sum: {any}\n", .{mat.sum()});

    // mat.show();
    // mat.sigmoid().show();
    // mat.show();



    nn.forward();

    nn.layers[0].z.show();
    nn.layers[0].a.show();

    nn.layers[3].z.show();
    nn.layers[3].a.show();



    // nn.view();
    nn.destroy();
}