const std = @import("std");
const testing = std.testing;
const expect = testing.expect;

const zffnn = @import("zffnn");
const mat_equal = @import("tests.zig").mat_equal;

test "inference" {
    const def: []const struct { usize, zffnn.Activation } = &.{
      .{3, .none},
      .{10, .relu},
      .{30, .sigmoid},
      .{2, .softmax}
    };
    var nn = zffnn.NN(def, 1).new();
    nn.random_init(103);
    
    var input = zffnn.Mat(1, 3).create(0);
    input.load([_][3]f32{
        .{ 1, 2, 3 }
    });
    
    _ = nn.forward(input);
    try expect(true);
}

test "determinism" {
    const def: []const struct { usize, zffnn.Activation } = &.{
      .{3, .none},
      .{10, .relu},
      .{30, .sigmoid},
      .{2, .softmax}
    };
    const nn_type = zffnn.NN(def, 1);
    const seed = 103;
    
    var nn = nn_type.new();
    nn.random_init(seed);
    
    var nn2 = nn_type.new();
    nn2.random_init(seed);
    
    var input = zffnn.Mat(1, 3).create(0);
    input.load([_][3]f32{
        .{ 1, 2, 3 }
    });
    
    const output = nn.forward(input);
    const output2 = nn2.forward(input);
    try expect(mat_equal(output, output2));    
}