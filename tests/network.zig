const std = @import("std");
const testing = std.testing;

const zffnn = @import("zffnn");
const expect_mat_approx_equal = @import("test_helpers.zig").expect_mat_approx_equal;

const linear_def: []const struct { usize, zffnn.Activation } = &.{
    .{ 2, .none },
    .{ 2, .none },
};

const softmax_def: []const struct { usize, zffnn.Activation } = &.{
    .{ 2, .none },
    .{ 2, .softmax },
};

fn set_known_parameters(nn: anytype) void {
    nn.layers[1].weights.load(.{
        .{ 2, -1 },
        .{ 0.5, 3 },
    });
    nn.layers[1].bias.load(.{
        .{1},
        .{-2},
    });
}

test "single-sample inference matches known linear output" {
    var nn = zffnn.NN(linear_def, 1).new();
    set_known_parameters(&nn);

    var input = zffnn.Mat(1, 2).create(0);
    input.load(.{
        .{ 4, 5 },
    });
    var expected = zffnn.Mat(2, 1).create(0);
    expected.load(.{
        .{4},
        .{15},
    });

    const actual = nn.forward(input);

    try testing.expectEqualDeep(expected.data, actual.data);
}

test "multi-sample inference matches known linear output" {
    var nn = zffnn.NN(linear_def, 3).new();
    set_known_parameters(&nn);

    var input = zffnn.Mat(3, 2).create(0);
    input.load(.{
        .{ 4, 5 },
        .{ -2, 1 },
        .{ 0, -3 },
    });
    var expected = zffnn.Mat(2, 3).create(0);
    expected.load(.{
        .{ 4, -4, 4 },
        .{ 15, 0, -11 },
    });

    const actual = nn.forward(input);

    try testing.expectEqualDeep(expected.data, actual.data);
}

test "forward and forward_ produce the same output" {
    var nn = zffnn.NN(linear_def, 3).new();
    set_known_parameters(&nn);

    var input = zffnn.Mat(3, 2).create(0);
    input.load(.{
        .{ 4, 5 },
        .{ -2, 1 },
        .{ 0, -3 },
    });

    const returning_output = nn.forward(input);
    var in_place_output = zffnn.Mat(2, 3).create(std.math.nan(f32));
    nn.forward_(input, &in_place_output);

    try testing.expectEqualDeep(returning_output.data, in_place_output.data);
}

test "network applies batched softmax per sample" {
    var nn = zffnn.NN(softmax_def, 3).new();
    set_known_parameters(&nn);

    var input = zffnn.Mat(3, 2).create(0);
    input.load(.{
        .{ 4, 5 },
        .{ -2, 1 },
        .{ 0, -3 },
    });

    const actual = nn.forward(input);
    const sums: [3]f32 = actual.sum_cwise();

    for (sums) |sum| {
        try testing.expectApproxEqAbs(@as(f32, 1), sum, 1e-6);
    }
    for (0..actual.rows()) |row| {
        for (0..actual.cols()) |col| {
            try testing.expect(std.math.isFinite(actual.get(row, col)));
        }
    }
}

test "random initialization is deterministic by seed" {
    const def: []const struct { usize, zffnn.Activation } = &.{
        .{ 3, .none },
        .{ 4, .relu },
        .{ 2, .softmax },
    };
    const Net = zffnn.NN(def, 2);

    var first = Net.new();
    first.random_init(103);
    var second = Net.new();
    second.random_init(103);
    var different = Net.new();
    different.random_init(104);

    try testing.expectEqualDeep(first.layers[1].weights.data, second.layers[1].weights.data);
    try testing.expectEqualDeep(first.layers[1].bias.data, second.layers[1].bias.data);
    try testing.expect(!std.meta.eql(first.layers[1].weights.data, different.layers[1].weights.data));

    var input = zffnn.Mat(2, 3).create(0);
    input.load(.{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    const first_output = first.forward(input);
    const second_output = second.forward(input);
    try expect_mat_approx_equal(first_output, second_output, 0);
}
