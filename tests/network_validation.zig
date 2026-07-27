const std = @import("std");
const testing = std.testing;

const zffnn = @import("zffnn");
const network_validation = zffnn.network_validation;

const valid_definition: []const struct { usize, zffnn.Activation } = &.{
    .{ 2, .none },
    .{ 3, .relu },
    .{ 1, .none },
};

test "accepts a valid network definition" {
    try testing.expectEqual(
        @as(?network_validation.DefinitionError, null),
        network_validation.check(valid_definition, 4),
    );

    const Net = zffnn.NN(valid_definition, 4);
    const network = Net.new();
    try testing.expectEqual(@as(usize, 6), network.num_nodes);
}

test "rejects definitions with fewer than two layers" {
    const empty: []const struct { usize, zffnn.Activation } = &.{};
    const input_only: []const struct { usize, zffnn.Activation } = &.{
        .{ 2, .none },
    };

    try testing.expectEqual(
        network_validation.DefinitionError.too_few_layers,
        network_validation.check(empty, 1).?,
    );
    try testing.expectEqual(
        network_validation.DefinitionError.too_few_layers,
        network_validation.check(input_only, 1).?,
    );
}

test "rejects a zero batch size" {
    try testing.expectEqual(
        network_validation.DefinitionError.zero_batch_size,
        network_validation.check(valid_definition, 0).?,
    );
}

test "rejects zero-width layers" {
    const zero_input: []const struct { usize, zffnn.Activation } = &.{
        .{ 0, .none },
        .{ 1, .none },
    };
    const zero_hidden: []const struct { usize, zffnn.Activation } = &.{
        .{ 2, .none },
        .{ 0, .relu },
        .{ 1, .none },
    };
    const zero_output: []const struct { usize, zffnn.Activation } = &.{
        .{ 2, .none },
        .{ 0, .none },
    };

    inline for (&.{ zero_input, zero_hidden, zero_output }) |definition| {
        try testing.expectEqual(
            network_validation.DefinitionError.zero_width,
            network_validation.check(definition, 1).?,
        );
    }
}

test "rejects an activated input layer" {
    inline for (&.{ zffnn.Activation.relu, .sigmoid, .softmax }) |activation| {
        const definition: []const struct { usize, zffnn.Activation } = &.{
            .{ 2, activation },
            .{ 1, .none },
        };
        try testing.expectEqual(
            network_validation.DefinitionError.input_activation_must_be_none,
            network_validation.check(definition, 1).?,
        );
    }
}
