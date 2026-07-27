const Activation = @import("activations.zig").Activation;

pub const DefinitionError = enum {
    too_few_layers,
    zero_batch_size,
    zero_width,
    input_activation_must_be_none,
};

pub fn check(
    comptime def: []const struct { usize, Activation },
    comptime batch_size: usize,
) ?DefinitionError {
    if (def.len < 2) return .too_few_layers;
    if (batch_size == 0) return .zero_batch_size;
    if (def[0][1] != .none) return .input_activation_must_be_none;

    for (def) |layer| {
        if (layer[0] == 0) return .zero_width;
    }

    return null;
}

pub fn assertValid(
    comptime def: []const struct { usize, Activation },
    comptime batch_size: usize,
) void {
    const validation_error = check(def, batch_size) orelse return;
    switch (validation_error) {
        .too_few_layers => @compileError("A network definition must contain at least an input and output layer"),
        .zero_batch_size => @compileError("A network batch size must be greater than zero"),
        .zero_width => @compileError("Every network layer must contain at least one node"),
        .input_activation_must_be_none => @compileError("The input layer activation must be .none"),
    }
}
