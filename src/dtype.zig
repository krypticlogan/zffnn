pub const Dtype = enum {
    f32,
    f16,
    i8,

    pub fn Scalar(comptime self: Dtype) type {
        return switch (self) {
            .f32 => f32,
            .f16 => f16,
            .i8 => i8
        };
    }
};