pub const Dtype = enum {
    f32,
    f16,
    i8,

    pub const Kind = enum {
        float,
        signed_integer,
    };

    pub fn Scalar(comptime self: Dtype) type {
        return switch (self) {
            .f32 => f32,
            .f16 => f16,
            .i8 => i8,
        };
    }
    
    pub fn fromScalar(comptime T: type) Dtype {
        return if (T == f32)
            .f32
        else if (T == f16)
            .f16
        else if (T == i8)
            .i8
        else
            @compileError("unsupported tensor scalar type: " ++ @typeName(T));
    }

    pub fn kind(comptime dtype: Dtype) Kind {
        return switch (dtype) {
            .f32, .f16 => .float,
            .i8 => .signed_integer,
        };
    }

    pub fn Vector(comptime dtype: Dtype, comptime len: usize) type {
        return @Vector(len, dtype.Scalar());
    }

    pub fn zero(comptime dtype: Dtype) dtype.Scalar() {
        return 0;
    }

    pub fn vectorZero(comptime dtype: Dtype, comptime len: usize) dtype.Vector(len) {
        return @splat(dtype.zero());
    }

    pub fn byteSize(comptime dtype: Dtype) usize {
        return switch (dtype) {
            .f32 => @sizeOf(f32),
            .f16 => @sizeOf(f16),
            .i8 => @sizeOf(i8),
        };
    }

    pub fn alignment(comptime dtype: Dtype) usize {
        return switch (dtype) {
            .f32 => @alignOf(f32),
            .f16 => @alignOf(f16),
            .i8 => @alignOf(i8),
        };
    }
};