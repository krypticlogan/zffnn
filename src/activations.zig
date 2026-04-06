const std = @import("std");
pub const Activation = union(enum(u8)) {
    none,
    relu,
    sigmoid,
    softmax,

    pub fn apply(self: Activation, mat: anytype, batched: bool) @TypeOf(mat) {
        switch (self) {
            .none => return mat,
            .relu => return relu(mat),
            .sigmoid => return sigmoid(mat),
            .softmax => return softmax(mat, batched),
        }
    }
};

fn row_relu(v: anytype) @TypeOf(v) {
    const zero: @TypeOf(v) = @splat(0);
    return @select(f32, v > zero, v, zero);
}

pub fn relu(mat: anytype) @TypeOf(mat) {
    var out = mat.dupe_like(.zero);
    for (0..mat.rows()) |row| {
        out.data[row] = row_relu(mat.data[row]);
    }
    return out;
}

fn row_sigmoid(v: anytype) @TypeOf(v) {
    const one: @TypeOf(v) = @splat(1);
    return one / (one + @exp(-v));
}

pub fn sigmoid(mat: anytype) @TypeOf(mat) {
    var out = mat.dupe_like(.zero);
    for (0..mat.rows()) |row| {
        out.data[row] = row_sigmoid(mat.data[row]);
    }
    return out;
}

pub fn softmax(mat: anytype, batched: bool) @TypeOf(mat) {
    if (batched) {
        return batch_softmax(mat);
    } else {
        return single_softmax(mat);
    }
}

pub fn batch_softmax(mat: anytype) @TypeOf(mat) {
    var out = mat.dupe_like(.clone);
    
    var max_vec: @TypeOf(mat.data[0]) = @splat(std.math.floatMin(f32));
    for (mat.data, 0..) |row, r| { // Here, we subtract the max value from each element's row before exponentiating to avoid overflow
        max_vec = @select(f32, row > max_vec, row, max_vec);
        out.data[r] -= max_vec; 
    }
    
    const e_mat = out.exp();
    const sum_vec = e_mat.sum_cwise();
    inline for (0..mat.rows()) |i| {
        out.data[i] = e_mat.data[i] / sum_vec;
    }
    return out;
}
pub fn single_softmax(mat: anytype) @TypeOf(mat) {
    // We transpose the matrix immediately, so that we may compute softmax per column in, but treat them per row for SIMD purposes
    var out = mat.t();

    // Here, we subtract the max value from each element's row before exponentiating to avoid overflow
    for (0..out.rows()) |r| {
        const maxv = @reduce(.Max, out.data[r]);
        out.data[r] -= @as(@Vector(@TypeOf(out).m, f32), @splat(maxv));
    }

    // softmax per row (transposed)
    const e_mat = out.exp();
    const e_sum = e_mat.sum_rwise();
    for (0..out.rows()) |i| {
        const inv = 1.0 / e_sum[i];
        out.data[i] = e_mat.data[i] * @as(@Vector(@TypeOf(mat).n, f32), @splat(inv));
    }
    // transpose the output again to retain correct shape
    return out.t();
}
