const std = @import("std");
pub const Activation = union(enum(u8)) {
    none,
    relu,
    sigmoid,
    softmax,

    pub fn apply(self: Activation, mat: anytype, batched: bool) void {
        switch (self) {
            .none => return,
            .relu => relu(mat),
            .sigmoid => sigmoid(mat),
            .softmax => softmax(mat, batched),
        }
    }
};

fn row_relu(v: anytype) void {
    const zero: @TypeOf(v.*) = @splat(0);
    v.* = @select(f32, v.* > zero, v.*, zero);
}

pub fn relu(mat: anytype) void {
    for (0..mat.rows()) |row| {
        row_relu(&mat.data[row]);
    }
}

fn row_sigmoid(v: anytype) void {
    const one: @TypeOf(v.*) = @splat(1);
    v.* = one / (one + @exp(-v.*));
}

pub fn sigmoid(mat: anytype) void {
    for (0..mat.rows()) |row| {
        row_sigmoid(&mat.data[row]);
    }
}

pub fn softmax(mat: anytype, batched: bool) void {
    if (batched) {
        batch_softmax(mat);
    } else {
        single_softmax(mat);
    }
}

pub fn batch_softmax(mat: anytype) void {
    var temp = mat.dupe_like(.clone);
    
    // var max_vec: @TypeOf(temp.data[0]) = @splat(std.math.floatMin(f32));
    const max_vec = temp.max_cwise();
    for (0..temp.rows()) |r| { // Here, we subtract the max value from each element's row before exponentiating to avoid overflow
        temp.data[r] -= max_vec; 
    }
    
    const e_mat = temp.exp();
    const sum_vec = e_mat.sum_cwise();
    for (0..temp.rows()) |i| {
        mat.data[i] = e_mat.data[i] / sum_vec;
    }
}
pub fn single_softmax(mat: anytype) void {
    // We transpose the matrix immediately, so that we may compute softmax per column in, but treat them per row for SIMD purposes
    var temp = mat.t();

    // Here, we subtract the max value from each element's row before exponentiating to avoid overflow
    for (0..temp.rows()) |r| {
        const maxv = @reduce(.Max, temp.data[r]);
        temp.data[r] -= @as(@Vector(@TypeOf(temp).m, f32), @splat(maxv));
    }

    // softmax per row (transposed)
    const e_mat = temp.exp();
    const e_sum = e_mat.sum_rwise();
    for (0..temp.rows()) |i| {
        const inv = 1.0 / e_sum[i];
        temp.data[i] = e_mat.data[i] * @as(@Vector(@TypeOf(mat.*).n, f32), @splat(inv));
    }
    // transpose the output again to retain correct shape
    mat.* = temp.t();
}
