pub const Activation = union(enum(u8)) {
    none,
    relu,
    sigmoid,
    softmax,
    
    pub fn apply(self: Activation, mat: anytype) @TypeOf(mat) {
        switch (self) {
            .none => return mat,
            .relu => return relu(mat),
            .sigmoid => return sigmoid(mat),
            .softmax => return softmax(mat),
        }
    }
};

pub fn relu(mat: anytype) @TypeOf(mat) {
    var out = mat.dupe_like();

    for (0..mat.rows()) |row| {
        for (0..mat.cols()) |col| {
            out.set(row, col, @max(mat.get(row, col), 0));
        }
    }
    return out;
}

pub fn sigmoid(mat: anytype) @TypeOf(mat) {
    var out = mat.dupe_like();
    for (0..mat.rows()) |row| {
        for (0..mat.cols()) |col| {
            out.set(row, col, 1 / (1 + @exp(-mat.get(row, col))));
        }
    }
    return out;
}

pub fn softmax(mat: anytype) @TypeOf(mat) {
    // We transpose the matrix immediately, so that we may compute softmax per column in, but treat them per row for SIMD purposes
    var out = mat.t();

    // Here, we subtract the max value from each element's row before exponentiating to avoid overflow
    for (0..out.rows()) |r| {
        const maxv = @reduce(.Max, out.data[r]);
        out.data[r] -= @as(@Vector(@TypeOf(out).m, f32), @splat(maxv));
    }

    // softmax per row (transposed)
    const e_mat = out.exp();
    const e_sum = e_mat.sum();
    for (0..out.rows()) |i| {
        const inv = 1.0 / e_sum.get(i, 0);
        out.data[i] = e_mat.data[i] * @as(@Vector(@TypeOf(mat).n, f32), @splat(inv));
    }
    // transpose the output again to retain correct shape
    return out.t();
}
