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

pub fn relu(mat: anytype) void {
    const Matrix = @TypeOf(mat.*);
    const zero: Matrix.RowVec = @splat(0);
    for (0..mat.rows()) |row| {
        const values: Matrix.RowVec = mat.data[row];
        mat.data[row] = @select(f32, values > zero, values, zero);
    }
}

pub fn sigmoid(mat: anytype) void {
    const Matrix = @TypeOf(mat.*);
    const one: Matrix.RowVec = @splat(1);
    for (0..mat.rows()) |row| {
        const values: Matrix.RowVec = mat.data[row];
        mat.data[row] = one / (one + @exp(-values));
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
    const Matrix = @TypeOf(mat.*);
    var temp = mat.dupe_like(.clone);

    const max_vec: Matrix.RowVec = temp.max_cwise();
    for (0..temp.rows()) |r| { // Here, we subtract the max value from each element's row before exponentiating to avoid overflow
        temp.data[r] = @as(Matrix.RowVec, temp.data[r]) - max_vec;
    }

    const e_mat = temp.exp();
    const sum_vec: Matrix.RowVec = e_mat.sum_cwise();
    for (0..temp.rows()) |i| {
        mat.data[i] = @as(Matrix.RowVec, e_mat.data[i]) / sum_vec;
    }
}

pub fn single_softmax(mat: anytype) void {
    // We transpose the matrix immediately, so that we may compute softmax per column in, but treat them per row for SIMD purposes
    var temp = mat.t();
    const Temp = @TypeOf(temp);

    // Here, we subtract the max value from each element's row before exponentiating to avoid overflow
    for (0..temp.rows()) |r| {
        const values: Temp.RowVec = temp.data[r];
        const maxv = @reduce(.Max, values);
        temp.data[r] = values - @as(Temp.RowVec, @splat(maxv));
    }

    // softmax per row (transposed)
    const e_mat = temp.exp();
    const e_sum = e_mat.sum_rwise();
    for (0..temp.rows()) |i| {
        const inv = 1.0 / e_sum[i];
        temp.data[i] = @as(Temp.RowVec, e_mat.data[i]) * @as(Temp.RowVec, @splat(inv));
    }
    // transpose the output again to retain correct shape
    mat.* = temp.t();
}
