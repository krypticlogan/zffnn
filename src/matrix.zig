const std = @import("std");
const print = std.debug.print;

pub fn isMatrix(comptime maybe_mat_type: anytype) bool {
    // @compileLog(maybe_mat_type);
    const fields = std.meta.fields(maybe_mat_type);
    // todo : gate on comptime n and m too
    return std.mem.eql(u8, fields[0].name, "data");
}

test "test_isMatrix" {
    var test_mat = Mat(2, 4).create(0);
    const test_mat_ptr = &test_mat;
    try std.testing.expect(isMatrix(Mat(3, 4)) == true);
    try std.testing.expect(isMatrix(@TypeOf(test_mat)) == true);
    try std.testing.expect(isMatrix(@TypeOf(test_mat_ptr.*)) == true);
    try std.testing.expect(isMatrix(struct { not_data: f32 }) == false);
}

pub fn Mat(comptime row_ct: usize, comptime col_ct: usize) type {
    return struct {
        const This = @This();
        pub const n = row_ct;
        pub const m = col_ct;
        const Axis = union(enum) { r, c };

        data: [n]@Vector(m, f32), // row-major [2, 4] is 2 rows x 4 columns

        pub fn init(mat: *This, fill_with: f32) void {
            mat.fill(fill_with);
        }

        pub fn create(fill_with: f32) This {
            var mat: Mat(n, m) = undefined;
            mat.fill(fill_with);
            return mat;
        }
        
        pub fn dupe_like(_: This) This {
            return This.create(0);
        }

        pub fn load(mat: *This, arr_mat: [n][m]f32) void {
            for (0..n) |row| {
                for (0..m) |col| {
                    mat.data[row][col] = arr_mat[row][col];
                }
            }
        }

        pub fn show(mat: *const This) void {
            print("Mat: \n", .{});
            for (mat.data) |row| {
                print("{any}\n", .{row});
            }
            print("\n", .{});
        }

        pub inline fn rows(_: This) usize {
            return n;
        }

        pub inline fn cols(_: This) usize {
            return m;
        }
        
        pub inline fn set(self: *This, row: usize, col: usize, val: f32) void {
            self.data[row][col] = val;
        }

        pub inline fn get(self: *const This, row: usize, col: usize) f32 {
            return self.data[row][col];
        }

        pub fn randomFill(mat: *This, rand: std.Random) void {
            for (&mat.data) |*row| {
                var i: usize = 0;
                while (i < m) : (i += 1) {
                    row[i] = randomNormalizedFloat(rand);
                }
            }
        }

        pub fn fill(mat: *This, fill_with: f32) void {
            for (&mat.data) |*row| {
                row.* = @splat(fill_with);
            }
        }

        pub fn clear(mat: *This) void {
            mat.* = undefined;
            mat.* = This.create(0);
        }

        fn max_row(row: @Vector(m, f32)) f32 {
            return @reduce(.Max, row);
        }

        pub fn max(mat: *const This) Mat(n, 1) {
            var out = Mat(n, 1).create(0);
            for (0..mat.rows()) |row| {
                out.data[row][0] = max_row(mat.data[row]);
            }
            return out;
        }

        fn sum_row(row: @Vector(m, f32)) f32 {
            return @reduce(.Add, row);
        }

        pub fn sum(mat: *const This) Mat(n, 1) {
            var out = Mat(n, 1).create(0);
            for (0..mat.rows()) |row| {
                out.data[row][0] = sum_row(mat.data[row]);
            }
            return out;
        }

        fn exp_row(row: @Vector(m, f32)) @Vector(m, f32) {
            var out: @Vector(m, f32) = undefined;
            for (0..m) |i| {
                out[i] = @exp(row[i]);
            }
            return out;
        }

        pub fn exp(mat: *const This) This {
            var out = Mat(n, m).create(0);
            for (0..mat.rows()) |row| {
                out.data[row] = exp_row(mat.data[row]);
            }
            return out;
        }

        pub fn t(mat: *const This) Mat(m, n) { // transpose
            var out = Mat(m, n).create(0);
            for (0..mat.rows()) |row| {
                for (0..mat.cols()) |col| {
                    out.set(col, row, mat.get(row, col));
                }
            }
            return out;
        }

        // todo : inplace variants
        pub fn add(a: *const This, b: anytype) This {
            var out = This.create(0);
            switch (comptime addIsDefined(@TypeOf(a.*), @TypeOf(b))) {
                .full => for (0..out.rows()) |row| { // simd per row due to vector type
                    out.data[row] = a.data[row] + b.data[row];
                },
                .per_row => for (0..out.rows()) |row| { // simd per row due to vector type
                    const bi: f32 = b.data[row][0];
                    out.data[row] = a.data[row] + @as(@Vector(m, f32), @splat(bi));
                },
                .none => @compileError("Your add is misaligned, A and B must have matching rows!"),
            }
            return out;
        }

        pub fn sub(a: *const This, b: anytype) This {
            var out = This.create(0);
            switch (comptime addIsDefined(@TypeOf(a.*), @TypeOf(b))) {
                .full => for (0..out.rows()) |row| { // simd per row due to vector type
                    out.data[row] = a.data[row] - b.data[row];
                },
                .per_row => for (0..out.rows()) |row| { // simd per row due to vector type
                    const bi: f32 = b.data[row][0];
                    out.data[row] = a.data[row] - @as(@Vector(m, f32), @splat(bi));
                },
                .none => @compileError("Your sub is misaligned, A and B must have matching rows!"),
            }
            return out;
        }

        fn addIsDefined(comptime a_type: anytype, comptime b_type: anytype) union(enum) { none, full, per_row } {
            if (a_type.n == b_type.n and a_type.m == b_type.m) return .full;
            if (a_type.n == b_type.n and b_type.m == 1) return .per_row;
            return .none;
        }

        test "test_isAddDefined" {
            try std.testing.expect(addIsDefined(Mat(2, 4), Mat(2, 4)) == .full);
            try std.testing.expect(addIsDefined(Mat(15, 25), Mat(15, 1)) == .per_row);
            try std.testing.expect(addIsDefined(Mat(15, 25), Mat(15, 2)) == .none);
            try std.testing.expect(addIsDefined(Mat(1, 2), Mat(3, 4)) == .none);
        }

        fn dot(comptime l: usize, a: @Vector(l, f32), b: @Vector(l, f32)) f32 {
            return @reduce(.Add, a * b);
        }

        pub fn mul(a: *const This, b: anytype) Mat(n, @TypeOf(b).m) {
            if (!comptime isMatrix(@TypeOf(a.*)) or !isMatrix(@TypeOf(b))) @compileError("The 'matrix' you provided is not really a matrix");
            if (!comptime mulIsDefined(@TypeOf(a.*), @TypeOf(b))) @compileError("Your multipication is misaligned, B must have the same number of rows as A has columns!");

            var out = Mat(n, @TypeOf(b).m).create(0);
            const b_t = b.t();

            for (0..n) |row| {
                var out_row: @Vector(@TypeOf(b).m, f32) = undefined;
                for (0..@TypeOf(b).m) |col| {
                    out_row[col] = dot(m, a.data[row], b_t.data[col]);
                }
                out.data[row] = out_row;
            }

            return out;
        }

        fn mulIsDefined(comptime a_type: anytype, comptime b_type: anytype) bool {
            return a_type.m == b_type.n;
        }

        test "test_isMulDefined" {
            try std.testing.expect(mulIsDefined(Mat(4, 2), Mat(2, 5)) == true);
            try std.testing.expect(mulIsDefined(Mat(1, 2), Mat(3, 4)) == false);
        }

        // pub fn relu(mat: *const This) This {
        //     var out = Mat(n, m).create(0);

        //     for (0..mat.rows()) |row| {
        //         for (0..mat.cols()) |col| {
        //             out.set(row, col, @max(mat.get(row, col), 0));
        //         }
        //     }

        //     return out;
        // }

        // pub fn sigmoid(mat: *const This) This {
        //     var out = Mat(n, m).create(0);
        //     for (0..mat.rows()) |row| {
        //         for (0..mat.cols()) |col| {
        //             out.set(row, col, 1 / (1 + @exp(-mat.get(row, col))));
        //         }
        //     }
        //     return out;
        // }

        // pub fn softmax(mat: *const This) This {
        //     // We transpose the matrix immediately, so that we may compute softmax per column in, but treat them per row for SIMD purposes
        //     var out = mat.t();

        //     // Here, we subtract the max value from each element's row before exponentiating to avoid overflow
        //     for (0..out.rows()) |r| {
        //         const maxv = @reduce(.Max, out.data[r]);
        //         out.data[r] -= @as(@Vector(@TypeOf(out).m, f32), @splat(maxv));
        //     }

        //     // softmax per row (allowed since transposed)
        //     const e_mat = out.exp();
        //     const e_sum = e_mat.sum();
        //     for (0..out.rows()) |i| {
        //         const inv = 1.0 / e_sum.get(i, 0);
        //         out.data[i] = e_mat.data[i] * @as(@Vector(n, f32), @splat(inv));
        //     }
        //     // transpose the output again to retain correct shape
        //     return out.t();
        // }

        fn randomNormalizedFloat(rand: std.Random) f32 {
            const rand_float = rand.float(f32);
            return 2 * rand_float - 1;
        }
    };
}
