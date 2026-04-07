const std = @import("std");
const print = std.debug.print;

pub fn is_matrix(comptime maybe_mat_type: anytype) bool {
    // @compileLog(maybe_mat_type);
    const fields = std.meta.fields(maybe_mat_type);
    // todo : gate on comptime n and m too
    return std.mem.eql(u8, fields[0].name, "data");
}

test "test_isMatrix" {
    var test_mat = Mat(2, 4).create(0);
    const test_mat_ptr = &test_mat;
    try std.testing.expect(is_matrix(Mat(3, 4)) == true);
    try std.testing.expect(is_matrix(@TypeOf(test_mat)) == true);
    try std.testing.expect(is_matrix(@TypeOf(test_mat_ptr.*)) == true);
    try std.testing.expect(is_matrix(struct { not_data: f32 }) == false);
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

        pub fn createRandom(prng: *std.Random.Xoshiro256) This {
            var mat: Mat(n, m) = undefined;
            mat.random_fill(prng.random());
            return mat;
        }

        pub fn dupe_like(mat: This, clone_or_zero: union(enum) { clone, zero }) This {
            return switch (clone_or_zero) {
                .clone => {
                    var new: Mat(n, m) = undefined;
                    new.data = mat.data;
                    return new;
                },
                .zero => This.create(0),
            };
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

        pub inline fn r(self: *const This, comptime i: usize) @Vector(n, f32) {
            return self.data[i];
        }

        pub inline fn c(self: *const This, comptime i: usize) @Vector(m, f32) { // todo: come back to this
            var col: @Vector(m, f32) = undefined;
            inline for (0..m) |j| {
                col[j] = self.data[j][i];
            }
            return col;
        }

        pub inline fn set(self: *This, row: usize, col: usize, val: f32) void {
            self.data[row][col] = val;
        }

        pub inline fn get(self: *const This, row: usize, col: usize) f32 {
            return self.data[row][col];
        }

        pub fn random_fill(mat: *This, rand: std.Random) void {
            for (&mat.data) |*row| {
                var i: usize = 0;
                while (i < m) : (i += 1) {
                    row[i] = random_normalized_float(rand);
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

        pub inline fn max_vec(vec: anytype) f32 {
            return @reduce(.Max, vec);
        }

        pub fn max_rwise(mat: *const This) @Vector(n, f32) {
            var out: @Vector(n, f32) = undefined;
            for (0..n) |row| {
                out[row] = max_vec(mat.data[row]);
            }
            return out;
        }

        pub fn max_cwise(mat: *const This) @Vector(m, f32) {
            var out: @Vector(m, f32) = @splat(std.math.floatMin(f32));
            for (mat.data) |row| {
                out = @max(out, row);
            }
            return out;
        }

        pub inline fn sum_vec(vec: anytype) f32 {
            return @reduce(.Add, vec);
        }

        pub fn sum_rwise(mat: *const This) @Vector(n, f32) {
            var out: @Vector(n, f32) = @splat(0);
            for (0..n) |i| {
                out[i] = sum_vec(mat.data[i]);
            }
            return out;
        }

        pub fn sum_cwise(mat: *const This) @Vector(m, f32) {
            var out: @Vector(m, f32) = @splat(0);
            for (mat.data) |row| {
                out += row;
            }
            return out;
        }

        inline fn exp_row(row: @Vector(m, f32)) @Vector(m, f32) {
            return @exp(row);
        }

        pub fn exp(mat: *const This) This {
            var out = Mat(n, m).create(0);
            for (0..n) |row| {
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
            switch (comptime add_is_defined(@TypeOf(a.*), @TypeOf(b))) {
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
            switch (comptime add_is_defined(@TypeOf(a.*), @TypeOf(b))) {
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

        fn add_is_defined(comptime a_type: anytype, comptime b_type: anytype) union(enum) { none, full, per_row } {
            if (a_type.n == b_type.n and a_type.m == b_type.m) return .full;
            if (a_type.n == b_type.n and b_type.m == 1) return .per_row;
            return .none;
        }

        test "addIsDefined" {
            try std.testing.expect(add_is_defined(Mat(2, 4), Mat(2, 4)) == .full);
            try std.testing.expect(add_is_defined(Mat(15, 25), Mat(15, 1)) == .per_row);
            try std.testing.expect(add_is_defined(Mat(15, 25), Mat(15, 2)) == .none);
            try std.testing.expect(add_is_defined(Mat(1, 2), Mat(3, 4)) == .none);
        }

        fn dot(comptime l: usize, a: @Vector(l, f32), b: @Vector(l, f32)) f32 {
            return @reduce(.Add, a * b);
        }

        pub fn mul(a: *const This, b: anytype, batched: bool) Mat(n, @TypeOf(b).m) {
            if (!comptime is_matrix(@TypeOf(a.*)) or !is_matrix(@TypeOf(b))) @compileError("The 'matrix' you provided is not really a matrix");
            if (!comptime mul_is_defined(@TypeOf(a.*), @TypeOf(b))) @compileError("Your multipication is misaligned, B must have the same number of rows as A has columns!");
            if (batched) {
                return batch_mul(a, b);
            } else {
                return single_mul(a, b);
            }
        }

        fn batch_mul(a: *const This, b: anytype) Mat(n, @TypeOf(b).m) {
            var out = Mat(n, @TypeOf(b).m).create(0);
            for (0..n) |row| {
                for (0..m) |col| { // broadcasts the row of A to each column of B and sums their product to the output
                    out.data[row] += @as(@TypeOf(out.data[row]), @splat(a.data[row][col])) * b.data[col];
                }
            }
            return out;
        }

        fn single_mul(a: *const This, b: anytype) Mat(n, @TypeOf(b).m) {
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

        fn mul_is_defined(comptime a_type: anytype, comptime b_type: anytype) bool {
            return a_type.m == b_type.n;
        }

        test "mulIsDefined" {
            try std.testing.expect(mul_is_defined(Mat(4, 2), Mat(2, 5)) == true);
            try std.testing.expect(mul_is_defined(Mat(1, 2), Mat(3, 4)) == false);
        }

        fn random_normalized_float(rand: std.Random) f32 {
            const rand_float = rand.float(f32);
            return 2 * rand_float - 1;
        }
    };
}
