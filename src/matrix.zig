const std = @import("std");
const validation = @import("validation.zig");
const print = std.debug.print;

pub fn Mat(comptime row_ct: usize, comptime col_ct: usize) type {
    return struct {
        const This = @This();
        pub const n = row_ct;
        pub const m = col_ct;
        pub const Axis = union(enum) { r, c };
        pub const Row = [m]f32;
        pub const Col = [n]f32;
        pub const RowVec = @Vector(m, f32);
        pub const ColVec = @Vector(n, f32);

        data: [n]Row,

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
                mat.data[row] = arr_mat[row];
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

        pub inline fn r(self: *const This, comptime i: usize) Row {
            return self.data[i];
        }

        pub inline fn c(self: *const This, comptime i: usize) Col { // todo: come back to this
            var col: Col = undefined;
            inline for (0..n) |j| {
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
                for (row) |*elem| {
                    elem.* = random_normalized_float(rand);
                }
            }
        }

        pub fn fill(mat: *This, fill_with: f32) void {
            for (&mat.data) |*row| {
                @memset(row, fill_with);
            }
        }

        pub fn clear(mat: *This) void {
            mat.fill(0);
        }

        inline fn max_vec(vec: anytype) f32 {
            return @reduce(.Max, vec);
        }

        pub fn max_rwise(mat: *const This) Col {
            var out: Col = undefined;
            for (0..n) |row| {
                out[row] = max_vec(@as(RowVec, mat.data[row]));
            }
            return out;
        }

        pub fn max_cwise(mat: *const This) Row {
            var out: RowVec = @splat(-std.math.inf(f32));
            for (mat.data) |row| {
                out = @max(out, @as(RowVec, row));
            }
            return out;
        }

        inline fn sum_vec(vec: anytype) f32 {
            return @reduce(.Add, vec);
        }

        pub fn sum_rwise(mat: *const This) Col {
            var out: Col = undefined;
            for (0..n) |i| {
                out[i] = sum_vec(@as(RowVec, mat.data[i]));
            }
            return out;
        }

        pub fn sum_cwise(mat: *const This) Row {
            var out: RowVec = @splat(0);
            for (mat.data) |row| {
                out += @as(RowVec, row);
            }
            return out;
        }

        inline fn exp_row(row: RowVec) RowVec {
            return @exp(row);
        }

        pub fn exp(mat: *const This) This {
            var out = Mat(n, m).create(0);
            for (0..mat.rows()) |row| {
                out.data[row] = exp_row(@as(RowVec, mat.data[row]));
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

        pub fn add(a: *const This, b: anytype) This {
            var out = This.create(0);
            switch (comptime validation.elemwise_is_defined(@TypeOf(a.*), @TypeOf(b))) {
                .full => for (0..out.rows()) |row| { // simd per row due to vector type
                    out.data[row] = @as(RowVec, a.data[row]) + @as(RowVec, b.data[row]);
                },
                .per_row => for (0..out.rows()) |row| { // simd per row due to vector type
                    const bi: f32 = b.data[row][0];
                    out.data[row] = @as(RowVec, a.data[row]) + @as(RowVec, @splat(bi));
                },
                .none => @compileError("Your add is misaligned, A and B must have matching rows!"),
            }
            return out;
        }

        pub fn add_(a: *This, b: anytype) void {
            switch (comptime validation.elemwise_is_defined(@TypeOf(a.*), @TypeOf(b))) {
                .full => for (0..a.rows()) |row| { // simd per row due to vector type
                    a.data[row] = @as(RowVec, a.data[row]) + @as(RowVec, b.data[row]);
                },
                .per_row => for (0..a.rows()) |row| { // simd per row due to vector type
                    const bi: f32 = b.data[row][0];
                    a.data[row] = @as(RowVec, a.data[row]) + @as(RowVec, @splat(bi));
                },
                .none => @compileError("Your add is misaligned, A and B must have matching rows!"),
            }
        }

        pub fn sub(a: *const This, b: anytype) This {
            var out = This.create(0);
            switch (comptime validation.elemwise_is_defined(@TypeOf(a.*), @TypeOf(b))) {
                .full => for (0..out.rows()) |row| { // simd per row due to vector type
                    out.data[row] = @as(RowVec, a.data[row]) - @as(RowVec, b.data[row]);
                },
                .per_row => for (0..out.rows()) |row| { // simd per row due to vector type
                    const bi: f32 = b.data[row][0];
                    out.data[row] = @as(RowVec, a.data[row]) - @as(RowVec, @splat(bi));
                },
                .none => @compileError("Your sub is misaligned, A and B must have matching rows!"),
            }
            return out;
        }

        pub fn sub_(a: *This, b: anytype) void {
            switch (comptime validation.elemwise_is_defined(@TypeOf(a.*), @TypeOf(b))) {
                .full => for (0..a.rows()) |row| { // simd per row due to vector type
                    a.data[row] = @as(RowVec, a.data[row]) - @as(RowVec, b.data[row]);
                },
                .per_row => for (0..a.rows()) |row| { // simd per row due to vector type
                    const bi: f32 = b.data[row][0];
                    a.data[row] = @as(RowVec, a.data[row]) - @as(RowVec, @splat(bi));
                },
                .none => @compileError("Your sub is misaligned, A and B must have matching rows!"),
            }
        }

        inline fn dot(comptime l: usize, a: @Vector(l, f32), b: @Vector(l, f32)) f32 {
            return @reduce(.Add, a * b);
        }

        pub fn mul(a: *const This, b: anytype, batched: bool) Mat(n, @TypeOf(b.*).m) { // fast path
            if (!comptime validation.is_matrix(@TypeOf(a.*)) or !validation.is_matrix(@TypeOf(b.*))) @compileError("The 'matrix' you provided is not really a matrix");
            if (!comptime validation.mul_is_defined(@TypeOf(a.*), @TypeOf(b.*))) @compileError("Your multipication is misaligned, B must have the same number of rows as A has columns!");
            if (batched) {
                return batch_mul(a, b);
            } else {
                return single_mul(a, b);
            }
        }

        fn batch_mul(a: *const This, b: anytype) Mat(n, @TypeOf(b.*).m) {
            const B = @TypeOf(b.*);
            var out = Mat(n, B.m).create(0);
            for (0..n) |row| {
                var accumulator: B.RowVec = @splat(0);
                for (0..m) |col| { // broadcasts the row of A to each column of B and sums their product to the output
                    accumulator += @as(B.RowVec, @splat(a.data[row][col])) * @as(B.RowVec, b.data[col]);
                }
                out.data[row] = accumulator;
            }
            return out;
        }

        fn single_mul(a: *const This, b: anytype) Mat(n, @TypeOf(b.*).m) { // fast path todo: room for improvement probably
            const b_m = @TypeOf(b.*).m;
            var out = Mat(n, b_m).create(0);
            const b_t = b.t();
            for (0..n) |row| {
                var out_arr: [b_m]f32 = undefined;
                for (0..b_m) |col| {
                    out_arr[col] = dot(
                        m,
                        @as(RowVec, a.data[row]),
                        @as(@TypeOf(b_t).RowVec, b_t.data[col]),
                    );
                }
                out.data[row] = out_arr;
            }
            return out;
        }

        pub fn mul_(a: *const This, b: anytype, out: *Mat(n, @TypeOf(b.*).m), batched: bool) void {
            if (!comptime validation.is_matrix(@TypeOf(a.*)) or !validation.is_matrix(@TypeOf(b.*))) @compileError("The 'matrix' you provided is not really a matrix");
            if (!comptime validation.mul_is_defined(@TypeOf(a.*), @TypeOf(b.*))) @compileError("Your multipication is misaligned, B must have the same number of rows as A has columns!");
            if (batched) {
                batch_mul_(a, b, out);
            } else single_mul_(a, b, out);
        }

        pub fn batch_mul_(a: *const This, b: anytype, out: *Mat(n, @TypeOf(b.*).m)) void {
            const B = @TypeOf(b.*);
            for (0..n) |row| {
                var accumulator: B.RowVec = @splat(0);
                for (0..m) |col| { // broadcasts the row of A to each column of B and sums their product to the output
                    accumulator += @as(B.RowVec, @splat(a.data[row][col])) * @as(B.RowVec, b.data[col]);
                }
                out.data[row] = accumulator;
            }
        }

        pub fn single_mul_(a: *const This, b: anytype, out: *Mat(n, @TypeOf(b.*).m)) void {
            const out_cols = @TypeOf(b.*).m;
            const b_t = b.t();
            for (0..n) |row| {
                var out_row: [out_cols]f32 = undefined;
                for (0..out_cols) |col| {
                    out_row[col] = dot(
                        m,
                        @as(RowVec, a.data[row]),
                        @as(@TypeOf(b_t).RowVec, b_t.data[col]),
                    );
                }
                out.data[row] = out_row;
            }
        }

        fn random_normalized_float(rand: std.Random) f32 {
            const rand_float = rand.float(f32);
            return 2 * rand_float - 1;
        }
    };
}
