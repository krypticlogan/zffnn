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
        const Row = @Vector(m, f32);
        const Col = @Vector(n, f32);

        const tile_width = std.simd.suggestVectorLength(f32) orelse 1;
        const TiledRow = @Vector(tile_width, f32);
        const row_tiles = m / tile_width;
        const row_tiles_leftover = m % tile_width;

        const col_tiles = n / tile_width;
        const col_tiles_leftover = n % tile_width;

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
            mat.fill(0);
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
            switch (comptime add_is_defined(@TypeOf(a.*), @TypeOf(b))) {
                .full => for (0..out.rows()) |row| { // simd per row due to vector type
                    out.data[row] = a.data[row] + b.data[row];
                },
                .per_row => for (0..out.rows()) |row| { // simd per row due to vector type
                    const bi: f32 = b.data[row][0];
                    out.data[row] = a.data[row] + @as(Row, @splat(bi));
                },
                .none => @compileError("Your add is misaligned, A and B must have matching rows!"),
            }
            return out;
        }

        pub fn i_add(a: *This, b: anytype) void {
            switch (comptime add_is_defined(@TypeOf(a.*), @TypeOf(b))) {
                .full => for (0..a.rows()) |row| { // simd per row due to vector type
                    a.data[row] += b.data[row];
                },
                .per_row => for (0..a.rows()) |row| { // simd per row due to vector type
                    const bi: f32 = b.data[row][0];
                    a.data[row] += @as(Row, @splat(bi));
                },
                .none => @compileError("Your add is misaligned, A and B must have matching rows!"),
            }
        }

        pub fn sub(a: *const This, b: anytype) This {
            var out = This.create(0);
            switch (comptime add_is_defined(@TypeOf(a.*), @TypeOf(b))) {
                .full => for (0..out.rows()) |row| { // simd per row due to vector type
                    out.data[row] = a.data[row] - b.data[row];
                },
                .per_row => for (0..out.rows()) |row| { // simd per row due to vector type
                    const bi: f32 = b.data[row][0];
                    out.data[row] = a.data[row] - @as(Row, @splat(bi));
                },
                .none => @compileError("Your sub is misaligned, A and B must have matching rows!"),
            }
            return out;
        }

        pub fn i_sub(a: *This, b: anytype) void {
            switch (comptime add_is_defined(@TypeOf(a.*), @TypeOf(b))) {
                .full => for (0..a.rows()) |row| { // simd per row due to vector type
                    a.data[row] -= b.data[row];
                },
                .per_row => for (0..a.rows()) |row| { // simd per row due to vector type
                    const bi: f32 = b.data[row][0];
                    a.data[row] -= @as(Row, @splat(bi));
                },
                .none => @compileError("Your sub is misaligned, A and B must have matching rows!"),
            }
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

        pub fn mul(a: *const This, b: anytype, batched: bool) Mat(n, @TypeOf(b.*).m) { // fast path
            if (!comptime is_matrix(@TypeOf(a.*)) or !is_matrix(@TypeOf(b.*))) @compileError("The 'matrix' you provided is not really a matrix");
            if (!comptime mul_is_defined(@TypeOf(a.*), @TypeOf(b.*))) @compileError("Your multipication is misaligned, B must have the same number of rows as A has columns!");
            if (batched) {
                return batch_mul(a, b);
            } else {
                return single_mul(a, b);
            }
        }

        fn batch_mul(a: *const This, b: anytype) Mat(n, @TypeOf(b.*).m) {
            var out = Mat(n, @TypeOf(b.*).m).create(0);
            for (0..n) |row| {
                for (0..m) |col| { // broadcasts the row of A to each column of B and sums their product to the output
                    out.data[row] += @as(@TypeOf(out.data[row]), @splat(a.data[row][col])) * b.data[col];
                }
            }
            return out;
        }

        fn single_mul(a: *const This, b: anytype) Mat(n, @TypeOf(b.*).m) { // fast path todo: room for improvement probably
            var out = Mat(n, @TypeOf(b.*).m).create(0);
            const b_t = b.t();
            for (0..n) |row| {
                var out_row: @TypeOf(b.*).Row = undefined;
                for (0..@TypeOf(b.*).m) |col| {
                    out_row[col] = dot(m, a.data[row], b_t.data[col]);
                }
                out.data[row] = out_row;
            }
            return out;
        }

        pub fn mul_into(a: *const This, b: anytype, out: *Mat(n, @TypeOf(b.*).m), batched: bool, large: bool) void {
            if (!comptime is_matrix(@TypeOf(a.*)) or !is_matrix(@TypeOf(b.*))) @compileError("The 'matrix' you provided is not really a matrix");
            if (!comptime mul_is_defined(@TypeOf(a.*), @TypeOf(b.*))) @compileError("Your multipication is misaligned, B must have the same number of rows as A has columns!");
            if (batched) {
                if (large) {
                    large_mul_into(a, b, out);
                } else {
                    batch_mul_into(a, b, out);
                }
            } else {
                single_mul_into(a, b, out);
            }
        }
        
        pub fn large_mul_into(a: *const This, b: anytype, out: *Mat(n, @TypeOf(b.*).m)) void {
            out.clear();
            // std.debug.print("large_mul_into: n={} m={}\n", .{n, m});
            const R = @TypeOf(a.*).tile_width;
            var _r: usize = 0;
            while (_r < n) : (_r += R) {
                const r_end = @min(_r + R, n);
                var k: usize = 0;
                while (k < m) : (k += 1) {
                    var rr = _r;
                    while (rr < r_end) : (rr += 1) {
                        out.data[rr] += @as(@TypeOf(b.*).Row, @splat(a.data[rr][k])) *  b.data[k];
                    }
                }
            }
        }

        pub fn batch_mul_into(a: *const This, b: anytype, out: *Mat(n, @TypeOf(b.*).m)) void {
            out.clear();
            for (0..n) |row| {
                for (0..m) |col| { // broadcasts the row of A to each column of B and sums their product to the output
                    out.data[row] += @as(@TypeOf(b.*).Row, @splat(a.data[row][col])) * b.data[col];
                }
            }
        }

        pub fn single_mul_into(a: *const This, b: anytype, out: *Mat(n, @TypeOf(b.*).m)) void {
            out.clear();
            const b_t = b.t();
            for (0..n) |row| {
                for (0..@TypeOf(b.*).m) |col| {
                    out.data[row][col] += dot(m, a.data[row], b_t.data[col]);
                }
            }
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
