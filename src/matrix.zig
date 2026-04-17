const std = @import("std");
const validation = @import("validation.zig");
const print = std.debug.print;
const as_arr = @import("helpers.zig").as_arr;

pub fn Mat(comptime row_ct: usize, comptime col_ct: usize) type {
    return struct {
        const This = @This();
        pub const n = row_ct;
        pub const m = col_ct;
        pub const Axis = union(enum) { r, c };
        pub const Row = @Vector(m, f32);
        pub const Col = @Vector(n, f32);

        const tile_width = std.simd.suggestVectorLength(f32) orelse 1;

        pub const TiledRow = @Vector(tile_width, f32);
        const row_tiles: usize = @ceil(@as(f16, (@floatFromInt(m))) / tile_width);

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

        pub fn get_tile(self: *const This, row_i: usize, start: usize) TiledRow {
            const end = @min(start + tile_width, m);
            
            var tile_arr = as_arr(TiledRow, @splat(0));
            // std.debug.print("get_tile: row_i={d} start={d} end={d}\n", .{row_i, start, end});
            for (&tile_arr, start..end) |*elem, i| {
                elem.* = as_arr(Row, self.data[row_i])[i];
            }
            return tile_arr;
        }

        pub fn load_tile(self: *This, row_i: usize, start: usize, tile: TiledRow) void {
            var arr = as_arr(Row, self.data[row_i]);
            const end = @min(start + tile_width, m);
            // std.debug.print("load_tile: row_i={d} start={d} end={d}\n", .{row_i, start, end});
            for (start..end) |i| {
                arr[i] = as_arr(TiledRow, tile)[i - start];
            }
            self.data[row_i] = arr;
        }

        pub inline fn set(self: *This, row: usize, col: usize, val: f32) void {
            var arr = as_arr(Row, self.data[row]);
            arr[col] = val;
            self.data[row] = arr;
        }

        pub inline fn get(self: *const This, row: usize, col: usize) f32 {
            return as_arr(Row, self.data[row])[col];
        }

        pub fn random_fill(mat: *This, rand: std.Random) void {
            for (&mat.data) |*row| {
                var arr = as_arr(Row, row.*);
                for (&arr) |*elem| {
                    elem.* = random_normalized_float(rand);
                }
                row.* = arr;
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
            var out_arr = as_arr(Col, @splat(0));
            for (0..n) |i| {
                out_arr[i] = sum_vec(mat.data[i]);
            }
            return out_arr;
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

        pub fn add(a: *const This, b: anytype) This {
            var out = This.create(0);
            switch (comptime validation.elemwise_is_defined(@TypeOf(a.*), @TypeOf(b))) {
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

        pub fn add_(a: *This, b: anytype) void {
            switch (comptime validation.elemwise_is_defined(@TypeOf(a.*), @TypeOf(b))) {
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
            switch (comptime validation.elemwise_is_defined(@TypeOf(a.*), @TypeOf(b))) {
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

        pub fn sub_(a: *This, b: anytype) void {
            switch (comptime validation.elemwise_is_defined(@TypeOf(a.*), @TypeOf(b))) {
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
            var out = Mat(n, @TypeOf(b.*).m).create(0);
            for (0..n) |row| {
                for (0..m) |col| { // broadcasts the row of A to each column of B and sums their product to the output
                    out.data[row] += @as(@TypeOf(out.data[row]), @splat(as_arr(Row, a.data[row])[col])) * b.data[col];
                }
            }
            return out;
        }

        fn single_mul(a: *const This, b: anytype) Mat(n, @TypeOf(b.*).m) { // fast path todo: room for improvement probably
            var out = Mat(n, @TypeOf(b.*).m).create(0);
            const b_t = b.t();
            for (0..n) |row| {
                var out_arr = as_arr(@TypeOf(b.*).Row, out.data[row]);
                for (0..@TypeOf(b.*).m) |col| {
                    out_arr[col] = dot(m, a.data[row], b_t.data[col]);
                }
                out.data[row] = out_arr;
            }
            return out;
        }

        pub fn mul_(a: *const This, b: anytype, out: *Mat(n, @TypeOf(b.*).m), mul_size: usize) void {
            // _ = batched;
            if (!comptime validation.is_matrix(@TypeOf(a.*)) or !validation.is_matrix(@TypeOf(b.*))) @compileError("The 'matrix' you provided is not really a matrix");
            if (!comptime validation.mul_is_defined(@TypeOf(a.*), @TypeOf(b.*))) @compileError("Your multipication is misaligned, B must have the same number of rows as A has columns!");
            // const batch = @TypeOf(b.*).m;
            switch (mul_size) {
                1 => batch_mul_(a, b, out),
                2 => large_mul_(a, b, out),
                3 => blocked_large_mul_(a, b, out),
                else => @panic("unsupported mul_size"),
            }
        }

        pub fn large_mul_(a: *const This, b: anytype, out: *Mat(n, @TypeOf(b.*).m)) void { // todo:  test for equality between other muls
            out.clear();
            // std.debug.print("large_mul_into: n={} m={}\n", .{n, m});
            const R = tile_width * 2;
            var _r: usize = 0;
            while (_r < n) : (_r += R) {
                var accs: [R]@TypeOf(b.*).Row = undefined; // accumulator for the tile
                inline for (&accs) |*acc| acc.* = @splat(0);

                const r_end = @min(_r + R, n);
                var k: usize = 0;

                while (k < m) : (k += 1) {
                    var rr = _r;
                    while (rr < r_end) : (rr += 1) {
                        accs[rr - _r] += @as(@TypeOf(b.*).Row, @splat(as_arr(Row, a.data[rr])[k])) * b.data[k];
                    }
                }
                var rr = _r;
                while (rr < r_end) : (rr += 1) {
                    out.data[rr] = accs[rr - _r];
                }
            }
        }

        pub fn blocked_large_mul_(a: *const This, b: anytype, out: *Mat(n, @TypeOf(b.*).m)) void { // todo:  test for equality between other muls
            out.clear();
            // std.debug.print("blocked large_mul_into: n={} m={} tile_width={}\n", .{n, m, tile_width});
            // const R = @TypeOf(a.*).tile_width;
            const R = tile_width * 2;
            const C = @TypeOf(b.*).tile_width;
            var _r: usize = 0;
            while (_r < n) : (_r += R) {
                var j: usize = 0;
                while (j < @TypeOf(b.*).m) : (j += C) {
                    var accs: [R]@TypeOf(b.*).TiledRow = undefined; // accumulator for the tile
                    inline for (&accs) |*acc| acc.* = @splat(0);
                    const r_end = @min(_r + R, n);
                    var _k: usize = 0;
                    while (_k < m) : (_k += 1) {
                        const b_tile: @TypeOf(b.*).TiledRow = b.get_tile(_k, j);
                        var rr = _r;
                        while (rr < r_end) : (rr += 1) {
                            accs[rr - _r] += @as(@TypeOf(b.*).TiledRow, @splat(as_arr(Row, a.data[rr])[_k])) * b_tile;
                        }
                    }
                    var rr = _r;
                    while (rr < r_end) : (rr += 1) {
                        out.load_tile(rr, j, accs[rr - _r]);
                    }
                }
            }
        }

        pub fn batch_mul_(a: *const This, b: anytype, out: *Mat(n, @TypeOf(b.*).m)) void {
            out.clear();
            for (0..n) |row| {
                for (0..m) |col| { // broadcasts the row of A to each column of B and sums their product to the output
                    out.data[row] += @as(@TypeOf(b.*).Row, @splat(as_arr(Row, a.data[row])[col])) * b.data[col];
                }
            }
        }

        pub fn single_mul_(a: *const This, b: anytype, out: *Mat(n, @TypeOf(b.*).m)) void {
            out.clear();
            const b_t = b.t();
            for (0..n) |row| {
                for (0..@TypeOf(b.*).m) |col| {
                    out.data[row][col] += dot(m, a.data[row], b_t.data[col]);
                }
            }
        }
        
        fn random_normalized_float(rand: std.Random) f32 {
            const rand_float = rand.float(f32);
            return 2 * rand_float - 1;
        }
    };
}
