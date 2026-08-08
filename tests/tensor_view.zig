const std = @import("std");
const zgc = @import("zgc");

test "contiguous layout computes row-major strides" {
    const Shape = zgc.Tensor.Shape(3);
    const Layout = zgc.Tensor.Layout(3);
    const layout = Layout.contiguous(Shape.init(&.{ 2, 3, 4 }));

    try std.testing.expectEqual(@as(usize, 0), layout.offset);
    try std.testing.expectEqual([3]isize{ 12, 4, 1 }, layout.strides);
}

test "view maps logical indices through offset and strides" {
    var storage = [_]i32{ 0, 1, 2, 3, 4, 5, 6, 7 };
    var view: zgc.Tensor.View(i32, 2) = .{
        .storage = &storage,
        .shape = .{ 2, 2 },
        .strides = .{ 3, 1 },
        .offset = 1,
    };

    try std.testing.expectEqual(@as(usize, 4), view.len());
    try std.testing.expectEqual(@as(usize, 1), view.elementOffset(.{ 0, 0 }));
    try std.testing.expectEqual(@as(usize, 5), view.elementOffset(.{ 1, 1 }));
    try std.testing.expectEqual(@as(i32, 5), view.get(.{ 1, 1 }));

    view.set(.{ 1, 0 }, 40);
    try std.testing.expectEqual(@as(i32, 40), storage[4]);
}

test "views identify row-major contiguity" {
    var storage: [8]f32 = @splat(0);
    const contiguous: zgc.Tensor.View(f32, 2) = .{
        .storage = &storage,
        .shape = .{ 2, 3 },
        .strides = .{ 3, 1 },
        .offset = 0,
    };
    const transposed: zgc.Tensor.View(f32, 2) = .{
        .storage = &storage,
        .shape = .{ 3, 2 },
        .strides = .{ 1, 3 },
        .offset = 0,
    };
    const singleton_axis: zgc.Tensor.View(f32, 3) = .{
        .storage = &storage,
        .shape = .{ 2, 1, 4 },
        .strides = .{ 4, 99, 1 },
        .offset = 0,
    };

    try std.testing.expect(contiguous.isContiguous());
    try std.testing.expect(!transposed.isContiguous());
    try std.testing.expect(singleton_axis.isContiguous());
}

test "contiguous slices respect logical offset and length" {
    var storage = [_]i32{ -1, -1, 10, 20, 30, 40, -1 };
    var view: zgc.Tensor.View(i32, 2) = .{
        .storage = &storage,
        .shape = .{ 2, 2 },
        .strides = .{ 2, 1 },
        .offset = 2,
    };
    const const_view: zgc.Tensor.ConstView(i32, 2) = .{
        .storage = &storage,
        .shape = .{ 2, 2 },
        .strides = .{ 2, 1 },
        .offset = 2,
    };
    const strided: zgc.Tensor.ConstView(i32, 2) = .{
        .storage = &storage,
        .shape = .{ 2, 2 },
        .strides = .{ 3, 1 },
        .offset = 1,
    };

    const mutable_slice = view.contiguousSlice().?;
    try std.testing.expectEqualSlices(i32, &.{ 10, 20, 30, 40 }, mutable_slice);
    mutable_slice[1] = 200;
    try std.testing.expectEqual(@as(i32, 200), storage[3]);
    try std.testing.expectEqualSlices(
        i32,
        &.{ 10, 200, 30, 40 },
        const_view.contiguousSlice().?,
    );
    try std.testing.expect(strided.contiguousSlice() == null);
}
