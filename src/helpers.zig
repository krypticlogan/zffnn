pub inline fn as_arr(comptime vec_type: type, vec: vec_type) [@typeInfo(vec_type).vector.len]@typeInfo(vec_type).vector.child {
    return vec;
}