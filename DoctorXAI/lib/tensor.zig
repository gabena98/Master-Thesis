const std = @import("std");

pub fn Indexer(comptime nd: usize) type {
    return struct {
        shape: [nd]usize,
        strides: [nd]usize,
        size: usize,

        const Self = @This();

        pub fn init(input_shape: anytype) Self {
            const infos = @typeInfo(@TypeOf(input_shape));
            switch (infos) {
                .Struct => |struct_info| {
                    if (!struct_info.is_tuple) @compileError("Indexer should be initted with a tuple");
                    var buf: [2048]u8 = undefined;
                    if (struct_info.fields.len != nd) {
                        const msg = std.fmt.bufPrint(
                            &buf,
                            "Indexer init tuple should have size {}. Found: {}",
                            .{ nd, struct_info.fields.len },
                        ) catch unreachable;
                        @panic(msg);
                    }
                },
                else => @compileError("Indexer should be initted with a tuple"),
            }

            var ret: Self = undefined;

            inline for (0..nd) |it| {
                ret.shape[it] = @intCast(input_shape[it]);
            }
            ret.strides[nd - 1] = 1;
            inline for (1..nd) |it| {
                const jt = nd - it;
                ret.strides[jt - 1] = ret.strides[jt] * @as(usize, @intCast(input_shape[jt]));
            }

            ret.size = ret.strides[0] * @as(usize, @intCast(input_shape[0]));

            return ret;
        }

        pub fn ix(self: Self, index: anytype) usize {
            const infos = @typeInfo(@TypeOf(index));
            switch (infos) {
                .Struct => |struct_info| {
                    if (!struct_info.is_tuple) @compileError("Indexer ix function be called with a tuple");
                    var buf: [2048]u8 = undefined;
                    if (struct_info.fields.len != nd) {
                        const msg = std.fmt.bufPrint(
                            &buf,
                            "Indexer ix argument tuple should have size {}. Found: {}",
                            .{ nd, struct_info.fields.len },
                        );
                        @compileError(msg);
                    }
                },
                else => @compileError("Indexer ix function be called with a tuple"),
            }

            var ret: usize = 0;
            inline for (0..nd) |it| {
                ret += self.strides[it] * @as(usize, @intCast(index[it]));
            }

            return ret;
        }
    };
}

test "tensor 2d" {
    const index = Indexer(2).init(.{ 2, 3 });
    std.debug.assert(index.ix(.{ 0, 0 }) == 0);
    std.debug.assert(index.ix(.{ 0, 1 }) == 1);
    std.debug.assert(index.ix(.{ 0, 2 }) == 2);
    std.debug.assert(index.ix(.{ 1, 0 }) == 3);
    std.debug.assert(index.ix(.{ 1, 1 }) == 4);
    std.debug.assert(index.ix(.{ 1, 2 }) == 5);
}

test "tensor 3d" {
    const index = Indexer(3).init(.{ 2, 3, 4 });
    std.debug.assert(index.ix(.{ 0, 0, 0 }) == 0);
    std.debug.assert(index.ix(.{ 0, 1, 0 }) == 4);
    std.debug.assert(index.ix(.{ 0, 1, 2 }) == 6);
    std.debug.assert(index.ix(.{ 1, 1, 2 }) == 18);
}

test "size" {
    const index1 = Indexer(1).init(.{1});
    const index2 = Indexer(2).init(.{ 2, 4 });
    const index3 = Indexer(3).init(.{ 2, 3, 4 });
    const index4 = Indexer(4).init(.{ 2, 3, 4, 2 });

    std.debug.assert(index1.size == 1);
    std.debug.assert(index2.size == 8);
    std.debug.assert(index3.size == 24);
    std.debug.assert(index4.size == 48);
}
