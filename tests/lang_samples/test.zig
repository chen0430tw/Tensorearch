const std = @import("std");

pub fn fibonacci(n: u64) u64 {
    if (n <= 1) return n;
    var a: u64 = 0;
    var b: u64 = 1;
    for (0..n - 1) |_| {
        const temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}

fn dangerousOp() !void {
    const allocator = std.heap.page_allocator;
    const buf = try allocator.alloc(u8, 1024);
    defer allocator.free(buf);
    errdefer std.log.err("allocation cleanup failed", .{});
    if (buf.len == 0) @panic("zero length buffer");
}

fn comptime_demo() void {
    comptime {
        const x = 42;
        _ = x;
    }
}

test "fibonacci" {
    try std.testing.expectEqual(fibonacci(10), 55);
}
