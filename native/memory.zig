const std = @import("std");
const builtin = @import("builtin");

pub const MemoryPool = struct {
    buffer: []u8,
    allocator: std.mem.Allocator,
    used: usize,
    alignment: usize,

    pub fn init(allocator: std.mem.Allocator, size: usize, alignment: usize) !MemoryPool {
        const buffer = try allocator.alignedAlloc(u8, alignment, size);
        return MemoryPool{
            .buffer = buffer,
            .allocator = allocator,
            .used = 0,
            .alignment = alignment,
        };
    }

    pub fn deinit(self: *MemoryPool) void {
        self.allocator.free(self.buffer);
    }

    pub fn alloc(self: *MemoryPool, size: usize) ?[]u8 {
        const aligned_size = std.mem.alignForward(usize, size, self.alignment);
        if (self.used + aligned_size > self.buffer.len) return null;

        const ptr = self.buffer[self.used .. self.used + size];
        self.used += aligned_size;
        return ptr;
    }

    pub fn reset(self: *MemoryPool) void {
        self.used = 0;
    }

    pub fn getUsage(self: *const MemoryPool) f32 {
        return @as(f32, @floatFromInt(self.used)) / @as(f32, @floatFromInt(self.buffer.len));
    }
};

pub const ArenaAllocator = struct {
    backing_allocator: std.mem.Allocator,
    pools: std.ArrayList(*MemoryPool),
    current_pool: ?*MemoryPool,
    pool_size: usize,
    alignment: usize,

    pub fn init(backing_allocator: std.mem.Allocator, pool_size: usize, alignment: usize) ArenaAllocator {
        return ArenaAllocator{
            .backing_allocator = backing_allocator,
            .pools = std.ArrayList(*MemoryPool).init(backing_allocator),
            .current_pool = null,
            .pool_size = pool_size,
            .alignment = alignment,
        };
    }

    pub fn deinit(self: *ArenaAllocator) void {
        for (self.pools.items) |pool| {
            pool.deinit();
            self.backing_allocator.destroy(pool);
        }
        self.pools.deinit();
    }

    pub fn allocator(self: *ArenaAllocator) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .free = free,
            },
        };
    }

    fn alloc(ctx: *anyopaque, len: usize, ptr_align: u8, ret_addr: usize) ?[*]u8 {
        _ = ret_addr;
        const self: *ArenaAllocator = @ptrCast(@alignCast(ctx));

        if (self.current_pool) |pool| {
            if (pool.alloc(len)) |mem| {
                return @ptrCast(mem.ptr);
            }
        }

        const new_pool = self.backing_allocator.create(MemoryPool) catch return null;
        new_pool.* = MemoryPool.init(self.backing_allocator, @max(self.pool_size, len * 2), @as(usize, @intCast(ptr_align))) catch {
            self.backing_allocator.destroy(new_pool);
            return null;
        };

        self.pools.append(new_pool) catch {
            new_pool.deinit();
            self.backing_allocator.destroy(new_pool);
            return null;
        };

        self.current_pool = new_pool;

        const mem = new_pool.alloc(len) orelse return null;
        return @ptrCast(mem.ptr);
    }

    fn resize(ctx: *anyopaque, buf: []u8, buf_align: u8, new_len: usize, ret_addr: usize) bool {
        _ = ctx;
        _ = buf;
        _ = buf_align;
        _ = new_len;
        _ = ret_addr;
        return false;
    }

    fn free(ctx: *anyopaque, buf: []u8, buf_align: u8, ret_addr: usize) void {
        _ = ctx;
        _ = buf;
        _ = buf_align;
        _ = ret_addr;
    }

    pub fn reset(self: *ArenaAllocator) void {
        for (self.pools.items) |pool| {
            pool.reset();
        }
    }
};

pub const CacheAlignedArray = struct {
     []align(64) f32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, size: usize) !CacheAlignedArray {
        const data = try allocator.alignedAlloc(f32, 64, size);
        return CacheAlignedArray{
            .data = data,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *CacheAlignedArray) void {
        self.allocator.free(self.data);
    }

    pub fn len(self: *const CacheAlignedArray) usize {
        return self.data.len;
    }
};

pub const SIMDVector = struct {
    pub fn dotProduct(a: []const f32, b: []const f32) f32 {
        std.debug.assert(a.len == b.len);

        var sum: f32 = 0.0;
        var i: usize = 0;

        if (builtin.cpu.arch == .x86_64) {
            const vec_size = 8;
            while (i + vec_size <= a.len) : (i += vec_size) {
                const va: @Vector(8, f32) = a[i..][0..vec_size].*;
                const vb: @Vector(8, f32) = b[i..][0..vec_size].*;
                const vprod = va * vb;
                sum += @reduce(.Add, vprod);
            }
        }

        while (i < a.len) : (i += 1) {
            sum += a[i] * b[i];
        }

        return sum;
    }

    pub fn vectorAdd(dst: []f32, a: []const f32, b: []const f32) void {
        std.debug.assert(dst.len == a.len and a.len == b.len);

        var i: usize = 0;

        if (builtin.cpu.arch == .x86_64) {
            const vec_size = 8;
            while (i + vec_size <= dst.len) : (i += vec_size) {
                const va: @Vector(8, f32) = a[i..][0..vec_size].*;
                const vb: @Vector(8, f32) = b[i..][0..vec_size].*;
                const vsum = va + vb;
                @memcpy(dst[i..][0..vec_size], &vsum);
            }
        }

        while (i < dst.len) : (i += 1) {
            dst[i] = a[i] + b[i];
        }
    }

    pub fn vectorScale(dst: []f32, src: []const f32, scale: f32) void {
        std.debug.assert(dst.len == src.len);

        var i: usize = 0;

        if (builtin.cpu.arch == .x86_64) {
            const vec_size = 8;
            const vscale: @Vector(8, f32) = @splat(scale);
            while (i + vec_size <= dst.len) : (i += vec_size) {
                const vsrc: @Vector(8, f32) = src[i..][0..vec_size].*;
                const vresult = vsrc * vscale;
                @memcpy(dst[i..][0..vec_size], &vresult);
            }
        }

        while (i < dst.len) : (i += 1) {
            dst[i] = src[i] * scale;
        }
    }

    pub fn matrixMultiply(C: []f32, A: []const f32, B: []const f32, m: usize, n: usize, k: usize) void {
        std.debug.assert(C.len == m * n);
        std.debug.assert(A.len == m * k);
        std.debug.assert(B.len == k * n);

        @memset(C, 0.0);

        var i: usize = 0;
        while (i < m) : (i += 1) {
            var j: usize = 0;
            while (j < n) : (j += 1) {
                var sum: f32 = 0.0;
                var l: usize = 0;
                while (l < k) : (l += 1) {
                    sum += A[i * k + l] * B[l * n + j];
                }
                C[i * n + j] = sum;
            }
        }
    }
};

pub const BFloat16 = packed struct {
    mantissa: u7,
    exponent: u8,
    sign: u1,

    pub fn fromF32(value: f32) BFloat16 {
        const bits = @as(u32, @bitCast(value));
        return BFloat16{
            .sign = @intCast((bits >> 31) & 1),
            .exponent = @intCast((bits >> 23) & 0xFF),
            .mantissa = @intCast((bits >> 16) & 0x7F),
        };
    }

    pub fn toF32(self: BFloat16) f32 {
        const sign: u32 = @as(u32, self.sign) << 31;
        const exponent: u32 = @as(u32, self.exponent) << 23;
        const mantissa: u32 = @as(u32, self.mantissa) << 16;
        const bits = sign | exponent | mantissa;
        return @bitCast(bits);
    }
};

pub const CompressedArray = struct {
     []BFloat16,
    original_length: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, original: []const f32) !CompressedArray {
        const data = try allocator.alloc(BFloat16, original.len);
        for (original, 0..) |value, i| {
            data[i] = BFloat16.fromF32(value);
        }
        return CompressedArray{
            .data = data,
            .original_length = original.len,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *CompressedArray) void {
        self.allocator.free(self.data);
    }

    pub fn decompress(self: *const CompressedArray, allocator: std.mem.Allocator) ![]f32 {
        const result = try allocator.alloc(f32, self.original_length);
        for (self.data, 0..) |bf16, i| {
            result[i] = bf16.toF32();
        }
        return result;
    }

    pub fn getCompressionRatio(self: *const CompressedArray) f32 {
        const compressed_size = self.data.len * @sizeOf(BFloat16);
        const original_size = self.original_length * @sizeOf(f32);
        return @as(f32, @floatFromInt(compressed_size)) / @as(f32, @floatFromInt(original_size));
    }
};

pub const PrefetchStrategy = struct {
    pub fn prefetchRead(ptr: [*]const u8) void {
        if (builtin.cpu.arch == .x86_64) {
            asm volatile ("prefetcht0 (%[ptr])"
                :
                : [ptr] "r" (ptr),
            );
        }
    }

    pub fn prefetchWrite(ptr: [*]u8) void {
        if (builtin.cpu.arch == .x86_64) {
            asm volatile ("prefetchw (%[ptr])"
                :
                : [ptr] "r" (ptr),
            );
        }
    }
};
