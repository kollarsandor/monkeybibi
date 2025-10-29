const std = @import("std");

const Sequence = struct {
     []const u8,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, seq: []const u8) !Sequence {
        const data = try allocator.dupe(u8, seq);
        return Sequence{
            .data = data,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Sequence) void {
        self.allocator.free(self.data);
    }

    pub fn validate(self: Sequence) bool {
        for (self.data) |c| {
            if (c < 'A' or c > 'Z') {
                return false;
            }
        }
        return true;
    }

    pub fn len(self: Sequence) usize {
        return self.data.len;
    }
};

const AlignmentResult = struct {
    score: i32,
    identity: f32,
    gaps: usize,
    aligned_seq1: []u8,
    aligned_seq2: []u8,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *AlignmentResult) void {
        self.allocator.free(self.aligned_seq1);
        self.allocator.free(self.aligned_seq2);
    }
};

const SmithWatermanAligner = struct {
    gap_open: i32,
    gap_extend: i32,

    pub fn init(gap_open: i32, gap_extend: i32) SmithWatermanAligner {
        return SmithWatermanAligner{
            .gap_open = gap_open,
            .gap_extend = gap_extend,
        };
    }

    fn scoringMatrix(a: u8, b: u8) i32 {
        if (a == b) return 5;
        return -3;
    }

    pub fn align(self: SmithWatermanAligner, allocator: std.mem.Allocator, seq1: Sequence, seq2: Sequence) !AlignmentResult {
        const m = seq1.len();
        const n = seq2.len();

        const H = try allocator.alloc(i32, (m + 1) * (n + 1));
        defer allocator.free(H);

        @memset(H, 0);

        var max_score: i32 = 0;
        var max_i: usize = 0;
        var max_j: usize = 0;

        var i: usize = 1;
        while (i <= m) : (i += 1) {
            var j: usize = 1;
            while (j <= n) : (j += 1) {
                const match = H[(i - 1) * (n + 1) + (j - 1)] + scoringMatrix(seq1.data[i - 1], seq2.data[j - 1]);
                const delete = H[(i - 1) * (n + 1) + j] + self.gap_extend;
                const insert = H[i * (n + 1) + (j - 1)] + self.gap_extend;

                const score = @max(@max(@max(match, delete), insert), 0);
                H[i * (n + 1) + j] = score;

                if (score > max_score) {
                    max_score = score;
                    max_i = i;
                    max_j = j;
                }
            }
        }

        var aligned1 = std.ArrayList(u8).init(allocator);
        var aligned2 = std.ArrayList(u8).init(allocator);

        i = max_i;
        var j = max_j;
        var gaps: usize = 0;
        var matches: usize = 0;
        var total: usize = 0;

        while (i > 0 and j > 0 and H[i * (n + 1) + j] > 0) {
            const current = H[i * (n + 1) + j];
            const diag = H[(i - 1) * (n + 1) + (j - 1)];
            const up = H[(i - 1) * (n + 1) + j];
            const left = H[i * (n + 1) + (j - 1)];

            if (current == diag + scoringMatrix(seq1.data[i - 1], seq2.data[j - 1])) {
                try aligned1.append(seq1.data[i - 1]);
                try aligned2.append(seq2.data[j - 1]);
                if (seq1.data[i - 1] == seq2.data[j - 1]) {
                    matches += 1;
                }
                total += 1;
                i -= 1;
                j -= 1;
            } else if (current == up + self.gap_extend) {
                try aligned1.append(seq1.data[i - 1]);
                try aligned2.append('-');
                gaps += 1;
                total += 1;
                i -= 1;
            } else {
                try aligned1.append('-');
                try aligned2.append(seq2.data[j - 1]);
                gaps += 1;
                total += 1;
                j -= 1;
            }
        }

        std.mem.reverse(u8, aligned1.items);
        std.mem.reverse(u8, aligned2.items);

        const identity = if (total > 0) @as(f32, @floatFromInt(matches)) / @as(f32, @floatFromInt(total)) else 0.0;

        return AlignmentResult{
            .score = max_score,
            .identity = identity,
            .gaps = gaps,
            .aligned_seq1 = try aligned1.toOwnedSlice(),
            .aligned_seq2 = try aligned2.toOwnedSlice(),
            .allocator = allocator,
        };
    }
};

const NeedlemanWunschAligner = struct {
    gap_penalty: i32,

    pub fn init(gap_penalty: i32) NeedlemanWunschAligner {
        return NeedlemanWunschAligner{
            .gap_penalty = gap_penalty,
        };
    }

    fn scoringMatrix(a: u8, b: u8) i32 {
        if (a == b) return 5;
        return -3;
    }

    pub fn align(self: NeedlemanWunschAligner, allocator: std.mem.Allocator, seq1: Sequence, seq2: Sequence) !AlignmentResult {
        const m = seq1.len();
        const n = seq2.len();

        const F = try allocator.alloc(i32, (m + 1) * (n + 1));
        defer allocator.free(F);

        F[0] = 0;
        var i: usize = 1;
        while (i <= m) : (i += 1) {
            F[i * (n + 1)] = @as(i32, @intCast(i)) * self.gap_penalty;
        }
        var j: usize = 1;
        while (j <= n) : (j += 1) {
            F[j] = @as(i32, @intCast(j)) * self.gap_penalty;
        }

        i = 1;
        while (i <= m) : (i += 1) {
            j = 1;
            while (j <= n) : (j += 1) {
                const match = F[(i - 1) * (n + 1) + (j - 1)] + scoringMatrix(seq1.data[i - 1], seq2.data[j - 1]);
                const delete = F[(i - 1) * (n + 1) + j] + self.gap_penalty;
                const insert = F[i * (n + 1) + (j - 1)] + self.gap_penalty;

                F[i * (n + 1) + j] = @max(@max(match, delete), insert);
            }
        }

        var aligned1 = std.ArrayList(u8).init(allocator);
        var aligned2 = std.ArrayList(u8).init(allocator);

        i = m;
        j = n;
        var gaps: usize = 0;
        var matches: usize = 0;
        var total: usize = 0;

        while (i > 0 or j > 0) {
            if (i > 0 and j > 0) {
                const current = F[i * (n + 1) + j];
                const diag = F[(i - 1) * (n + 1) + (j - 1)] + scoringMatrix(seq1.data[i - 1], seq2.data[j - 1]);

                if (current == diag) {
                    try aligned1.append(seq1.data[i - 1]);
                    try aligned2.append(seq2.data[j - 1]);
                    if (seq1.data[i - 1] == seq2.data[j - 1]) {
                        matches += 1;
                    }
                    total += 1;
                    i -= 1;
                    j -= 1;
                    continue;
                }
            }

            if (i > 0) {
                const up = F[(i - 1) * (n + 1) + j] + self.gap_penalty;
                if (F[i * (n + 1) + j] == up) {
                    try aligned1.append(seq1.data[i - 1]);
                    try aligned2.append('-');
                    gaps += 1;
                    total += 1;
                    i -= 1;
                    continue;
                }
            }

            if (j > 0) {
                try aligned1.append('-');
                try aligned2.append(seq2.data[j - 1]);
                gaps += 1;
                total += 1;
                j -= 1;
            }
        }

        std.mem.reverse(u8, aligned1.items);
        std.mem.reverse(u8, aligned2.items);

        const identity = if (total > 0) @as(f32, @floatFromInt(matches)) / @as(f32, @floatFromInt(total)) else 0.0;
        const final_score = F[m * (n + 1) + n];

        return AlignmentResult{
            .score = final_score,
            .identity = identity,
            .gaps = gaps,
            .aligned_seq1 = try aligned1.toOwnedSlice(),
            .aligned_seq2 = try aligned2.toOwnedSlice(),
            .allocator = allocator,
        };
    }
};

const KMerIndex = struct {
    k: usize,
    index: std.AutoHashMap(u64, std.ArrayList(usize)),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, k: usize) KMerIndex {
        return KMerIndex{
            .k = k,
            .index = std.AutoHashMap(u64, std.ArrayList(usize)).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *KMerIndex) void {
        var it = self.index.valueIterator();
        while (it.next()) |list| {
            list.deinit();
        }
        self.index.deinit();
    }

    fn kmerToHash(kmer: []const u8) u64 {
        var hash: u64 = 0;
        for (kmer) |c| {
            hash = hash *% 31 +% @as(u64, c);
        }
        return hash;
    }

    pub fn buildIndex(self: *KMerIndex, seq: Sequence) !void {
        if (seq.len() < self.k) return;

        var i: usize = 0;
        while (i <= seq.len() - self.k) : (i += 1) {
            const kmer = seq.data[i .. i + self.k];
            const hash = kmerToHash(kmer);

            const result = try self.index.getOrPut(hash);
            if (!result.found_existing) {
                result.value_ptr.* = std.ArrayList(usize).init(self.allocator);
            }
            try result.value_ptr.append(i);
        }
    }
};

const ArenaAllocator = struct {
    child_allocator: std.mem.Allocator,
    buffer: []u8,
    offset: usize,
    alignment: usize,

    pub fn init(child_allocator: std.mem.Allocator, size: usize, alignment: usize) ArenaAllocator {
        return ArenaAllocator{
            .child_allocator = child_allocator,
            .buffer = &[_]u8{},
            .offset = 0,
            .alignment = alignment,
        };
    }

    pub fn deinit(self: *ArenaAllocator) void {
        if (self.buffer.len > 0) {
            self.child_allocator.free(self.buffer);
        }
    }

    pub fn allocator(self: *ArenaAllocator) std.mem.Allocator {
        return std.mem.Allocator{
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

        if (self.buffer.len == 0) {
            self.buffer = self.child_allocator.alloc(u8, 1024 * 1024) catch return null;
        }

        const alignment = @as(usize, 1) << @as(u6, @intCast(ptr_align));
        const aligned_offset = std.mem.alignForward(usize, self.offset, alignment);

        if (aligned_offset + len > self.buffer.len) {
            return null;
        }

        const result = self.buffer[aligned_offset..].ptr;
        self.offset = aligned_offset + len;
        return result;
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
};

const SIMDVector = struct {
    pub fn dotProduct(a: []const f32, b: []const f32) f32 {
        var sum: f32 = 0.0;
        const len = @min(a.len, b.len);

        var i: usize = 0;
        const vec_len = 8;

        while (i + vec_len <= len) : (i += vec_len) {
            const va: @Vector(vec_len, f32) = a[i..][0..vec_len].*;
            const vb: @Vector(vec_len, f32) = b[i..][0..vec_len].*;
            const vmul = va * vb;

            var j: usize = 0;
            while (j < vec_len) : (j += 1) {
                sum += vmul[j];
            }
        }

        while (i < len) : (i += 1) {
            sum += a[i] * b[i];
        }

        return sum;
    }

    pub fn vectorAdd(result: []f32, a: []const f32, b: []const f32) void {
        const len = @min(@min(result.len, a.len), b.len);

        var i: usize = 0;
        const vec_len = 8;

        while (i + vec_len <= len) : (i += vec_len) {
            const va: @Vector(vec_len, f32) = a[i..][0..vec_len].*;
            const vb: @Vector(vec_len, f32) = b[i..][0..vec_len].*;
            const vsum = va + vb;

            var j: usize = 0;
            while (j < vec_len) : (j += 1) {
                result[i + j] = vsum[j];
            }
        }

        while (i < len) : (i += 1) {
            result[i] = a[i] + b[i];
        }
    }

    pub fn vectorScale(result: []f32, a: []const f32, scalar: f32) void {
        const len = @min(result.len, a.len);

        var i: usize = 0;
        const vec_len = 8;

        while (i + vec_len <= len) : (i += vec_len) {
            const va: @Vector(vec_len, f32) = a[i..][0..vec_len].*;
            const vs: @Vector(vec_len, f32) = @splat(scalar);
            const vmul = va * vs;

            var j: usize = 0;
            while (j < vec_len) : (j += 1) {
                result[i + j] = vmul[j];
            }
        }

        while (i < len) : (i += 1) {
            result[i] = a[i] * scalar;
        }
    }

    pub fn matrixMultiply(c: []f32, a: []const f32, b: []const f32, m: usize, n: usize, k: usize) void {
        @memset(c, 0.0);

        var i: usize = 0;
        while (i < m) : (i += 1) {
            var j: usize = 0;
            while (j < n) : (j += 1) {
                var sum: f32 = 0.0;
                var ki: usize = 0;
                while (ki < k) : (ki += 1) {
                    sum += a[i * k + ki] * b[ki * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }
};

const CompressedArray = struct {
     []u16,
    original_len: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, values: []const f32) !CompressedArray {
        const compressed = try allocator.alloc(u16, values.len);

        for (values, 0..) |v, i| {
            compressed[i] = floatToBFloat16(v);
        }

        return CompressedArray{
            .data = compressed,
            .original_len = values.len,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *CompressedArray) void {
        self.allocator.free(self.data);
    }

    fn floatToBFloat16(value: f32) u16 {
        const bits = @as(u32, @bitCast(value));
        return @as(u16, @intCast((bits >> 16) & 0xFFFF));
    }

    fn bfloat16ToFloat(value: u16) f32 {
        const bits = @as(u32, value) << 16;
        return @as(f32, @bitCast(bits));
    }

    pub fn decompress(self: CompressedArray, allocator: std.mem.Allocator) ![]f32 {
        const result = try allocator.alloc(f32, self.original_len);

        for (self.data, 0..) |v, i| {
            result[i] = bfloat16ToFloat(v);
        }

        return result;
    }

    pub fn getCompressionRatio(self: CompressedArray) f32 {
        const original_size = self.original_len * @sizeOf(f32);
        const compressed_size = self.data.len * @sizeOf(u16);
        return @as(f32, @floatFromInt(original_size)) / @as(f32, @floatFromInt(compressed_size));
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("JADED Zig Bioinformatics Engine\n", .{});
    std.debug.print("================================\n\n", .{});

    const seq1_str = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL";

    var seq1 = try Sequence.init(allocator, seq1_str);
    defer seq1.deinit();

    std.debug.print("Sequence validation: {}\n", .{seq1.validate()});
    std.debug.print("Sequence length: {}\n\n", .{seq1.len()});

    const seq2_str = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVY";
    var seq2 = try Sequence.init(allocator, seq2_str);
    defer seq2.deinit();

    std.debug.print("Testing Smith-Waterman alignment:\n", .{});
    var sw_aligner = SmithWatermanAligner.init(-10, -1);
    var sw_result = try sw_aligner.align(allocator, seq1, seq2);
    defer sw_result.deinit();

    std.debug.print("SW Score: {}\n", .{sw_result.score});
    std.debug.print("SW Identity: {d:.2}%\n", .{sw_result.identity * 100.0});
    std.debug.print("SW Gaps: {}\n", .{sw_result.gaps});
    std.debug.print("Aligned sequence 1: {s}\n", .{sw_result.aligned_seq1[0..@min(80, sw_result.aligned_seq1.len)]});
    std.debug.print("Aligned sequence 2: {s}\n\n", .{sw_result.aligned_seq2[0..@min(80, sw_result.aligned_seq2.len)]});

    std.debug.print("Testing Needleman-Wunsch alignment:\n", .{});
    var nw_aligner = NeedlemanWunschAligner.init(-2);
    var nw_result = try nw_aligner.align(allocator, seq1, seq2);
    defer nw_result.deinit();

    std.debug.print("NW Score: {}\n", .{nw_result.score});
    std.debug.print("NW Identity: {d:.2}%\n", .{nw_result.identity * 100.0});
    std.debug.print("NW Gaps: {}\n\n", .{nw_result.gaps});

    std.debug.print("Testing K-mer indexing:\n", .{});
    var kmer_index = KMerIndex.init(allocator, 6);
    defer kmer_index.deinit();

    try kmer_index.buildIndex(seq1);
    std.debug.print("K-mer index built successfully\n\n", .{});

    std.debug.print("Testing memory management:\n", .{});
    var arena = ArenaAllocator.init(allocator, 1024 * 1024, 16);
    defer arena.deinit();

    const arena_allocator = arena.allocator();
    const test_data = try arena_allocator.alloc(f32, 1000);
    for (test_data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i)) * 0.1;
    }
    std.debug.print("Arena allocated {} floats\n", .{test_data.len});

    std.debug.print("\nTesting SIMD operations:\n", .{});
    const size: usize = 1000;
    const vec_a = try allocator.alloc(f32, size);
    defer allocator.free(vec_a);
    const vec_b = try allocator.alloc(f32, size);
    defer allocator.free(vec_b);
    const vec_result = try allocator.alloc(f32, size);
    defer allocator.free(vec_result);

    for (vec_a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i));
    for (vec_b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 2.0;

    const dot = SIMDVector.dotProduct(vec_a, vec_b);
    std.debug.print("Dot product: {d:.2}\n", .{dot});

    SIMDVector.vectorAdd(vec_result, vec_a, vec_b);
    std.debug.print("Vector addition completed\n", .{});

    SIMDVector.vectorScale(vec_result, vec_a, 3.14);
    std.debug.print("Vector scaling completed\n", .{});

    std.debug.print("\nTesting BFloat16 compression:\n", .{});
    const original_data = try allocator.alloc(f32, 100);
    defer allocator.free(original_data);

    for (original_data, 0..) |*v, i| {
        v.* = @sin(@as(f32, @floatFromInt(i)) * 0.1) * 1000.0;
    }

    var compressed = try CompressedArray.init(allocator, original_data);
    defer compressed.deinit();

    std.debug.print("Compression ratio: {d:.2}\n", .{compressed.getCompressionRatio()});

    const decompressed = try compressed.decompress(allocator);
    defer allocator.free(decompressed);

    var max_error: f32 = 0.0;
    for (original_data, decompressed) |orig, decomp| {
        const error_val = @abs(orig - decomp);
        if (error_val > max_error) max_error = error_val;
    }
    std.debug.print("Max decompression error: {d:.4}\n", .{max_error});

    std.debug.print("\nTesting matrix operations:\n", .{});
    const m: usize = 10;
    const n: usize = 10;
    const k: usize = 10;

    const mat_a = try allocator.alloc(f32, m * k);
    defer allocator.free(mat_a);
    const mat_b = try allocator.alloc(f32, k * n);
    defer allocator.free(mat_b);
    const mat_c = try allocator.alloc(f32, m * n);
    defer allocator.free(mat_c);

    for (mat_a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 10));
    for (mat_b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 10));

    SIMDVector.matrixMultiply(mat_c, mat_a, mat_b, m, n, k);
    std.debug.print("Matrix multiplication ({} x {} x {}) completed\n", .{ m, k, n });
    std.debug.print("Result[0,0]: {d:.2}\n", .{mat_c[0]});
    std.debug.print("Result[5,5]: {d:.2}\n", .{mat_c[5 * n + 5]});

    std.debug.print("\nAll bioinformatics and memory tests completed successfully!\n", .{});
}
