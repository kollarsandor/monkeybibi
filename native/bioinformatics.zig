const std = @import("std");
const builtin = @import("builtin");

pub const AminoAcid = enum(u8) {
    Ala = 'A', Arg = 'R', Asn = 'N', Asp = 'D', Cys = 'C',
    Gln = 'Q', Glu = 'E', Gly = 'G', His = 'H', Ile = 'I',
    Leu = 'L', Lys = 'K', Met = 'M', Phe = 'F', Pro = 'P',
    Ser = 'S', Thr = 'T', Trp = 'W', Tyr = 'Y', Val = 'V',
    Unknown = 'X',
};

pub const ScoringMatrix = struct {
    blosum62: [20][20]i32,

    pub fn init() ScoringMatrix {
        return ScoringMatrix{
            .blosum62 = [20][20]i32{
                [20]i32{4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0},
                [20]i32{-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3},
                [20]i32{-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3},
                [20]i32{-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3},
                [20]i32{0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1},
                [20]i32{-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2},
                [20]i32{-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2},
                [20]i32{0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3},
                [20]i32{-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3},
                [20]i32{-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3},
                [20]i32{-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1},
                [20]i32{-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2},
                [20]i32{-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1},
                [20]i32{-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1},
                [20]i32{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2},
                [20]i32{1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2},
                [20]i32{0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0},
                [20]i32{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3},
                [20]i32{-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1},
                [20]i32{0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4},
            },
        };
    }

    pub fn getScore(self: *const ScoringMatrix, a: AminoAcid, b: AminoAcid) i32 {
        const idx_a = @intFromEnum(a) % 20;
        const idx_b = @intFromEnum(b) % 20;
        return self.blosum62[idx_a][idx_b];
    }
};

pub const Sequence = struct {
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

    pub fn len(self: *const Sequence) usize {
        return self.data.len;
    }

    pub fn validate(self: *const Sequence) bool {
        for (self.data) |c| {
            const valid = switch (c) {
                'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X' => true,
                else => false,
            };
            if (!valid) return false;
        }
        return true;
    }
};

pub const AlignmentResult = struct {
    score: i32,
    aligned_seq1: []u8,
    aligned_seq2: []u8,
    identity: f32,
    similarity: f32,
    gaps: usize,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *AlignmentResult) void {
        self.allocator.free(self.aligned_seq1);
        self.allocator.free(self.aligned_seq2);
    }
};

pub const SmithWatermanAligner = struct {
    scoring_matrix: ScoringMatrix,
    gap_open: i32,
    gap_extend: i32,

    pub fn init(gap_open: i32, gap_extend: i32) SmithWatermanAligner {
        return SmithWatermanAligner{
            .scoring_matrix = ScoringMatrix.init(),
            .gap_open = gap_open,
            .gap_extend = gap_extend,
        };
    }

    pub fn align(self: *SmithWatermanAligner, allocator: std.mem.Allocator, seq1: Sequence, seq2: Sequence) !AlignmentResult {
        const m = seq1.len();
        const n = seq2.len();

        const H = try allocator.alloc([]i32, m + 1);
        defer {
            for (H) |row| allocator.free(row);
            allocator.free(H);
        }

        for (H) |*row| {
            row.* = try allocator.alloc(i32, n + 1);
            @memset(row.*, 0);
        }

        var max_score: i32 = 0;
        var max_i: usize = 0;
        var max_j: usize = 0;

        var i: usize = 1;
        while (i <= m) : (i += 1) {
            var j: usize = 1;
            while (j <= n) : (j += 1) {
                const aa1: AminoAcid = @enumFromInt(seq1.data[i - 1]);
                const aa2: AminoAcid = @enumFromInt(seq2.data[j - 1]);
                const match_score = self.scoring_matrix.getScore(&aa1, &aa2);

                const diag = H[i - 1][j - 1] + match_score;
                const up = H[i - 1][j] + self.gap_open;
                const left = H[i][j - 1] + self.gap_open;

                H[i][j] = @max(@max(@max(diag, up), left), 0);

                if (H[i][j] > max_score) {
                    max_score = H[i][j];
                    max_i = i;
                    max_j = j;
                }
            }
        }

        var aligned1 = std.ArrayList(u8).init(allocator);
        var aligned2 = std.ArrayList(u8).init(allocator);

        i = max_i;
        var j = max_j;
        var matches: usize = 0;
        var gaps: usize = 0;

        while (i > 0 and j > 0 and H[i][j] > 0) {
            const aa1: AminoAcid = @enumFromInt(seq1.data[i - 1]);
            const aa2: AminoAcid = @enumFromInt(seq2.data[j - 1]);
            const score = H[i][j];
            const diag = H[i - 1][j - 1];
            const up = H[i - 1][j];
            const left = H[i][j - 1];

            if (i > 0 and j > 0 and score == diag + self.scoring_matrix.getScore(&aa1, &aa2)) {
                try aligned1.insert(0, seq1.data[i - 1]);
                try aligned2.insert(0, seq2.data[j - 1]);
                if (seq1.data[i - 1] == seq2.data[j - 1]) matches += 1;
                i -= 1;
                j -= 1;
            } else if (i > 0 and score == up + self.gap_open) {
                try aligned1.insert(0, seq1.data[i - 1]);
                try aligned2.insert(0, '-');
                gaps += 1;
                i -= 1;
            } else {
                try aligned1.insert(0, '-');
                try aligned2.insert(0, seq2.data[j - 1]);
                gaps += 1;
                j -= 1;
            }
        }

        const align_len = aligned1.items.len;
        const identity = if (align_len > 0) @as(f32, @floatFromInt(matches)) / @as(f32, @floatFromInt(align_len)) else 0.0;

        return AlignmentResult{
            .score = max_score,
            .aligned_seq1 = try aligned1.toOwnedSlice(),
            .aligned_seq2 = try aligned2.toOwnedSlice(),
            .identity = identity,
            .similarity = identity,
            .gaps = gaps,
            .allocator = allocator,
        };
    }
};

pub const NeedlemanWunschAligner = struct {
    scoring_matrix: ScoringMatrix,
    gap_penalty: i32,

    pub fn init(gap_penalty: i32) NeedlemanWunschAligner {
        return NeedlemanWunschAligner{
            .scoring_matrix = ScoringMatrix.init(),
            .gap_penalty = gap_penalty,
        };
    }

    pub fn align(self: *NeedlemanWunschAligner, allocator: std.mem.Allocator, seq1: Sequence, seq2: Sequence) !AlignmentResult {
        const m = seq1.len();
        const n = seq2.len();

        const F = try allocator.alloc([]i32, m + 1);
        defer {
            for (F) |row| allocator.free(row);
            allocator.free(F);
        }

        for (F, 0..) |*row, idx| {
            row.* = try allocator.alloc(i32, n + 1);
            row.*[0] = @as(i32, @intCast(idx)) * self.gap_penalty;
        }

        for (0..n + 1) |jdx| {
            F[0][jdx] = @as(i32, @intCast(jdx)) * self.gap_penalty;
        }

        var i: usize = 1;
        while (i <= m) : (i += 1) {
            var j: usize = 1;
            while (j <= n) : (j += 1) {
                const aa1: AminoAcid = @enumFromInt(seq1.data[i - 1]);
                const aa2: AminoAcid = @enumFromInt(seq2.data[j - 1]);
                const match_score = self.scoring_matrix.getScore(&aa1, &aa2);

                const diag = F[i - 1][j - 1] + match_score;
                const up = F[i - 1][j] + self.gap_penalty;
                const left = F[i][j - 1] + self.gap_penalty;

                F[i][j] = @max(@max(diag, up), left);
            }
        }

        var aligned1 = std.ArrayList(u8).init(allocator);
        var aligned2 = std.ArrayList(u8).init(allocator);

        i = m;
        var j = n;
        var matches: usize = 0;
        var gaps: usize = 0;

        while (i > 0 or j > 0) {
            if (i > 0 and j > 0) {
                const aa1: AminoAcid = @enumFromInt(seq1.data[i - 1]);
                const aa2: AminoAcid = @enumFromInt(seq2.data[j - 1]);
                const score = F[i][j];
                const diag = F[i - 1][j - 1];
                const up = F[i - 1][j];
                const left = F[i][j - 1];

                if (score == diag + self.scoring_matrix.getScore(&aa1, &aa2)) {
                    try aligned1.insert(0, seq1.data[i - 1]);
                    try aligned2.insert(0, seq2.data[j - 1]);
                    if (seq1.data[i - 1] == seq2.data[j - 1]) matches += 1;
                    i -= 1;
                    j -= 1;
                } else if (score == up + self.gap_penalty) {
                    try aligned1.insert(0, seq1.data[i - 1]);
                    try aligned2.insert(0, '-');
                    gaps += 1;
                    i -= 1;
                } else {
                    try aligned1.insert(0, '-');
                    try aligned2.insert(0, seq2.data[j - 1]);
                    gaps += 1;
                    j -= 1;
                }
            } else if (i > 0) {
                try aligned1.insert(0, seq1.data[i - 1]);
                try aligned2.insert(0, '-');
                gaps += 1;
                i -= 1;
            } else {
                try aligned1.insert(0, '-');
                try aligned2.insert(0, seq2.data[j - 1]);
                gaps += 1;
                j -= 1;
            }
        }

        const align_len = aligned1.items.len;
        const identity = if (align_len > 0) @as(f32, @floatFromInt(matches)) / @as(f32, @floatFromInt(align_len)) else 0.0;

        return AlignmentResult{
            .score = F[m][n],
            .aligned_seq1 = try aligned1.toOwnedSlice(),
            .aligned_seq2 = try aligned2.toOwnedSlice(),
            .identity = identity,
            .similarity = identity,
            .gaps = gaps,
            .allocator = allocator,
        };
    }
};

pub const KMerIndex = struct {
    k: usize,
    index: std.StringHashMap(std.ArrayList(usize)),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, k: usize) KMerIndex {
        return KMerIndex{
            .k = k,
            .index = std.StringHashMap(std.ArrayList(usize)).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *KMerIndex) void {
        var it = self.index.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.index.deinit();
    }

    pub fn buildIndex(self: *KMerIndex, seq: Sequence) !void {
        if (seq.len() < self.k) return;

        var i: usize = 0;
        while (i <= seq.len() - self.k) : (i += 1) {
            const kmer = seq.data[i .. i + self.k];
            const kmer_copy = try self.allocator.dupe(u8, kmer);

            const result = try self.index.getOrPut(kmer_copy);
            if (!result.found_existing) {
                result.value_ptr.* = std.ArrayList(usize).init(self.allocator);
            } else {
                self.allocator.free(kmer_copy);
            }

            try result.value_ptr.append(i);
        }
    }

    pub fn findMatches(self: *const KMerIndex, kmer: []const u8) ?std.ArrayList(usize) {
        return self.index.get(kmer);
    }
};

pub fn calculateGCContent(seq: Sequence) f32 {
    var gc_count: usize = 0;
    for (seq.data) |c| {
        if (c == 'G' or c == 'C') gc_count += 1;
    }
    return @as(f32, @floatFromInt(gc_count)) / @as(f32, @floatFromInt(seq.len()));
}

pub fn reverseComplement(allocator: std.mem.Allocator, seq: Sequence) !Sequence {
    var result = try allocator.alloc(u8, seq.len());
    var i: usize = 0;
    while (i < seq.len()) : (i += 1) {
        result[seq.len() - 1 - i] = switch (seq.data[i]) {
            'A' => 'T',
            'T' => 'A',
            'G' => 'C',
            'C' => 'G',
            else => seq.data[i],
        };
    }
    return Sequence{
        .data = result,
        .allocator = allocator,
    };
}

pub fn translateDNA(allocator: std.mem.Allocator, seq: Sequence) !Sequence {
    const codon_table = std.StaticStringMap(u8).initComptime(.{
        .{ "TTT", 'F' }, .{ "TTC", 'F' }, .{ "TTA", 'L' }, .{ "TTG", 'L' },
        .{ "TCT", 'S' }, .{ "TCC", 'S' }, .{ "TCA", 'S' }, .{ "TCG", 'S' },
        .{ "TAT", 'Y' }, .{ "TAC", 'Y' }, .{ "TAA", '*' }, .{ "TAG", '*' },
        .{ "TGT", 'C' }, .{ "TGC", 'C' }, .{ "TGA", '*' }, .{ "TGG", 'W' },
        .{ "CTT", 'L' }, .{ "CTC", 'L' }, .{ "CTA", 'L' }, .{ "CTG", 'L' },
        .{ "CCT", 'P' }, .{ "CCC", 'P' }, .{ "CCA", 'P' }, .{ "CCG", 'P' },
        .{ "CAT", 'H' }, .{ "CAC", 'H' }, .{ "CAA", 'Q' }, .{ "CAG", 'Q' },
        .{ "CGT", 'R' }, .{ "CGC", 'R' }, .{ "CGA", 'R' }, .{ "CGG", 'R' },
        .{ "ATT", 'I' }, .{ "ATC", 'I' }, .{ "ATA", 'I' }, .{ "ATG", 'M' },
        .{ "ACT", 'T' }, .{ "ACC", 'T' }, .{ "ACA", 'T' }, .{ "ACG", 'T' },
        .{ "AAT", 'N' }, .{ "AAC", 'N' }, .{ "AAA", 'K' }, .{ "AAG", 'K' },
        .{ "AGT", 'S' }, .{ "AGC", 'S' }, .{ "AGA", 'R' }, .{ "AGG", 'R' },
        .{ "GTT", 'V' }, .{ "GTC", 'V' }, .{ "GTA", 'V' }, .{ "GTG", 'V' },
        .{ "GCT", 'A' }, .{ "GCC", 'A' }, .{ "GCA", 'A' }, .{ "GCG", 'A' },
        .{ "GAT", 'D' }, .{ "GAC", 'D' }, .{ "GAA", 'E' }, .{ "GAG", 'E' },
        .{ "GGT", 'G' }, .{ "GGC", 'G' }, .{ "GGA", 'G' }, .{ "GGG", 'G' },
    });

    var protein = std.ArrayList(u8).init(allocator);

    var i: usize = 0;
    while (i + 2 < seq.len()) : (i += 3) {
        const codon = seq.data[i .. i + 3];
        if (codon_table.get(codon)) |aa| {
            if (aa == '*') break;
            try protein.append(aa);
        } else {
            try protein.append('X');
        }
    }

    return Sequence{
        .data = try protein.toOwnedSlice(),
        .allocator = allocator,
    };
}
