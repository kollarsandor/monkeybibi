const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const bio_lib = b.addStaticLibrary(.{
        .name = "bioinformatics",
        .root_source_file = .{ .path = "bioinformatics.zig" },
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(bio_lib);

    const mem_lib = b.addStaticLibrary(.{
        .name = "memory",
        .root_source_file = .{ .path = "memory.zig" },
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(mem_lib);

    const exe = b.addExecutable(.{
        .name = "jaded-zig",
        .root_source_file = .{ .path = "main.zig" },
        .target = target,
        .optimize = optimize,
    });
    
    exe.linkLibrary(bio_lib);
    exe.linkLibrary(mem_lib);
    
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the Zig bioinformatics engine");
    run_step.dependOn(&run_cmd.step);

    const tests = b.addTest(.{
        .root_source_file = .{ .path = "main.zig" },
        .target = target,
        .optimize = optimize,
    });

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&tests.step);
}
