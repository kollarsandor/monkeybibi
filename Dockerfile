# ============================================================================
# Multi-stage Dockerfile for Production Deployment
# AlphaFold3 Gateway - Full Production Ready
# ============================================================================

# ============================================================================
# Stage 1: Elixir Builder
# ============================================================================
FROM hexpm/elixir:1.16.0-erlang-26.2.1-alpine-3.19.0 AS elixir-builder

# Install build dependencies
RUN apk add --no-cache \
    build-base \
    git \
    nodejs \
    npm \
    python3 \
    py3-pip

# Set working directory
WORKDIR /app

# Install hex and rebar
RUN mix local.hex --force && \
    mix local.rebar --force

# Copy mix files
COPY mix.exs mix.lock ./

# Install dependencies
ENV MIX_ENV=prod
RUN mix deps.get --only prod && \
    mix deps.compile

# Copy application source
COPY config config
COPY lib lib
COPY priv priv

# Compile the application
RUN mix compile

# Build release
RUN mix release

# ============================================================================
# Stage 2: Julia Builder
# ============================================================================
FROM julia:1.10-alpine AS julia-builder

WORKDIR /julia

# Copy Julia project files
COPY Project.toml Manifest.toml ./
COPY main.jl ./
COPY model/ model/

# Precompile Julia packages
RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# ============================================================================
# Stage 3: Rust Builder
# ============================================================================
FROM rust:1.75-alpine AS rust-builder

# Install build dependencies
RUN apk add --no-cache \
    musl-dev \
    openssl-dev \
    openssl-libs-static \
    pkgconfig

WORKDIR /rust

# Copy Rust project files
COPY binding/Cargo.toml binding/Cargo.lock ./
COPY binding/verification_orchestrator.rs ./src/

# Build Rust components
RUN cargo build --release

# ============================================================================
# Stage 4: Zig Builder
# ============================================================================
FROM alpine:3.19 AS zig-builder

# Install Zig
RUN apk add --no-cache wget xz tar && \
    wget https://ziglang.org/download/0.12.0/zig-linux-x86_64-0.12.0.tar.xz && \
    tar -xf zig-linux-x86_64-0.12.0.tar.xz && \
    mv zig-linux-x86_64-0.12.0 /usr/local/zig

ENV PATH="/usr/local/zig:${PATH}"

WORKDIR /zig

# Copy Zig source
COPY native/ native/

# Build Zig components
RUN cd native && zig build -Doptimize=ReleaseFast

# ============================================================================
# Stage 5: Python Builder
# ============================================================================
FROM python:3.11-slim AS python-builder

WORKDIR /python

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements
COPY modal_wrapper.py .
COPY deploy.py .
COPY dragonfly_config.py .
COPY quantum/quantum_noise_integration.py quantum/
COPY hardware/spintronics_simulator.py hardware/
COPY hardware/photonics_simulator.py hardware/
COPY hardware/accelerator_integration.py hardware/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    torch>=2.0.0 \
    jax>=0.4.0 \
    jaxlib>=0.4.0 \
    dm-haiku>=0.0.10 \
    biopython>=1.81 \
    pandas>=2.0.0 \
    matplotlib>=3.7.0 \
    scikit-learn>=1.3.0 \
    httpx>=0.24.0 \
    fastapi>=0.100.0 \
    uvicorn>=0.23.0 \
    pydantic>=2.0.0 \
    redis>=4.6.0 \
    celery>=5.3.0 \
    psycopg2-binary>=2.9.0 \
    sqlalchemy>=2.0.0 \
    modal>=0.55.0

# ============================================================================
# Stage 6: Final Production Runtime
# ============================================================================
FROM alpine:3.19

# Install runtime dependencies
RUN apk add --no-cache \
    bash \
    openssl \
    ncurses-libs \
    libstdc++ \
    libgcc \
    ca-certificates \
    tzdata \
    curl \
    htop \
    python3 \
    py3-pip \
    libgfortran \
    openblas \
    lapack

# Create non-root user
RUN addgroup -g 1000 alphafold && \
    adduser -D -u 1000 -G alphafold alphafold

# Set working directory
WORKDIR /app

# Copy Elixir release from builder
COPY --from=elixir-builder --chown=alphafold:alphafold /app/_build/prod/rel/alphafold3_gateway ./elixir

# Copy Julia environment
COPY --from=julia-builder --chown=alphafold:alphafold /julia ./julia
RUN apk add --no-cache julia

# Copy Rust binaries
COPY --from=rust-builder --chown=alphafold:alphafold /rust/target/release/verification_orchestrator ./bin/

# Copy Zig binaries
COPY --from=zig-builder --chown=alphafold:alphafold /zig/native/zig-out/bin/* ./bin/

# Copy Python environment
COPY --from=python-builder --chown=alphafold:alphafold /python ./python
COPY --from=python-builder --chown=alphafold:alphafold /usr/local/lib/python3.11/site-packages /usr/lib/python3.11/site-packages

# Copy additional files
COPY --chown=alphafold:alphafold priv priv
COPY --chown=alphafold:alphafold config config
COPY --chown=alphafold:alphafold quantum quantum
COPY --chown=alphafold:alphafold hardware hardware
COPY --chown=alphafold:alphafold metaprog metaprog

# Create necessary directories
RUN mkdir -p /app/priv/uploads && \
    mkdir -p /app/logs && \
    mkdir -p /app/tmp && \
    chown -R alphafold:alphafold /app

# Switch to non-root user
USER alphafold

# Expose ports
EXPOSE 4000 5000 6000 7000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:4000/api/health || exit 1

# Set environment variables
ENV MIX_ENV=prod \
    PORT=4000 \
    LANG=C.UTF-8 \
    HOME=/app

# Start script
COPY --chown=alphafold:alphafold docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["start"]
