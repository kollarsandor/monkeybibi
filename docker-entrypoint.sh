#!/bin/bash
# ============================================================================
# Docker Entrypoint Script - Production Ready
# AlphaFold3 Gateway Multi-Backend Startup
# ============================================================================

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# Environment Validation
# ============================================================================
validate_environment() {
    log_info "Validating environment variables..."
    
    local required_vars=("SECRET_KEY_BASE" "PHX_HOST")
    local missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -gt 0 ]; then
        log_error "Missing required environment variables: ${missing_vars[*]}"
        log_info "Generating defaults for development..."
        
        if [ -z "$SECRET_KEY_BASE" ]; then
            export SECRET_KEY_BASE=$(openssl rand -base64 64 | tr -d '\n')
            log_warn "Generated SECRET_KEY_BASE (DO NOT USE IN PRODUCTION)"
        fi
        
        if [ -z "$PHX_HOST" ]; then
            export PHX_HOST="localhost"
            log_warn "Using default PHX_HOST=localhost"
        fi
    fi
    
    log_success "Environment validation complete"
}

# ============================================================================
# Database Migration (if configured)
# ============================================================================
run_migrations() {
    if [ -n "$DATABASE_URL" ]; then
        log_info "Running database migrations..."
        cd /app/elixir && bin/alphafold3_gateway eval "AlphaFold3Gateway.Release.migrate()"
        log_success "Migrations complete"
    else
        log_info "No DATABASE_URL configured, skipping migrations"
    fi
}

# ============================================================================
# Start Julia Backend
# ============================================================================
start_julia_backend() {
    log_info "Starting Julia backend server..."
    
    cd /app/julia
    julia --project=. --threads=auto model/http_server.jl &
    JULIA_PID=$!
    
    log_success "Julia backend started (PID: $JULIA_PID)"
}

# ============================================================================
# Start Python Backend
# ============================================================================
start_python_backend() {
    log_info "Starting Python backend server..."
    
    cd /app/python
    python3 -m uvicorn modal_wrapper:app --host 0.0.0.0 --port 7000 &
    PYTHON_PID=$!
    
    log_success "Python backend started (PID: $PYTHON_PID)"
}

# ============================================================================
# Start Rust Verification Service
# ============================================================================
start_rust_verification() {
    log_info "Starting Rust verification orchestrator..."
    
    /app/bin/verification_orchestrator &
    RUST_PID=$!
    
    log_success "Rust verification service started (PID: $RUST_PID)"
}

# ============================================================================
# Start Quantum Server
# ============================================================================
start_quantum_server() {
    if [ -f "/app/quantum/quantum_server.cr" ]; then
        log_info "Quantum server found, starting..."
        # Note: Crystal runtime needed, skip if not available
        log_warn "Crystal runtime not installed, skipping quantum server"
    fi
}

# ============================================================================
# Start Elixir/Phoenix Application
# ============================================================================
start_phoenix_app() {
    log_info "Starting Phoenix application..."
    
    cd /app/elixir
    exec bin/alphafold3_gateway start
}

# ============================================================================
# Health Checks
# ============================================================================
wait_for_backends() {
    log_info "Waiting for backends to be ready..."
    
    local max_attempts=30
    local attempt=0
    
    # Wait for Julia backend
    while ! curl -s http://localhost:6000/health > /dev/null 2>&1; do
        attempt=$((attempt + 1))
        if [ $attempt -ge $max_attempts ]; then
            log_error "Julia backend failed to start within timeout"
            return 1
        fi
        log_info "Waiting for Julia backend... (attempt $attempt/$max_attempts)"
        sleep 2
    done
    log_success "Julia backend is ready"
    
    # Wait for Python backend
    attempt=0
    while ! curl -s http://localhost:7000/health > /dev/null 2>&1; do
        attempt=$((attempt + 1))
        if [ $attempt -ge $max_attempts ]; then
            log_error "Python backend failed to start within timeout"
            return 1
        fi
        log_info "Waiting for Python backend... (attempt $attempt/$max_attempts)"
        sleep 2
    done
    log_success "Python backend is ready"
    
    log_success "All backends are operational"
}

# ============================================================================
# Graceful Shutdown Handler
# ============================================================================
shutdown() {
    log_info "Received shutdown signal, gracefully stopping services..."
    
    # Kill all child processes
    if [ -n "$JULIA_PID" ]; then
        log_info "Stopping Julia backend..."
        kill -TERM "$JULIA_PID" 2>/dev/null || true
    fi
    
    if [ -n "$PYTHON_PID" ]; then
        log_info "Stopping Python backend..."
        kill -TERM "$PYTHON_PID" 2>/dev/null || true
    fi
    
    if [ -n "$RUST_PID" ]; then
        log_info "Stopping Rust verification service..."
        kill -TERM "$RUST_PID" 2>/dev/null || true
    fi
    
    log_success "All services stopped gracefully"
    exit 0
}

trap shutdown SIGTERM SIGINT

# ============================================================================
# Main Execution
# ============================================================================
main() {
    log_info "=========================================="
    log_info "AlphaFold3 Gateway - Production Startup"
    log_info "=========================================="
    
    # Validate environment
    validate_environment
    
    # Run migrations if needed
    run_migrations
    
    case "${1:-start}" in
        start)
            log_info "Starting all services..."
            
            # Start backend services
            start_julia_backend
            start_python_backend
            start_rust_verification
            start_quantum_server
            
            # Wait for backends to be ready
            sleep 5
            wait_for_backends
            
            # Start Phoenix application (blocking)
            log_info "All backends ready, starting main application..."
            start_phoenix_app
            ;;
            
        phoenix-only)
            log_info "Starting Phoenix application only..."
            start_phoenix_app
            ;;
            
        julia-only)
            log_info "Starting Julia backend only..."
            start_julia_backend
            wait
            ;;
            
        python-only)
            log_info "Starting Python backend only..."
            start_python_backend
            wait
            ;;
            
        shell)
            log_info "Starting interactive shell..."
            exec /bin/bash
            ;;
            
        migrate)
            log_info "Running migrations only..."
            run_migrations
            log_success "Migrations complete, exiting"
            ;;
            
        *)
            log_error "Unknown command: $1"
            log_info "Available commands: start, phoenix-only, julia-only, python-only, shell, migrate"
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"
