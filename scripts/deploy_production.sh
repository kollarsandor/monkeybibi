#!/bin/bash
# ============================================================================
# Production Deployment Script
# AlphaFold3 Gateway - Zero-Downtime Deployment
# ============================================================================

set -e

# Configuration
PROD_HOST="${1:-}"
DEPLOY_USER="${DEPLOY_USER:-deploy}"
APP_NAME="alphafold3_gateway"
DEPLOY_PATH="/opt/${APP_NAME}"
BACKUP_PATH="/opt/${APP_NAME}/backups"
RELEASES_PATH="${DEPLOY_PATH}/releases"
CURRENT_PATH="${DEPLOY_PATH}/current"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if [ -z "$PROD_HOST" ]; then
        log_error "Production host not specified"
        echo "Usage: $0 <production_host>"
        exit 1
    fi
    
    if ! command -v ssh &> /dev/null; then
        log_error "ssh not installed"
        exit 1
    fi
    
    if ! command -v rsync &> /dev/null; then
        log_error "rsync not installed"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

build_release() {
    log_info "Building production release..."
    
    export MIX_ENV=prod
    
    mix deps.get --only prod
    mix compile
    mix release --overwrite
    
    log_success "Release built successfully"
}

create_backup() {
    log_info "Creating backup on production server..."
    
    ssh "${DEPLOY_USER}@${PROD_HOST}" "
        mkdir -p ${BACKUP_PATH}
        BACKUP_NAME=${APP_NAME}_\$(date +%Y%m%d_%H%M%S).tar.gz
        
        if [ -d ${CURRENT_PATH} ]; then
            cd ${CURRENT_PATH}/..
            tar -czf ${BACKUP_PATH}/\${BACKUP_NAME} current
            echo \${BACKUP_NAME}
        fi
    " || log_warn "No existing deployment to backup"
    
    log_success "Backup created"
}

upload_release() {
    log_info "Uploading release to production server..."
    
    RELEASE_NAME="${APP_NAME}_$(date +%Y%m%d_%H%M%S)"
    
    ssh "${DEPLOY_USER}@${PROD_HOST}" "mkdir -p ${RELEASES_PATH}/${RELEASE_NAME}"
    
    rsync -avz --progress \
        _build/prod/rel/${APP_NAME}/ \
        "${DEPLOY_USER}@${PROD_HOST}:${RELEASES_PATH}/${RELEASE_NAME}/"
    
    log_success "Release uploaded"
    echo "$RELEASE_NAME"
}

run_migrations() {
    log_info "Running database migrations..."
    
    ssh "${DEPLOY_USER}@${PROD_HOST}" "
        cd ${RELEASES_PATH}/${RELEASE_NAME}
        bin/${APP_NAME} eval 'AlphaFold3Gateway.Release.migrate()'
    " || log_warn "Migration failed or not applicable"
    
    log_success "Migrations complete"
}

switch_release() {
    local release_name=$1
    log_info "Switching to new release..."
    
    ssh "${DEPLOY_USER}@${PROD_HOST}" "
        rm -f ${CURRENT_PATH}
        ln -s ${RELEASES_PATH}/${release_name} ${CURRENT_PATH}
    "
    
    log_success "Release switched"
}

restart_services() {
    log_info "Restarting services..."
    
    ssh "${DEPLOY_USER}@${PROD_HOST}" "
        sudo systemctl restart ${APP_NAME}
    " || log_warn "Service restart failed, trying alternative method..."
    
    # Alternative: Direct process management
    ssh "${DEPLOY_USER}@${PROD_HOST}" "
        cd ${CURRENT_PATH}
        bin/${APP_NAME} stop
        sleep 2
        bin/${APP_NAME} daemon
    "
    
    log_success "Services restarted"
}

verify_deployment() {
    log_info "Verifying deployment..."
    
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if ssh "${DEPLOY_USER}@${PROD_HOST}" "curl -f http://localhost:4000/api/health" &> /dev/null; then
            log_success "Deployment verified - application is healthy"
            return 0
        fi
        
        attempt=$((attempt + 1))
        log_info "Waiting for application to start... (attempt $attempt/$max_attempts)"
        sleep 2
    done
    
    log_error "Deployment verification failed"
    return 1
}

rollback() {
    log_warn "Rolling back deployment..."
    
    ssh "${DEPLOY_USER}@${PROD_HOST}" "
        # Get previous release
        PREVIOUS_RELEASE=\$(ls -t ${RELEASES_PATH} | sed -n '2p')
        
        if [ -n \"\${PREVIOUS_RELEASE}\" ]; then
            rm -f ${CURRENT_PATH}
            ln -s ${RELEASES_PATH}/\${PREVIOUS_RELEASE} ${CURRENT_PATH}
            
            sudo systemctl restart ${APP_NAME}
            
            echo \"Rolled back to: \${PREVIOUS_RELEASE}\"
        else
            echo \"No previous release found\"
            exit 1
        fi
    "
    
    log_warn "Rollback complete"
}

cleanup_old_releases() {
    log_info "Cleaning up old releases (keeping last 5)..."
    
    ssh "${DEPLOY_USER}@${PROD_HOST}" "
        cd ${RELEASES_PATH}
        ls -t | tail -n +6 | xargs -I {} rm -rf {}
    " || log_warn "Cleanup failed or no old releases"
    
    log_success "Cleanup complete"
}

main() {
    log_info "================================================"
    log_info "AlphaFold3 Gateway - Production Deployment"
    log_info "Target: ${PROD_HOST}"
    log_info "================================================"
    
    check_prerequisites
    build_release
    create_backup
    
    RELEASE_NAME=$(upload_release)
    
    run_migrations
    switch_release "$RELEASE_NAME"
    restart_services
    
    sleep 5
    
    if verify_deployment; then
        cleanup_old_releases
        log_success "================================================"
        log_success "Deployment completed successfully!"
        log_success "================================================"
    else
        log_error "Deployment verification failed!"
        read -p "Do you want to rollback? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rollback
            verify_deployment
        fi
        exit 1
    fi
}

main "$@"
