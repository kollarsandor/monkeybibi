# ============================================================================
# Makefile - Production Ready Build and Deployment
# AlphaFold3 Gateway
# ============================================================================

.PHONY: help build start stop restart logs clean test deps compile release deploy health

# Default target
.DEFAULT_GOAL := help

# Variables
DOCKER_COMPOSE = docker-compose
MIX = mix
ELIXIR_VERSION = 1.16.0
OTP_VERSION = 26

# Color output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

##@ General

help: ## Display this help message
	@echo "$(BLUE)AlphaFold3 Gateway - Production Commands$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make $(YELLOW)<target>$(NC)\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(GREEN)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development

deps: ## Install all dependencies
	@echo "$(BLUE)Installing Elixir dependencies...$(NC)"
	$(MIX) deps.get
	@echo "$(GREEN)Dependencies installed successfully$(NC)"

compile: deps ## Compile the application
	@echo "$(BLUE)Compiling application...$(NC)"
	$(MIX) compile
	@echo "$(GREEN)Compilation complete$(NC)"

format: ## Format code
	@echo "$(BLUE)Formatting code...$(NC)"
	$(MIX) format
	@echo "$(GREEN)Code formatted$(NC)"

lint: ## Run linter
	@echo "$(BLUE)Running linter...$(NC)"
	$(MIX) credo --strict
	@echo "$(GREEN)Linting complete$(NC)"

test: ## Run tests
	@echo "$(BLUE)Running tests...$(NC)"
	$(MIX) test
	@echo "$(GREEN)Tests complete$(NC)"

test-coverage: ## Run tests with coverage
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(MIX) coveralls.html
	@echo "$(GREEN)Coverage report generated at cover/excoveralls.html$(NC)"

##@ Local Development

dev: ## Start development server
	@echo "$(BLUE)Starting development server...$(NC)"
	$(MIX) phx.server

dev-iex: ## Start development server with IEx
	@echo "$(BLUE)Starting IEx with Phoenix...$(NC)"
	iex -S mix phx.server

console: ## Start IEx console
	@echo "$(BLUE)Starting IEx console...$(NC)"
	iex -S mix

##@ Docker Operations

docker-build: ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	$(DOCKER_COMPOSE) build
	@echo "$(GREEN)Docker images built successfully$(NC)"

docker-start: ## Start all Docker services
	@echo "$(BLUE)Starting Docker services...$(NC)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)Services started successfully$(NC)"
	@$(MAKE) health

docker-stop: ## Stop all Docker services
	@echo "$(BLUE)Stopping Docker services...$(NC)"
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)Services stopped$(NC)"

docker-restart: docker-stop docker-start ## Restart all Docker services

docker-logs: ## View logs from all services
	$(DOCKER_COMPOSE) logs -f

docker-logs-gateway: ## View gateway logs only
	$(DOCKER_COMPOSE) logs -f gateway

docker-logs-julia: ## View Julia backend logs
	$(DOCKER_COMPOSE) logs -f julia-backend

docker-logs-python: ## View Python backend logs
	$(DOCKER_COMPOSE) logs -f python-backend

docker-shell: ## Open shell in gateway container
	$(DOCKER_COMPOSE) exec gateway /bin/bash

docker-clean: ## Remove all containers, volumes, and images
	@echo "$(YELLOW)Cleaning Docker resources...$(NC)"
	$(DOCKER_COMPOSE) down -v --rmi all
	@echo "$(GREEN)Cleanup complete$(NC)"

##@ Production

release: ## Build production release
	@echo "$(BLUE)Building production release...$(NC)"
	MIX_ENV=prod $(MIX) deps.get --only prod
	MIX_ENV=prod $(MIX) compile
	MIX_ENV=prod $(MIX) assets.deploy
	MIX_ENV=prod $(MIX) release --overwrite
	@echo "$(GREEN)Release built successfully$(NC)"

release-run: release ## Run production release locally
	@echo "$(BLUE)Starting production release...$(NC)"
	_build/prod/rel/alphafold3_gateway/bin/alphafold3_gateway start

prod-deploy: ## Deploy to production (requires configuration)
	@echo "$(BLUE)Deploying to production...$(NC)"
	@if [ -z "$(PROD_HOST)" ]; then \
		echo "$(RED)Error: PROD_HOST not set$(NC)"; \
		exit 1; \
	fi
	@echo "Deploying to $(PROD_HOST)..."
	./scripts/deploy_production.sh $(PROD_HOST)
	@echo "$(GREEN)Deployment complete$(NC)"

##@ Database

db-create: ## Create database
	@echo "$(BLUE)Creating database...$(NC)"
	$(MIX) ecto.create
	@echo "$(GREEN)Database created$(NC)"

db-migrate: ## Run database migrations
	@echo "$(BLUE)Running migrations...$(NC)"
	$(MIX) ecto.migrate
	@echo "$(GREEN)Migrations complete$(NC)"

db-rollback: ## Rollback last migration
	@echo "$(YELLOW)Rolling back migration...$(NC)"
	$(MIX) ecto.rollback
	@echo "$(GREEN)Rollback complete$(NC)"

db-reset: ## Reset database (drop, create, migrate, seed)
	@echo "$(YELLOW)Resetting database...$(NC)"
	$(MIX) ecto.reset
	@echo "$(GREEN)Database reset complete$(NC)"

db-seed: ## Seed database
	@echo "$(BLUE)Seeding database...$(NC)"
	$(MIX) run priv/repo/seeds.exs
	@echo "$(GREEN)Seeding complete$(NC)"

##@ Health & Monitoring

health: ## Check health of all services
	@echo "$(BLUE)Checking service health...$(NC)"
	@echo ""
	@echo "Gateway API:"
	@curl -s http://localhost:4000/api/health | jq '.' || echo "$(RED)Gateway not responding$(NC)"
	@echo ""
	@echo "Julia Backend:"
	@curl -s http://localhost:6000/health || echo "$(RED)Julia backend not responding$(NC)"
	@echo ""
	@echo "Python Backend:"
	@curl -s http://localhost:7000/health || echo "$(RED)Python backend not responding$(NC)"
	@echo ""
	@echo "Redis:"
	@$(DOCKER_COMPOSE) exec -T redis redis-cli ping || echo "$(RED)Redis not responding$(NC)"
	@echo ""

metrics: ## View metrics
	@echo "$(BLUE)Current metrics:$(NC)"
	@curl -s http://localhost:4000/api/metrics | jq '.'

status: ## Show status of all services
	$(DOCKER_COMPOSE) ps

##@ Utilities

clean: ## Clean build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf _build
	rm -rf deps
	rm -rf .elixir_ls
	rm -rf priv/static
	@echo "$(GREEN)Cleanup complete$(NC)"

secret: ## Generate secret key base
	@echo "$(BLUE)Generated SECRET_KEY_BASE:$(NC)"
	@$(MIX) phx.gen.secret

env-example: ## Show environment variables example
	@cat .env.example

init: deps compile ## Initialize project (install deps and compile)
	@echo "$(GREEN)Project initialized successfully$(NC)"

verify: format lint test ## Verify code quality (format, lint, test)
	@echo "$(GREEN)All checks passed!$(NC)"

##@ Benchmarks

benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running benchmarks...$(NC)"
	$(MIX) run benchmarks/prediction_benchmark.exs
	@echo "$(GREEN)Benchmarks complete$(NC)"

load-test: ## Run load tests
	@echo "$(BLUE)Running load tests...$(NC)"
	./scripts/load_test.sh
	@echo "$(GREEN)Load tests complete$(NC)"
