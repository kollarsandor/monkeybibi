import Config

# Production configuration
# Note: Most runtime configuration is in config/runtime.exs
# This file contains compile-time production settings

config :alphafold3_gateway, AlphaFold3GatewayWeb.Endpoint,
  cache_static_manifest: "priv/static/cache_manifest.json",
  server: true

# Production logger configuration
config :logger,
  level: :info,
  backends: [:console],
  compile_time_purge_matching: [
    [level_lower_than: :info]
  ],
  console: [
    format: "$time $metadata[$level] $message\n",
    metadata: [:request_id, :job_id, :backend, :remote_ip]
  ]

# SSL/TLS force for production
config :alphafold3_gateway, AlphaFold3GatewayWeb.Endpoint,
  force_ssl: [rewrite_on: [:x_forwarded_proto], hsts: true],
  check_origin: true

# Production Phoenix configuration
config :phoenix, :serve_endpoints, true
config :phoenix, :json_library, Jason

# Production-specific optimizations
config :alphafold3_gateway,
  environment: :prod,
  compile_time_purge_level: :info
