import Config

# Runtime configuration for all environments
config :logger,
  level: String.to_existing_atom(System.get_env("LOG_LEVEL") || "info"),
  format: "$time $metadata[$level] $message\n",
  metadata: [:request_id, :job_id, :backend]

# Backend configuration with environment variable support
config :alphafold3_gateway, :backends,
  julia_backend: [
    host: System.get_env("JULIA_BACKEND_HOST") || "localhost",
    port: String.to_integer(System.get_env("JULIA_BACKEND_PORT") || "6000"),
    protocol: System.get_env("JULIA_BACKEND_PROTOCOL") || "http",
    timeout: String.to_integer(System.get_env("JULIA_BACKEND_TIMEOUT") || "300000"),
    pool_size: String.to_integer(System.get_env("JULIA_BACKEND_POOL_SIZE") || "10")
  ],
  python_backend: [
    host: System.get_env("PYTHON_BACKEND_HOST") || "localhost",
    port: String.to_integer(System.get_env("PYTHON_BACKEND_PORT") || "7000"),
    protocol: System.get_env("PYTHON_BACKEND_PROTOCOL") || "http",
    timeout: String.to_integer(System.get_env("PYTHON_BACKEND_TIMEOUT") || "300000"),
    pool_size: String.to_integer(System.get_env("PYTHON_BACKEND_POOL_SIZE") || "10")
  ]

# Rate limiting configuration
config :alphafold3_gateway, :rate_limiting,
  enabled: System.get_env("RATE_LIMITING_ENABLED", "true") == "true",
  requests_per_minute: String.to_integer(System.get_env("RATE_LIMIT_RPM") || "100"),
  burst_size: String.to_integer(System.get_env("RATE_LIMIT_BURST") || "20")

# Cache configuration
config :alphafold3_gateway, :cache,
  default_ttl: String.to_integer(System.get_env("CACHE_DEFAULT_TTL") || "3600"),
  max_size: String.to_integer(System.get_env("CACHE_MAX_SIZE") || "10000")

# File upload configuration
config :alphafold3_gateway, :uploads,
  max_file_size: String.to_integer(System.get_env("MAX_UPLOAD_SIZE") || "104857600"),
  allowed_extensions: ~w(.pdb .cif .fasta .mmcif .sdf .mol .mol2),
  upload_directory: System.get_env("UPLOAD_DIR") || "priv/uploads"

# Cerebras API configuration
config :alphafold3_gateway, :cerebras,
  api_key: System.get_env("CEREBRAS_API_KEY"),
  enabled: System.get_env("CEREBRAS_ENABLED", "false") == "true"

# Production-specific configuration
if config_env() == :prod do
  # Validate required environment variables
  secret_key_base =
    System.get_env("SECRET_KEY_BASE") ||
      raise """
      environment variable SECRET_KEY_BASE is missing.
      You can generate one by calling: mix phx.gen.secret
      """

  database_url =
    System.get_env("DATABASE_URL") ||
      System.get_env("POSTGRES_URL")

  # SSL/TLS configuration
  ssl_enabled = System.get_env("SSL_ENABLED", "true") == "true"
  
  ssl_config = if ssl_enabled do
    [
      keyfile: System.get_env("SSL_KEY_PATH"),
      certfile: System.get_env("SSL_CERT_PATH"),
      cacertfile: System.get_env("SSL_CA_CERT_PATH")
    ]
    |> Enum.reject(fn {_k, v} -> is_nil(v) end)
  else
    []
  end

  host = System.get_env("PHX_HOST") || raise "PHX_HOST environment variable is missing"
  port = String.to_integer(System.get_env("PORT") || "4000")

  config :alphafold3_gateway, AlphaFold3GatewayWeb.Endpoint,
    url: [host: host, port: 443, scheme: "https"],
    http: [
      ip: {0, 0, 0, 0, 0, 0, 0, 0},
      port: port,
      protocol_options: [
        max_header_name_length: 64,
        max_header_value_length: 4096,
        max_request_line_length: 8192,
        max_headers: 100
      ]
    ],
    https: if(ssl_enabled, do: [port: 443, cipher_suite: :strong] ++ ssl_config, else: nil),
    secret_key_base: secret_key_base,
    server: true,
    check_origin: String.split(System.get_env("ALLOWED_ORIGINS") || "", ",")
    
  # Production logger configuration
  config :logger,
    level: :info,
    backends: [:console],
    compile_time_purge_matching: [
      [level_lower_than: :info]
    ]

  # System monitoring
  config :alphafold3_gateway, :monitoring,
    enabled: true,
    metrics_port: String.to_integer(System.get_env("METRICS_PORT") || "9090"),
    health_check_interval: String.to_integer(System.get_env("HEALTH_CHECK_INTERVAL") || "30000")
end

# Development-specific configuration
if config_env() == :dev do
  config :alphafold3_gateway, AlphaFold3GatewayWeb.Endpoint,
    debug_errors: true,
    code_reloader: true,
    check_origin: false,
    watchers: []
end

# Test-specific configuration
if config_env() == :test do
  config :logger, level: :warning

  config :alphafold3_gateway, AlphaFold3GatewayWeb.Endpoint,
    http: [ip: {127, 0, 0, 1}, port: 4002],
    secret_key_base: "test_secret_key_base_min_64_chars_long_for_testing_purposes_only",
    server: false
end
