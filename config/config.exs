import Config

config :alphafold3_gateway, AlphaFold3GatewayWeb.Endpoint,
  url: [host: "localhost"],
  adapter: Phoenix.Endpoint.Cowboy2Adapter,
  render_errors: [
    formats: [html: AlphaFold3GatewayWeb.ErrorHTML, json: AlphaFold3GatewayWeb.ErrorJSON],
    layout: false
  ],
  pubsub_server: AlphaFold3Gateway.PubSub,
  live_view: [signing_salt: "alphafold3_secret_salt"],
  secret_key_base: "your_64_byte_secret_key_base_here_replace_in_production",
  http: [port: 4000],
  check_origin: false,
  websocket: [
    connect_info: [:peer_data, :x_headers],
    timeout: :infinity
  ]

config :alphafold3_gateway, :backends,
  julia_backend: [
    host: System.get_env("JULIA_BACKEND_HOST") || "localhost",
    port: String.to_integer(System.get_env("JULIA_BACKEND_PORT") || "6000"),
    protocol: System.get_env("JULIA_BACKEND_PROTOCOL") || "http",
    timeout: 300_000,
    pool_size: 10
  ],
  python_backend: [
    host: System.get_env("PYTHON_BACKEND_HOST") || "localhost",
    port: String.to_integer(System.get_env("PYTHON_BACKEND_PORT") || "8000"),
    protocol: System.get_env("PYTHON_BACKEND_PROTOCOL") || "http",
    timeout: 300_000,
    pool_size: 10
  ]

config :alphafold3_gateway, :cache,
  adapter: Cachex,
  ttl: :timer.hours(1),
  limit: 1000

config :alphafold3_gateway, :rate_limiting,
  enabled: true,
  requests_per_minute: 60,
  burst_size: 10

config :logger, :console,
  format: "$time $metadata[$level] $message\n",
  metadata: [:request_id, :user_id]

config :phoenix, :json_library, Jason

import_config "#{config_env()}.exs"
