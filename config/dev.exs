import Config

config :alphafold3_gateway, AlphaFold3GatewayWeb.Endpoint,
  http: [ip: {0, 0, 0, 0}, port: 3002],
  debug_errors: true,
  code_reloader: true,
  check_origin: false,
  watchers: []

config :logger, :console, format: "[$level] $message\n"

config :phoenix, :stacktrace_depth, 20

config :phoenix, :plug_init_mode, :runtime
