import Config

config :alphafold3_gateway, AlphaFold3GatewayWeb.Endpoint,
  http: [ip: {127, 0, 0, 1}, port: 4002],
  secret_key_base: "test_secret_key_base_for_testing_purposes_only",
  server: false

config :logger, level: :warning

config :phoenix, :plug_init_mode, :runtime
