defmodule AlphaFold3GatewayWeb.Endpoint do
  use Phoenix.Endpoint, otp_app: :alphafold3_gateway

  @session_options [
    store: :cookie,
    key: "_alphafold3_gateway_key",
    signing_salt: "alphafold3_gateway",
    same_site: "Lax"
  ]

  socket "/live", Phoenix.LiveView.Socket, websocket: [connect_info: [session: @session_options]]
  socket "/socket", AlphaFold3GatewayWeb.UserSocket, websocket: true, longpoll: false

  plug Plug.Static,
    at: "/",
    from: :alphafold3_gateway,
    gzip: false,
    only: AlphaFold3GatewayWeb.static_paths()

  if code_reloading? do
    plug Phoenix.CodeReloader
  end

  plug Phoenix.LiveDashboard.RequestLogger,
    param_key: "request_logger",
    cookie_key: "request_logger"

  plug Plug.RequestId
  plug Plug.Telemetry, event_prefix: [:phoenix, :endpoint]

  plug Plug.Parsers,
    parsers: [:urlencoded, :multipart, :json],
    pass: ["*/*"],
    json_decoder: Phoenix.json_library()

  plug Plug.MethodOverride
  plug Plug.Head
  plug Plug.Session, @session_options
  plug CORSPlug, origin: ["http://localhost:3000", "http://localhost:4000"]
  plug AlphaFold3GatewayWeb.Router
end
