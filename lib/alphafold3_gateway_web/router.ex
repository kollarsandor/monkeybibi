defmodule AlphaFold3GatewayWeb.Router do
  use AlphaFold3GatewayWeb, :router

  pipeline :api do
    plug :accepts, ["json"]
    plug AlphaFold3GatewayWeb.Plugs.RateLimitPlug
    plug AlphaFold3GatewayWeb.Plugs.MetricsPlug
  end

  pipeline :browser do
    plug :accepts, ["html"]
    plug :fetch_session
    plug :fetch_live_flash
    plug :put_root_layout, html: {AlphaFold3GatewayWeb.Layouts, :root}
    plug :protect_from_forgery
    plug :put_secure_browser_headers
  end

  scope "/", AlphaFold3GatewayWeb do
    pipe_through :browser
    get "/", PageController, :index
    get "/favicon.ico", PageController, :favicon
  end

  scope "/api", AlphaFold3GatewayWeb do
    pipe_through :api

    post "/predict", PredictionController, :predict
    post "/predict/batch", PredictionController, :batch_predict
    get "/predict/:job_id", PredictionController, :get_status
    post "/sequences/validate", SequenceController, :validate
    post "/sequences/analyze", SequenceController, :analyze
    post "/upload", UploadController, :upload
    get "/health", HealthController, :check
    get "/metrics", MetricsController, :index
    post "/chat", CerebrasController, :chat
    post "/chat/stream", CerebrasController, :stream_chat
  end

  scope "/proxy/julia", AlphaFold3GatewayWeb do
    pipe_through :api
    match :*, "/*path", ProxyController, :julia_backend
  end

  scope "/proxy/python", AlphaFold3GatewayWeb do
    pipe_through :api
    match :*, "/*path", ProxyController, :python_backend
  end

  if Mix.env() in [:dev, :test] do
    import Phoenix.LiveDashboard.Router

    scope "/" do
      pipe_through :browser
      live_dashboard "/dashboard", metrics: AlphaFold3GatewayWeb.Telemetry
    end
  end
end
