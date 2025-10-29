defmodule AlphaFold3Gateway.Application do
  use Application
  require Logger

  @impl true
  def start(_type, _args) do
    children = [
      {Phoenix.PubSub, name: AlphaFold3Gateway.PubSub},
      {Cachex, name: :gateway_cache},
      Supervisor.child_spec(
        {AlphaFold3Gateway.BackendPool, name: :julia_backend_pool, backend: :julia_backend},
        id: :julia_backend_pool
      ),
      Supervisor.child_spec(
        {AlphaFold3Gateway.BackendPool, name: :python_backend_pool, backend: :python_backend},
        id: :python_backend_pool
      ),
      AlphaFold3Gateway.RateLimiter,
      AlphaFold3Gateway.MetricsCollector,
      AlphaFold3GatewayWeb.Endpoint
    ]

    opts = [strategy: :one_for_one, name: AlphaFold3Gateway.Supervisor]
    Logger.info("🚀 AlphaFold3 Gateway starting on port 5000")
    Supervisor.start_link(children, opts)
  end

  @impl true
  def config_change(changed, _new, removed) do
    AlphaFold3GatewayWeb.Endpoint.config_change(changed, removed)
    :ok
  end
end
