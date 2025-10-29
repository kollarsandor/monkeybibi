defmodule AlphaFold3GatewayWeb.MetricsController do
  use AlphaFold3GatewayWeb, :controller

  def index(conn, _params) do
    metrics = AlphaFold3Gateway.MetricsCollector.get_metrics()
    
    conn
    |> json(metrics)
  end
end
