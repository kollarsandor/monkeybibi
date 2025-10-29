defmodule AlphaFold3GatewayWeb.HealthController do
  use AlphaFold3GatewayWeb, :controller
  require Logger

  def check(conn, _params) do
    backends_status = %{
      julia: check_backend_health(:julia_backend),
      python: check_backend_health(:python_backend)
    }

    all_healthy = Enum.all?(Map.values(backends_status), &(&1 == :healthy))

    status = if all_healthy, do: :ok, else: :service_unavailable

    conn
    |> put_status(status)
    |> json(%{
      status: if(all_healthy, do: "healthy", else: "unhealthy"),
      timestamp: DateTime.utc_now(),
      backends: backends_status,
      uptime: get_uptime()
    })
  end

  defp check_backend_health(backend) do
    pool_name = :"#{backend}_pool"
    
    try do
      result = :poolboy.transaction(pool_name, fn worker ->
        AlphaFold3Gateway.BackendWorker.make_request(worker, :get, "/health", "", [])
      end, 5000)

      case result do
        {:ok, %HTTPoison.Response{status_code: 200}} -> :healthy
        _ -> :unhealthy
      end
    rescue
      _ -> :unhealthy
    end
  end

  defp get_uptime do
    {uptime_ms, _} = :erlang.statistics(:wall_clock)
    uptime_ms / 1000
  end
end
