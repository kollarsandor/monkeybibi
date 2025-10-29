defmodule AlphaFold3GatewayWeb.Plugs.RateLimitPlug do
  import Plug.Conn
  require Logger

  def init(opts), do: opts

  def call(conn, _opts) do
    client_ip = get_client_ip(conn)

    case AlphaFold3Gateway.RateLimiter.check_rate_limit(client_ip) do
      :ok ->
        conn

      {:error, :rate_limited} ->
        Logger.warning("Rate limit exceeded for IP: #{inspect(client_ip)}")
        
        conn
        |> put_status(:too_many_requests)
        |> Phoenix.Controller.json(%{error: "Rate limit exceeded"})
        |> halt()
    end
  end

  defp get_client_ip(conn) do
    case get_req_header(conn, "x-forwarded-for") do
      [ip | _] -> ip
      [] -> to_string(:inet.ntoa(conn.remote_ip))
    end
  end
end
