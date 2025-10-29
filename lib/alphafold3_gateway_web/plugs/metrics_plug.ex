defmodule AlphaFold3GatewayWeb.Plugs.MetricsPlug do
  require Logger

  def init(opts), do: opts

  def call(conn, _opts) do
    start_time = System.monotonic_time(:millisecond)

    Plug.Conn.register_before_send(conn, fn conn ->
      latency = System.monotonic_time(:millisecond) - start_time
      Logger.info("Request completed in #{latency}ms: #{conn.method} #{conn.request_path}")
      conn
    end)
  end
end
