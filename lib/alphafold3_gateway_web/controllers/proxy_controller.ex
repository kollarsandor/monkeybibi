defmodule AlphaFold3GatewayWeb.ProxyController do
  use AlphaFold3GatewayWeb, :controller
  require Logger

  def julia_backend(conn, %{"path" => path}) do
    proxy_request(conn, :julia_backend, path)
  end

  def python_backend(conn, %{"path" => path}) do
    proxy_request(conn, :python_backend, path)
  end

  defp proxy_request(conn, backend, path) do
    start_time = System.monotonic_time(:millisecond)
    
    method = String.downcase(conn.method) |> String.to_atom()
    full_path = "/" <> Enum.join(path, "/")
    full_path = if conn.query_string != "", do: "#{full_path}?#{conn.query_string}", else: full_path
    
    {:ok, body, conn} = Plug.Conn.read_body(conn)
    
    headers = conn.req_headers
    |> Enum.filter(fn {k, _v} -> k not in ["host", "connection"] end)
    |> Enum.into([])

    pool_name = :"#{backend}_pool"
    
    result = :poolboy.transaction(pool_name, fn worker ->
      AlphaFold3Gateway.BackendWorker.make_request(worker, method, full_path, body, headers)
    end)

    latency = System.monotonic_time(:millisecond) - start_time

    case result do
      {:ok, %HTTPoison.Response{status_code: status, headers: resp_headers, body: resp_body}} ->
        AlphaFold3Gateway.MetricsCollector.record_request(backend, latency, true)
        Logger.info("✅ #{backend} request successful: #{method} #{full_path} (#{latency}ms)")
        
        conn
        |> put_resp_headers(resp_headers)
        |> send_resp(status, resp_body)

      {:error, %HTTPoison.Error{reason: reason}} ->
        AlphaFold3Gateway.MetricsCollector.record_request(backend, latency, false)
        Logger.error("❌ #{backend} request failed: #{inspect(reason)}")
        
        conn
        |> put_status(:bad_gateway)
        |> json(%{error: "Backend service unavailable", reason: inspect(reason)})
    end
  end

  defp put_resp_headers(conn, headers) do
    Enum.reduce(headers, conn, fn {key, value}, acc ->
      if key not in ["transfer-encoding", "connection"] do
        put_resp_header(acc, key, value)
      else
        acc
      end
    end)
  end
end
