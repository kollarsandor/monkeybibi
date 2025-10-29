defmodule AlphaFold3Gateway.BackendWorker do
  use GenServer
  require Logger

  def start_link(config) do
    GenServer.start_link(__MODULE__, config)
  end

  def make_request(pid, method, path, body \\ "", headers \\ []) do
    GenServer.call(pid, {:request, method, path, body, headers}, 300_000)
  end

  @impl true
  def init(config) when is_list(config) do
    {:ok, %{
      host: Keyword.get(config, :host, "localhost"),
      port: Keyword.get(config, :port, 6000),
      protocol: Keyword.get(config, :protocol, "http"),
      timeout: Keyword.get(config, :timeout, 300_000)
    }}
  end

  @impl true
  def handle_call({:request, method, path, body, headers}, _from, config) do
    url = build_url(config, path)

    result = case method do
      :get -> HTTPoison.get(url, headers, [recv_timeout: config[:timeout]])
      :post -> HTTPoison.post(url, body, headers, [recv_timeout: config[:timeout]])
      :put -> HTTPoison.put(url, body, headers, [recv_timeout: config[:timeout]])
      :delete -> HTTPoison.delete(url, headers, [recv_timeout: config[:timeout]])
      :patch -> HTTPoison.patch(url, body, headers, [recv_timeout: config[:timeout]])
    end

    {:reply, result, config}
  end

  defp build_url(config, path) do
    protocol = config[:protocol]
    host = config[:host]
    port = config[:port]
    "#{protocol}://#{host}:#{port}#{path}"
  end
end