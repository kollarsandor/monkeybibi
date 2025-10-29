defmodule AlphaFold3Gateway.RateLimiter do
  use GenServer
  require Logger

  def start_link(_opts) do
    GenServer.start_link(__MODULE__, %{}, name: __MODULE__)
  end

  @impl true
  def init(_) do
    config = Application.get_env(:alphafold3_gateway, :rate_limiting)
    state = %{
      enabled: config[:enabled],
      requests_per_minute: config[:requests_per_minute],
      burst_size: config[:burst_size],
      buckets: %{}
    }
    {:ok, state}
  end

  def check_rate_limit(client_ip) do
    GenServer.call(__MODULE__, {:check_rate_limit, client_ip})
  end

  @impl true
  def handle_call({:check_rate_limit, client_ip}, _from, state) do
    if state.enabled do
      now = System.system_time(:second)
      bucket = Map.get(state.buckets, client_ip, %{tokens: state.burst_size, last_update: now})
      
      time_passed = now - bucket.last_update
      tokens_to_add = time_passed * (state.requests_per_minute / 60.0)
      new_tokens = min(bucket.tokens + tokens_to_add, state.burst_size)
      
      if new_tokens >= 1.0 do
        new_bucket = %{tokens: new_tokens - 1.0, last_update: now}
        new_buckets = Map.put(state.buckets, client_ip, new_bucket)
        {:reply, :ok, %{state | buckets: new_buckets}}
      else
        {:reply, {:error, :rate_limited}, state}
      end
    else
      {:reply, :ok, state}
    end
  end
end
