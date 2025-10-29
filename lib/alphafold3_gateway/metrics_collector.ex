defmodule AlphaFold3Gateway.MetricsCollector do
  use GenServer
  require Logger

  def start_link(_opts) do
    GenServer.start_link(__MODULE__, %{}, name: __MODULE__)
  end

  @impl true
  def init(_) do
    state = %{
      requests: 0,
      successful_requests: 0,
      failed_requests: 0,
      total_latency: 0,
      backend_latencies: %{}
    }
    {:ok, state}
  end

  def record_request(backend, latency, success) do
    GenServer.cast(__MODULE__, {:record, backend, latency, success})
  end

  def get_metrics do
    GenServer.call(__MODULE__, :get_metrics)
  end

  @impl true
  def handle_cast({:record, backend, latency, success}, state) do
    new_state = state
    |> Map.update!(:requests, &(&1 + 1))
    |> Map.update!(if(success, do: :successful_requests, else: :failed_requests), &(&1 + 1))
    |> Map.update!(:total_latency, &(&1 + latency))
    |> Map.update(:backend_latencies, %{}, fn latencies ->
      Map.update(latencies, backend, [latency], &([latency | &1]))
    end)

    {:noreply, new_state}
  end

  @impl true
  def handle_call(:get_metrics, _from, state) do
    avg_latency = if state.requests > 0, do: state.total_latency / state.requests, else: 0
    
    metrics = %{
      total_requests: state.requests,
      successful_requests: state.successful_requests,
      failed_requests: state.failed_requests,
      average_latency_ms: avg_latency,
      backend_latencies: state.backend_latencies
    }

    {:reply, metrics, state}
  end
end
