defmodule AlphaFold3GatewayWeb.SystemChannel do
  use Phoenix.Channel
  require Logger

  @impl true
  def join("system:metrics", _payload, socket) do
    Logger.info("Client joined system:metrics channel")
    
    # Send initial system metrics
    send(self(), :send_metrics)
    schedule_metrics_broadcast()
    
    {:ok, socket}
  end

  @impl true
  def join("system:status", _payload, socket) do
    Logger.info("Client joined system:status channel")
    
    # Send initial system status
    send(self(), :send_status)
    schedule_status_broadcast()
    
    {:ok, socket}
  end

  @impl true
  def handle_info(:send_metrics, socket) do
    metrics = collect_system_metrics()
    push(socket, "metrics_update", metrics)
    {:noreply, socket}
  end

  @impl true
  def handle_info(:send_status, socket) do
    status = collect_system_status()
    push(socket, "status_update", status)
    {:noreply, socket}
  end

  @impl true
  def handle_in("request_metrics", _payload, socket) do
    metrics = collect_system_metrics()
    {:reply, {:ok, metrics}, socket}
  end

  @impl true
  def handle_in("request_status", _payload, socket) do
    status = collect_system_status()
    {:reply, {:ok, status}, socket}
  end

  # Private helper functions

  defp schedule_metrics_broadcast do
    Process.send_after(self(), :send_metrics, 10_000) # Every 10 seconds
  end

  defp schedule_status_broadcast do
    Process.send_after(self(), :send_status, 30_000) # Every 30 seconds
  end

  defp collect_system_metrics do
    %{
      timestamp: DateTime.utc_now(),
      system: %{
        memory: get_memory_usage(),
        cpu: get_cpu_usage(),
        processes: get_process_count(),
        uptime: get_uptime_seconds()
      },
      application: %{
        requests: AlphaFold3Gateway.MetricsCollector.get_metrics(),
        cache_size: get_cache_size(),
        active_jobs: get_active_jobs_count()
      }
    }
  end

  defp collect_system_status do
    %{
      timestamp: DateTime.utc_now(),
      status: "operational",
      backends: %{
        julia: check_backend(:julia_backend_pool),
        python: check_backend(:python_backend_pool)
      },
      services: %{
        cache: check_cache(),
        pubsub: check_pubsub(),
        rate_limiter: check_rate_limiter()
      }
    }
  end

  defp get_memory_usage do
    memory = :erlang.memory()
    %{
      total: memory[:total],
      processes: memory[:processes],
      system: memory[:system],
      atom: memory[:atom],
      binary: memory[:binary],
      ets: memory[:ets]
    }
  end

  defp get_cpu_usage do
    case :cpu_sup.util() do
      {:all, 0, 0, []} -> 0
      util when is_number(util) -> util
      _ -> 0
    end
  rescue
    _ -> 0
  end

  defp get_process_count do
    length(:erlang.processes())
  end

  defp get_uptime_seconds do
    {uptime_ms, _} = :erlang.statistics(:wall_clock)
    div(uptime_ms, 1000)
  end

  defp get_cache_size do
    case Cachex.size(:gateway_cache) do
      {:ok, size} -> size
      _ -> 0
    end
  end

  defp get_active_jobs_count do
    case Cachex.keys(:gateway_cache) do
      {:ok, keys} ->
        keys
        |> Enum.filter(&String.starts_with?(&1, "job:"))
        |> length()
      _ -> 0
    end
  end

  defp check_backend(pool_name) do
    try do
      status = :poolboy.status(pool_name)
      %{
        status: "healthy",
        workers: status[:workers],
        overflow: status[:overflow],
        monitors: status[:monitors]
      }
    rescue
      _ -> %{status: "unhealthy"}
    end
  end

  defp check_cache do
    case Cachex.get(:gateway_cache, "__health_check__") do
      {:ok, _} -> "healthy"
      _ -> "unhealthy"
    end
  rescue
    _ -> "unhealthy"
  end

  defp check_pubsub do
    try do
      Process.whereis(AlphaFold3Gateway.PubSub)
      |> case do
        nil -> "unhealthy"
        _pid -> "healthy"
      end
    rescue
      _ -> "unhealthy"
    end
  end

  defp check_rate_limiter do
    try do
      Process.whereis(AlphaFold3Gateway.RateLimiter)
      |> case do
        nil -> "unhealthy"
        _pid -> "healthy"
      end
    rescue
      _ -> "unhealthy"
    end
  end
end
