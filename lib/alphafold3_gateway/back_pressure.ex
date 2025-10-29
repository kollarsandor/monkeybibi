defmodule Alphafold3Gateway.BackPressure do
  use GenServer
  require Logger

  @default_max_queue_size 10_000
  @default_high_watermark 0.8
  @default_low_watermark 0.3

  defstruct [
    :max_queue_size,
    :high_watermark,
    :low_watermark,
    :current_queue_size,
    :drop_count,
    :total_requests,
    :strategy,
    :subscribers
  ]

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def init(opts) do
    state = %__MODULE__{
      max_queue_size: Keyword.get(opts, :max_queue_size, @default_max_queue_size),
      high_watermark: Keyword.get(opts, :high_watermark, @default_high_watermark),
      low_watermark: Keyword.get(opts, :low_watermark, @default_low_watermark),
      current_queue_size: 0,
      drop_count: 0,
      total_requests: 0,
      strategy: Keyword.get(opts, :strategy, :drop_oldest),
      subscribers: []
    }

    Logger.info("Back pressure controller started with max queue size: #{state.max_queue_size}")
    {:ok, state}
  end

  def enqueue(request) do
    GenServer.call(__MODULE__, {:enqueue, request})
  end

  def dequeue do
    GenServer.call(__MODULE__, :dequeue)
  end

  def get_status do
    GenServer.call(__MODULE__, :get_status)
  end

  def subscribe(pid) do
    GenServer.cast(__MODULE__, {:subscribe, pid})
  end

  def handle_call({:enqueue, _request}, _from, state) do
    new_total = state.total_requests + 1
    utilization = state.current_queue_size / state.max_queue_size

    cond do
      state.current_queue_size >= state.max_queue_size ->
        new_drop_count = state.drop_count + 1
        new_state = %{state | drop_count: new_drop_count, total_requests: new_total}

        notify_subscribers(new_state, :queue_full)

        {:reply, {:error, :queue_full}, new_state}

      utilization >= state.high_watermark ->
        new_state = %{
          state
          | current_queue_size: state.current_queue_size + 1,
            total_requests: new_total
        }

        notify_subscribers(new_state, :high_watermark_reached)

        {:reply, {:ok, :accepted_with_warning}, new_state}

      true ->
        new_state = %{
          state
          | current_queue_size: state.current_queue_size + 1,
            total_requests: new_total
        }

        {:reply, {:ok, :accepted}, new_state}
    end
  end

  def handle_call(:dequeue, _from, state) do
    if state.current_queue_size > 0 do
      new_size = state.current_queue_size - 1
      new_state = %{state | current_queue_size: new_size}

      utilization = new_size / state.max_queue_size

      if utilization <= state.low_watermark and utilization > 0 do
        notify_subscribers(new_state, :low_watermark_reached)
      end

      {:reply, {:ok, :dequeued}, new_state}
    else
      {:reply, {:error, :queue_empty}, state}
    end
  end

  def handle_call(:get_status, _from, state) do
    utilization = state.current_queue_size / state.max_queue_size
    drop_rate = if state.total_requests > 0, do: state.drop_count / state.total_requests, else: 0.0

    status = %{
      current_queue_size: state.current_queue_size,
      max_queue_size: state.max_queue_size,
      utilization: utilization,
      total_requests: state.total_requests,
      drop_count: state.drop_count,
      drop_rate: drop_rate,
      strategy: state.strategy,
      pressure_level: get_pressure_level(utilization, state)
    }

    {:reply, status, state}
  end

  def handle_cast({:subscribe, pid}, state) do
    new_subscribers = [pid | state.subscribers]
    {:noreply, %{state | subscribers: new_subscribers}}
  end

  defp get_pressure_level(utilization, state) do
    cond do
      utilization >= state.high_watermark -> :high
      utilization <= state.low_watermark -> :low
      true -> :normal
    end
  end

  defp notify_subscribers(state, event) do
    Enum.each(state.subscribers, fn pid ->
      send(pid, {:back_pressure_event, event, get_status_map(state)})
    end)
  end

  defp get_status_map(state) do
    utilization = state.current_queue_size / state.max_queue_size

    %{
      current_queue_size: state.current_queue_size,
      utilization: utilization,
      pressure_level: get_pressure_level(utilization, state)
    }
  end
end
