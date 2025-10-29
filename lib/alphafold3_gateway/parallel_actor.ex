defmodule Alphafold3Gateway.ParallelActor do
  use GenServer
  require Logger


  defstruct [
    :id,
    :state,
    :mailbox,
    :supervisor_pid,
    :restart_count,
    :max_restarts,
    :tasks,
    :metrics
  ]

  def start_link(opts) do
    id = Keyword.get(opts, :id, generate_id())
    GenServer.start_link(__MODULE__, opts, name: via_tuple(id))
  end

  def init(opts) do
    id = Keyword.get(opts, :id, generate_id())
    supervisor_pid = Keyword.get(opts, :supervisor_pid)

    state = %__MODULE__{
      id: id,
      state: :idle,
      mailbox: :queue.new(),
      supervisor_pid: supervisor_pid,
      restart_count: 0,
      max_restarts: 10,
      tasks: %{},
      metrics: %{
        messages_processed: 0,
        errors: 0,
        avg_processing_time: 0
      }
    }

    Logger.info("Actor #{id} initialized")
    {:ok, state}
  end

  def send_message(actor_id, message) do
    GenServer.cast(via_tuple(actor_id), {:message, message})
  end

  def get_state(actor_id) do
    GenServer.call(via_tuple(actor_id), :get_state)
  end

  def get_metrics(actor_id) do
    GenServer.call(via_tuple(actor_id), :get_metrics)
  end

  def handle_cast({:message, message}, state) do
    new_mailbox = :queue.in(message, state.mailbox)
    new_state = %{state | mailbox: new_mailbox}

    send(self(), :process_mailbox)

    {:noreply, new_state}
  end

  def handle_call(:get_state, _from, state) do
    {:reply, state.state, state}
  end

  def handle_call(:get_metrics, _from, state) do
    {:reply, state.metrics, state}
  end

  def handle_info(:process_mailbox, state) do
    case :queue.out(state.mailbox) do
      {{:value, message}, new_mailbox} ->
        start_time = :os.system_time(:millisecond)

        new_state =
          try do
            result = process_message(message, state)

            processing_time = :os.system_time(:millisecond) - start_time

            new_avg =
              (state.metrics.avg_processing_time * state.metrics.messages_processed + processing_time) /
                (state.metrics.messages_processed + 1)

            new_metrics = %{
              state.metrics
              | messages_processed: state.metrics.messages_processed + 1,
                avg_processing_time: new_avg
            }

            %{result | mailbox: new_mailbox, metrics: new_metrics}
          rescue
            e ->
              Logger.error("Actor #{state.id} error: #{inspect(e)}")

              new_metrics = %{
                state.metrics
                | errors: state.metrics.errors + 1
              }

              if state.supervisor_pid do
                send(state.supervisor_pid, {:actor_error, state.id, e})
              end

              %{state | mailbox: new_mailbox, metrics: new_metrics}
          end

        if :queue.is_empty(new_state.mailbox) do
          {:noreply, %{new_state | state: :idle}}
        else
          send(self(), :process_mailbox)
          {:noreply, %{new_state | state: :processing}}
        end

      {:empty, _} ->
        {:noreply, %{state | state: :idle}}
    end
  end

  defp process_message(message, state) do
    case message do
      {:compute, task_id, data} ->
        result = perform_computation(data)
        if state.supervisor_pid do
          send(state.supervisor_pid, {:task_completed, state.id, task_id, result})
        end
        state

      {:forward, target_actor, msg} ->
        send_message(target_actor, msg)
        state

      {:update_state, new_actor_state} ->
        %{state | state: new_actor_state}

      {:task_result, task_id, result} ->
        new_tasks = Map.put(state.tasks, task_id, result)
        %{state | tasks: new_tasks}

      _ ->
        Logger.warning("Unknown message type: #{inspect(message)}")
        state
    end
  end

  defp perform_computation(data) do
    case data do
      %{type: "matrix_multiply", a: a, b: b} ->
        matrix_multiply(a, b)

      %{type: "sequence_align", seq1: seq1, seq2: seq2} ->
        sequence_alignment(seq1, seq2)

      %{type: "energy_calc", coords: coords} ->
        calculate_energy(coords)

      _ ->
        {:ok, data}
    end
  end

  defp matrix_multiply(a, b) when is_list(a) and is_list(b) do
    rows_a = length(a)
    cols_b = length(hd(b))

    result =
      for i <- 0..(rows_a - 1) do
        for j <- 0..(cols_b - 1) do
          Enum.zip(Enum.at(a, i), Enum.map(b, &Enum.at(&1, j)))
          |> Enum.reduce(0, fn {x, y}, acc -> acc + x * y end)
        end
      end

    {:ok, result}
  end

  defp sequence_alignment(seq1, seq2) do
    score = Enum.zip(String.graphemes(seq1), String.graphemes(seq2))
            |> Enum.count(fn {a, b} -> a == b end)
    
    identity = score / max(String.length(seq1), String.length(seq2))
    
    {:ok, %{score: score, identity: identity}}
  end

  defp calculate_energy(coords) when is_list(coords) do
    energy =
      for i <- 0..(length(coords) - 2), j <- (i + 1)..(length(coords) - 1) do
        coord_i = Enum.at(coords, i)
        coord_j = Enum.at(coords, j)
        
        dist = :math.sqrt(
          Enum.zip(coord_i, coord_j)
          |> Enum.reduce(0, fn {xi, xj}, acc -> acc + (xi - xj) * (xi - xj) end)
        )
        
        -1.0 / (dist + 1.0)
      end
      |> Enum.sum()

    {:ok, energy}
  end

  defp via_tuple(actor_id) do
    {:via, Registry, {Alphafold3Gateway.ActorRegistry, actor_id}}
  end

  defp generate_id do
    "actor_#{:erlang.unique_integer([:positive])}"
  end
end
