defmodule AlphaFold3GatewayWeb.JobChannel do
  use Phoenix.Channel
  require Logger

  @impl true
  def join("job:" <> job_id, _payload, socket) do
    Logger.info("Client joined job channel: #{job_id}")

    case Phoenix.PubSub.subscribe(AlphaFold3Gateway.PubSub, "job:#{job_id}") do
      :ok ->
        case Cachex.get(:gateway_cache, "job:#{job_id}") do
          {:ok, nil} ->
            {:ok, %{status: "not_found"}, Phoenix.Socket.assign(socket, :job_id, job_id)}

          {:ok, job_data} ->
            {:ok, job_data, Phoenix.Socket.assign(socket, :job_id, job_id)}

          {:error, _reason} ->
            {:ok, %{status: "cache_error"}, Phoenix.Socket.assign(socket, :job_id, job_id)}
        end

      {:error, _reason} ->
        {:error, %{reason: "pubsub_subscription_failed"}}
    end
  end

  @impl true
  def handle_info({:job_completed, job_id, result}, socket) do
    push(socket, "job_completed", %{job_id: job_id, result: result})
    {:noreply, socket}
  end

  @impl true
  def handle_info({:job_failed, job_id, error}, socket) do
    push(socket, "job_failed", %{job_id: job_id, error: error})
    {:noreply, socket}
  end

  @impl true
  def handle_info({:job_progress, job_id, progress}, socket) do
    push(socket, "job_progress", %{job_id: job_id, progress: progress})
    {:noreply, socket}
  end
end
