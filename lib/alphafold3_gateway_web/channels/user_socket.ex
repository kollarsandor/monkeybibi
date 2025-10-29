defmodule AlphaFold3GatewayWeb.UserSocket do
  use Phoenix.Socket
  require Logger

  channel "job:*", AlphaFold3GatewayWeb.JobChannel
  channel "system:*", AlphaFold3GatewayWeb.SystemChannel

  @impl true
  def connect(params, socket, _connect_info) do
    Logger.info("WebSocket connection established with params: #{inspect(params)}")
    {:ok, assign(socket, :user_id, params["user_id"])}
  end

  @impl true
  def id(socket), do: "user_socket:#{socket.assigns.user_id}"
end
