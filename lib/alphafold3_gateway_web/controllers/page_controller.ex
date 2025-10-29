defmodule AlphaFold3GatewayWeb.PageController do
  use AlphaFold3GatewayWeb, :controller
  require Logger

  @doc """
  Serves the main index.html page from priv/static
  """
  def index(conn, _params) do
    index_path = Application.app_dir(:alphafold3_gateway, "priv/static/index.html")
    
    case File.read(index_path) do
      {:ok, content} ->
        conn
        |> put_resp_content_type("text/html")
        |> put_resp_header("cache-control", "no-cache, no-store, must-revalidate")
        |> put_resp_header("pragma", "no-cache")
        |> put_resp_header("expires", "0")
        |> send_resp(200, content)

      {:error, reason} ->
        Logger.error("Failed to read index.html: #{inspect(reason)}")
        
        conn
        |> put_status(:internal_server_error)
        |> json(%{
          error: "Failed to load application",
          message: "The application interface could not be loaded. Please contact support."
        })
    end
  end

  def favicon(conn, _params) do
    send_resp(conn, :no_content, "")
  end
end
