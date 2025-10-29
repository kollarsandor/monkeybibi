
defmodule AlphaFold3GatewayWeb.CerebrasController do
  use AlphaFold3GatewayWeb, :controller
  alias AlphaFold3Gateway.CerebrasClient
  require Logger

  def chat(conn, %{"message" => message, "history" => history}) do
    messages = build_messages(message, history)
    
    case CerebrasClient.chat_completion(messages, max_tokens: 65000) do
      {:ok, response} ->
        content = get_in(response, ["choices", Access.at(0), "message", "content"])
        usage = response["usage"]
        
        json(conn, %{
          success: true,
          response: content,
          usage: usage,
          model: "gpt-oss-120b"
        })
      
      {:error, error} ->
        conn
        |> put_status(:internal_server_error)
        |> json(%{success: false, error: error})
    end
  end

  def stream_chat(conn, %{"message" => message, "history" => history}) do
    messages = build_messages(message, history)
    
    conn = 
      conn
      |> put_resp_header("content-type", "text/event-stream")
      |> put_resp_header("cache-control", "no-cache")
      |> put_resp_header("connection", "keep-alive")
      |> send_chunked(200)

    case CerebrasClient.stream_chat_completion(messages, max_tokens: 65000) do
      {:ok, %HTTPoison.AsyncResponse{id: ref}} ->
        stream_loop(conn, ref, "")
      
      {:error, error} ->
        chunk(conn, "data: #{Jason.encode!(%{error: error})}\n\n")
    end

    conn
  end

  defp build_messages(message, history) when is_list(history) do
    system_message = %{
      "role" => "system",
      "content" => "Te egy fejlett AI asszisztens vagy JADED AlphaFold3 rendszerben. Segítesz a tudományos kutatásban, protein struktúra előrejelzésben és adatelemzésben."
    }

    history_messages = Enum.map(history, fn msg ->
      %{
        "role" => msg["role"],
        "content" => msg["content"]
      }
    end)

    [system_message | history_messages] ++ [%{"role" => "user", "content" => message}]
  end

  defp build_messages(message, _history) do
    [
      %{
        "role" => "system",
        "content" => "Te egy fejlett AI asszisztens vagy JADED AlphaFold3 rendszerben. Segítesz a tudományos kutatásban, protein struktúra előrejelzésben és adatelemzésben."
      },
      %{"role" => "user", "content" => message}
    ]
  end

  defp stream_loop(conn, ref, accumulated) do
    receive do
      %HTTPoison.AsyncStatus{code: 200} ->
        HTTPoison.stream_next(%HTTPoison.AsyncResponse{id: ref})
        stream_loop(conn, ref, accumulated)
      
      %HTTPoison.AsyncHeaders{} ->
        HTTPoison.stream_next(%HTTPoison.AsyncResponse{id: ref})
        stream_loop(conn, ref, accumulated)
      
      %HTTPoison.AsyncChunk{chunk: chunk} ->
        lines = String.split(accumulated <> chunk, "\n")
        {complete_lines, rest} = Enum.split(lines, -1)
        
        Enum.each(complete_lines, fn line ->
          if String.starts_with?(line, "data: ") do
            data = String.trim_leading(line, "data: ")
            unless data == "[DONE]" do
              chunk(conn, "data: #{data}\n\n")
            end
          end
        end)
        
        HTTPoison.stream_next(%HTTPoison.AsyncResponse{id: ref})
        stream_loop(conn, ref, List.first(rest) || "")
      
      %HTTPoison.AsyncEnd{} ->
        chunk(conn, "data: [DONE]\n\n")
        conn
      
      %HTTPoison.Error{reason: reason} ->
        Logger.error("Stream error: #{inspect(reason)}")
        chunk(conn, "data: #{Jason.encode!(%{error: reason})}\n\n")
        conn
    after
      120_000 ->
        Logger.error("Stream timeout")
        conn
    end
  end
end
