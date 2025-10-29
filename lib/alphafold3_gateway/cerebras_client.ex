
defmodule AlphaFold3Gateway.CerebrasClient do
  @moduledoc """
  Cerebras Inference API kliens GPT-OSS-120B modellhez
  """
  require Logger

  @base_url "https://api.cerebras.ai/v1"
  @model "gpt-oss-120b"
  @max_tokens 65000

  def chat_completion(messages, opts \\ []) do
    api_key = get_api_key()
    
    body = %{
      model: @model,
      messages: messages,
      max_tokens: Keyword.get(opts, :max_tokens, @max_tokens),
      temperature: Keyword.get(opts, :temperature, 0.7),
      stream: Keyword.get(opts, :stream, false)
    }

    headers = [
      {"Authorization", "Bearer #{api_key}"},
      {"Content-Type", "application/json"}
    ]

    case HTTPoison.post("#{@base_url}/chat/completions", Jason.encode!(body), headers, recv_timeout: 120_000) do
      {:ok, %HTTPoison.Response{status_code: 200, body: response_body}} ->
        {:ok, Jason.decode!(response_body)}
      
      {:ok, %HTTPoison.Response{status_code: status_code, body: error_body}} ->
        Logger.error("Cerebras API error: #{status_code} - #{error_body}")
        {:error, %{status: status_code, message: error_body}}
      
      {:error, %HTTPoison.Error{reason: reason}} ->
        Logger.error("Cerebras HTTP error: #{inspect(reason)}")
        {:error, %{message: "HTTP request failed: #{inspect(reason)}"}}
    end
  end

  def stream_chat_completion(messages, opts \\ []) do
    api_key = get_api_key()
    
    body = %{
      model: @model,
      messages: messages,
      max_tokens: Keyword.get(opts, :max_tokens, @max_tokens),
      temperature: Keyword.get(opts, :temperature, 0.7),
      stream: true
    }

    headers = [
      {"Authorization", "Bearer #{api_key}"},
      {"Content-Type", "application/json"}
    ]

    HTTPoison.post("#{@base_url}/chat/completions", Jason.encode!(body), headers, 
      stream_to: self(), 
      async: :once,
      recv_timeout: 120_000
    )
  end

  defp get_api_key do
    System.get_env("CEREBRAS_API_KEY") || 
      raise "CEREBRAS_API_KEY environment variable not set"
  end
end
