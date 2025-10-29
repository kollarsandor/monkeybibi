defmodule AlphaFold3GatewayWeb.Layouts do
  use Phoenix.Component

  def root(assigns) do
    ~H"""
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="csrf-token" content={get_csrf_token()} />
        <title>AlphaFold3 Gateway</title>
      </head>
      <body>
        <%= @inner_content %>
      </body>
    </html>
    """
  end

  defp get_csrf_token do
    Plug.CSRFProtection.get_csrf_token()
  end
end
