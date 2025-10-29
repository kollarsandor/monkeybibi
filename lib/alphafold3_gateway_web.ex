defmodule AlphaFold3GatewayWeb do
  def static_paths, do: ~w(assets fonts images js favicon.ico robots.txt)

  def router do
    quote do
      use Phoenix.Router, helpers: false
      import Plug.Conn
      import Phoenix.Controller
      import Phoenix.LiveView.Router
    end
  end

  def channel do
    quote do
      use Phoenix.Channel
    end
  end

  def controller do
    quote do
      use Phoenix.Controller,
        formats: [:html, :json],
        layouts: [html: AlphaFold3GatewayWeb.Layouts]

      import Plug.Conn
      unquote(verified_routes())
    end
  end

  def verified_routes do
    quote do
      use Phoenix.VerifiedRoutes,
        endpoint: AlphaFold3GatewayWeb.Endpoint,
        router: AlphaFold3GatewayWeb.Router,
        statics: AlphaFold3GatewayWeb.static_paths()
    end
  end

  defmacro __using__(which) when is_atom(which) do
    apply(__MODULE__, which, [])
  end
end
