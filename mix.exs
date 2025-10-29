defmodule AlphaFold3Gateway.MixProject do
  use Mix.Project

  def project do
    [
      app: :alphafold3_gateway,
      version: "1.0.0",
      elixir: "~> 1.14",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [
      mod: {AlphaFold3Gateway.Application, []},
      extra_applications: [:logger, :runtime_tools, :os_mon]
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      {:phoenix, "~> 1.7.14"},
      {:phoenix_html, "~> 4.1"},
      {:phoenix_live_view, "~> 0.20.17"},
      {:phoenix_live_dashboard, "~> 0.8.4"},
      {:telemetry_metrics, "~> 1.0"},
      {:telemetry_poller, "~> 1.1"},
      {:jason, "~> 1.4"},
      {:plug_cowboy, "~> 2.7"},
      {:cors_plug, "~> 3.0"},
      {:httpoison, "~> 2.2"},
      {:websockex, "~> 0.4.3"},
      {:poolboy, "~> 1.5"},
      {:cachex, "~> 3.6"},
      {:ex_json_schema, "~> 0.10.2"},
      {:quantum, "~> 3.5"},
      {:timex, "~> 3.7"},
      {:dotenv, "~> 3.1"}
    ]
  end
end
