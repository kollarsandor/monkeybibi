defmodule AlphaFold3Gateway.BackendPool do
  use Supervisor
  require Logger

  def start_link(opts) do
    backend = Keyword.fetch!(opts, :backend)
    Supervisor.start_link(__MODULE__, opts, name: :"#{backend}_supervisor")
  end

  @impl true
  def init(opts) do
    backend = Keyword.fetch!(opts, :backend)
    config = Application.get_env(:alphafold3_gateway, :backends)[backend]
    
    pool_config = [
      name: {:local, :"#{backend}_pool"},
      worker_module: AlphaFold3Gateway.BackendWorker,
      size: config[:pool_size] || 10,
      max_overflow: 5
    ]

    children = [
      :poolboy.child_spec(:"#{backend}_pool", pool_config, config)
    ]

    Logger.info("✅ Backend pool started for #{backend}")
    Supervisor.init(children, strategy: :one_for_one)
  end

  def execute(pool_name, fun) do
    :poolboy.transaction(pool_name, fun)
  end
end
