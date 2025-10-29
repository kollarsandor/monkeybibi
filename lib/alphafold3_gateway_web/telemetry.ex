defmodule AlphaFold3GatewayWeb.Telemetry do
  use Supervisor
  import Telemetry.Metrics
  require Logger

  def start_link(opts) do
    Supervisor.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @impl true
  def init(_opts) do
    children = [
      {:telemetry_poller, measurements: periodic_measurements(), period: 10_000}
    ]

    Supervisor.init(children, strategy: :one_for_one)
  end

  def metrics do
    [
      # Phoenix Metrics
      summary("phoenix.endpoint.start.system_time",
        unit: {:native, :millisecond}
      ),
      summary("phoenix.endpoint.stop.duration",
        unit: {:native, :millisecond}
      ),
      summary("phoenix.router_dispatch.start.system_time",
        unit: {:native, :millisecond}
      ),
      summary("phoenix.router_dispatch.exception.duration",
        unit: {:native, :millisecond}
      ),
      summary("phoenix.router_dispatch.stop.duration",
        unit: {:native, :millisecond},
        tags: [:route]
      ),
      summary("phoenix.socket_connected.duration",
        unit: {:native, :millisecond}
      ),
      summary("phoenix.channel_joined.duration",
        unit: {:native, :millisecond}
      ),
      summary("phoenix.channel_handled_in.duration",
        unit: {:native, :millisecond},
        tags: [:event]
      ),

      # VM Metrics
      summary("vm.memory.total", unit: {:byte, :kilobyte}),
      summary("vm.total_run_queue_lengths.total"),
      summary("vm.total_run_queue_lengths.cpu"),
      summary("vm.total_run_queue_lengths.io"),

      # Database Metrics (if using Ecto)
      # summary("alphafold3_gateway.repo.query.total_time", unit: {:native, :millisecond}),
      # summary("alphafold3_gateway.repo.query.decode_time", unit: {:native, :millisecond}),
      # summary("alphafold3_gateway.repo.query.query_time", unit: {:native, :millisecond}),
      # summary("alphafold3_gateway.repo.query.queue_time", unit: {:native, :millisecond}),
      # summary("alphafold3_gateway.repo.query.idle_time", unit: {:native, :millisecond}),

      # Custom Gateway Metrics
      counter("alphafold3_gateway.prediction.submitted.count"),
      counter("alphafold3_gateway.prediction.completed.count"),
      counter("alphafold3_gateway.prediction.failed.count"),
      summary("alphafold3_gateway.prediction.duration",
        unit: {:native, :millisecond}
      ),
      
      counter("alphafold3_gateway.backend.request.count",
        tags: [:backend, :status]
      ),
      summary("alphafold3_gateway.backend.request.duration",
        unit: {:native, :millisecond},
        tags: [:backend]
      ),

      counter("alphafold3_gateway.upload.count",
        tags: [:status]
      ),
      summary("alphafold3_gateway.upload.size",
        unit: {:byte, :kilobyte}
      ),

      counter("alphafold3_gateway.sequence.validation.count",
        tags: [:status]
      ),
      counter("alphafold3_gateway.sequence.analysis.count",
        tags: [:status]
      )
    ]
  end

  defp periodic_measurements do
    [
      # A module, function and arguments to be invoked periodically.
      {__MODULE__, :dispatch_vm_stats, []}
    ]
  end

  def dispatch_vm_stats do
    :telemetry.execute(
      [:vm, :memory],
      Map.new(:erlang.memory()),
      %{}
    )
  end
end
