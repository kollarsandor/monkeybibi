defmodule AlphaFold3GatewayWeb.PredictionController do
  use AlphaFold3GatewayWeb, :controller
  require Logger

  def predict(conn, params) do
    start_time = System.monotonic_time(:millisecond)
    
    case validate_prediction_params(params) do
      :ok ->
        job_id = generate_job_id()
        
        Cachex.put(:gateway_cache, "job:#{job_id}", %{
          job_id: job_id,
          status: "queued",
          queued_at: DateTime.utc_now(),
          protein_sequence: params["protein_sequence"],
          dna_sequence: params["dna_sequence"],
          rna_sequence: params["rna_sequence"],
          ligand_files: params["ligand_files"] || [],
          num_recycles: params["num_recycles"] || 10,
          num_diffusion_samples: params["num_diffusion_samples"] || 5,
          quantum_backend: params["quantum_backend"] || "simulator",
          quantum_opt_level: params["quantum_opt_level"] || 3,
          quantum_shots: params["quantum_shots"] || 1024,
          conformer_samples: params["conformer_samples"] || 20
        })
        
        Task.start(fn ->
          process_prediction_full(job_id, params)
        end)

        conn
        |> put_status(:accepted)
        |> json(%{
          job_id: job_id,
          status: "queued",
          message: "Fehérjeszerkezet előrejelzési feladat sikeresen elindítva",
          estimated_time_minutes: estimate_computation_time(params)
        })

      {:error, reason} ->
        conn
        |> put_status(:bad_request)
        |> json(%{error: reason})
    end
  end

  def get_status(conn, %{"job_id" => job_id}) do
    case Cachex.get(:gateway_cache, "job:#{job_id}") do
      {:ok, nil} ->
        conn
        |> put_status(:not_found)
        |> json(%{error: "A feladat nem található"})

      {:ok, job_data} ->
        conn
        |> json(job_data)
    end
  end

  defp process_prediction_full(job_id, params) do
    Logger.info("Előrejelzési feladat indítása: #{job_id}")
    
    Cachex.put(:gateway_cache, "job:#{job_id}", %{
      job_id: job_id,
      status: "processing",
      started_at: DateTime.utc_now(),
      current_phase: "initialization"
    })

    Phoenix.PubSub.broadcast(
      AlphaFold3Gateway.PubSub,
      "job:#{job_id}",
      {:job_progress, job_id, %{
        stage: "initialization",
        progress: 5,
        message: "Feladat inicializálása...",
        overall_progress: 5,
        sequence_progress: 0,
        msa_progress: 0,
        quantum_progress: 0,
        inference_progress: 0,
        postprocess_progress: 0
      }}
    )

    Phoenix.PubSub.broadcast(
      AlphaFold3Gateway.PubSub,
      "job:#{job_id}",
      {:job_progress, job_id, %{
        stage: "sequence_processing",
        progress: 10,
        message: "Szekvencia validálása és előfeldolgozása...",
        overall_progress: 10,
        sequence_progress: 50,
        msa_progress: 0,
        quantum_progress: 0,
        inference_progress: 0,
        postprocess_progress: 0
      }}
    )

    result = :poolboy.transaction(:julia_backend_pool, fn worker ->
      request_body = %{
        job_id: job_id,
        protein_sequence: params["protein_sequence"],
        dna_sequence: params["dna_sequence"],
        rna_sequence: params["rna_sequence"],
        ligand_files: params["ligand_files"] || [],
        num_recycles: params["num_recycles"] || 10,
        num_diffusion_samples: params["num_diffusion_samples"] || 5,
        quantum_backend: params["quantum_backend"] || "simulator",
        quantum_opt_level: params["quantum_opt_level"] || 3,
        quantum_shots: params["quantum_shots"] || 1024,
        conformer_samples: params["conformer_samples"] || 20,
        use_quantum_enhancement: params["use_quantum_enhancement"] != false
      }
      
      body = Jason.encode!(request_body)
      headers = [{"content-type", "application/json"}]
      
      Logger.info("Julia backend hívása: #{job_id}")
      AlphaFold3Gateway.BackendWorker.make_request(worker, :post, "/api/predict", body, headers)
    end)

    case result do
      {:ok, %HTTPoison.Response{status_code: 200, body: response_body}} ->
        parsed_result = Jason.decode!(response_body)
        
        Logger.info("Sikeres előrejelzés: #{job_id}")
        
        Cachex.put(:gateway_cache, "job:#{job_id}", %{
          job_id: job_id,
          status: "completed",
          result: parsed_result,
          completed_at: DateTime.utc_now(),
          computation_time: parsed_result["computation_time"]
        })

        Phoenix.PubSub.broadcast(
          AlphaFold3Gateway.PubSub,
          "job:#{job_id}",
          {:job_completed, job_id, parsed_result}
        )

      {:ok, %HTTPoison.Response{status_code: status_code, body: response_body}} ->
        error_msg = "Julia backend hiba (#{status_code}): #{response_body}"
        Logger.error("Előrejelzés sikertelen: #{job_id} - #{error_msg}")
        
        Cachex.put(:gateway_cache, "job:#{job_id}", %{
          job_id: job_id,
          status: "failed",
          error: error_msg,
          failed_at: DateTime.utc_now()
        })

        Phoenix.PubSub.broadcast(
          AlphaFold3Gateway.PubSub,
          "job:#{job_id}",
          {:job_failed, job_id, error_msg}
        )

      {:error, %HTTPoison.Error{reason: :econnrefused}} ->
        error_msg = "A Julia backend nem elérhető. Kérjük, indítsa el a szervert."
        Logger.error("Backend kapcsolat sikertelen: #{job_id}")
        
        Cachex.put(:gateway_cache, "job:#{job_id}", %{
          job_id: job_id,
          status: "failed",
          error: error_msg,
          failed_at: DateTime.utc_now()
        })

        Phoenix.PubSub.broadcast(
          AlphaFold3Gateway.PubSub,
          "job:#{job_id}",
          {:job_failed, job_id, error_msg}
        )

      {:error, reason} ->
        error_msg = "Hiba történt: #{inspect(reason)}"
        Logger.error("Általános hiba: #{job_id} - #{error_msg}")
        
        Cachex.put(:gateway_cache, "job:#{job_id}", %{
          job_id: job_id,
          status: "failed",
          error: error_msg,
          failed_at: DateTime.utc_now()
        })

        Phoenix.PubSub.broadcast(
          AlphaFold3Gateway.PubSub,
          "job:#{job_id}",
          {:job_failed, job_id, error_msg}
        )
    end
  end

  defp validate_prediction_params(params) do
    cond do
      !Map.has_key?(params, "protein_sequence") or params["protein_sequence"] == "" ->
        {:error, "A fehérje szekvencia megadása kötelező"}
      
      String.length(params["protein_sequence"]) > 100000 ->
        {:error, "A szekvencia túl hosszú (maximum 100,000 karakter)"}
      
      !valid_amino_acid_sequence?(params["protein_sequence"]) ->
        {:error, "Érvénytelen aminosav szekvencia. Csak standard aminosav kódok használhatók."}
      
      true ->
        :ok
    end
  end

  defp valid_amino_acid_sequence?(sequence) do
    valid_chars = ~w(A C D E F G H I K L M N P Q R S T V W Y)
    sequence
    |> String.upcase()
    |> String.graphemes()
    |> Enum.all?(fn char -> char in valid_chars end)
  end

  defp estimate_computation_time(params) do
    seq_length = String.length(params["protein_sequence"] || "")
    num_recycles = params["num_recycles"] || 10
    num_diffusion = params["num_diffusion_samples"] || 5
    
    base_time = 2.0
    sequence_factor = seq_length / 100.0
    recycle_factor = num_recycles * 0.5
    diffusion_factor = num_diffusion * 0.3
    
    total_minutes = base_time + sequence_factor + recycle_factor + diffusion_factor
    round(total_minutes)
  end

  def batch_predict(conn, %{"predictions" => predictions}) when is_list(predictions) do
    :telemetry.execute(
      [:alphafold3_gateway, :prediction, :batch_submitted],
      %{count: length(predictions)},
      %{batch_size: length(predictions)}
    )

    job_ids = Enum.map(predictions, fn prediction_params ->
      job_id = generate_job_id()
      
      Cachex.put(:gateway_cache, "job:#{job_id}", %{
        job_id: job_id,
        status: "queued",
        queued_at: DateTime.utc_now(),
        protein_sequence: prediction_params["protein_sequence"],
        dna_sequence: prediction_params["dna_sequence"],
        rna_sequence: prediction_params["rna_sequence"],
        ligand_files: prediction_params["ligand_files"] || [],
        num_recycles: prediction_params["num_recycles"] || 10,
        num_diffusion_samples: prediction_params["num_diffusion_samples"] || 5,
        quantum_backend: prediction_params["quantum_backend"] || "simulator",
        quantum_opt_level: prediction_params["quantum_opt_level"] || 3,
        quantum_shots: prediction_params["quantum_shots"] || 1024,
        conformer_samples: prediction_params["conformer_samples"] || 20
      })
      
      Task.start(fn ->
        process_prediction_full(job_id, prediction_params)
      end)

      job_id
    end)

    conn
    |> put_status(:accepted)
    |> json(%{
      batch_job_ids: job_ids,
      total_jobs: length(job_ids),
      status: "queued",
      message: "Batch prediction jobs submitted successfully"
    })
  end

  def batch_predict(conn, _params) do
    conn
    |> put_status(:bad_request)
    |> json(%{error: "Invalid batch prediction request. 'predictions' array is required."})
  end

  defp generate_job_id do
    timestamp = System.system_time(:millisecond)
    random = :crypto.strong_rand_bytes(8) |> Base.url_encode64(padding: false)
    "job_#{timestamp}_#{random}"
  end
end
