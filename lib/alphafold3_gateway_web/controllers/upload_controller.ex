defmodule AlphaFold3GatewayWeb.UploadController do
  use AlphaFold3GatewayWeb, :controller
  require Logger

  @upload_dir "priv/uploads"
  @max_file_size 50_000_000  # 50 MB
  @allowed_extensions ~w(.pdb .cif .sdf .mol .mol2 .xyz .json)

  @doc """
  Handles ligand file uploads
  POST /api/upload
  Multipart form data with file field
  """
  def upload(conn, params) do
    start_time = System.monotonic_time()

    ensure_upload_directory()

    case validate_and_save_upload(params) do
      {:ok, file_info} ->
        duration = System.monotonic_time() - start_time
        
        :telemetry.execute(
          [:alphafold3_gateway, :upload],
          %{count: 1},
          %{status: :success}
        )

        :telemetry.execute(
          [:alphafold3_gateway, :upload, :size],
          %{size: file_info.size},
          %{}
        )

        Logger.info("File uploaded successfully: #{file_info.filename} (#{format_bytes(file_info.size)})")

        conn
        |> put_status(:created)
        |> json(%{
          success: true,
          file: %{
            id: file_info.id,
            filename: file_info.filename,
            original_filename: file_info.original_filename,
            size: file_info.size,
            size_formatted: format_bytes(file_info.size),
            mime_type: file_info.mime_type,
            path: file_info.path,
            uploaded_at: file_info.uploaded_at
          },
          message: "File uploaded successfully",
          processing_time_ms: System.convert_time_unit(duration, :native, :millisecond)
        })

      {:error, reason} ->
        :telemetry.execute(
          [:alphafold3_gateway, :upload],
          %{count: 1},
          %{status: :error}
        )

        Logger.warning("File upload failed: #{reason}")

        conn
        |> put_status(:bad_request)
        |> json(%{
          success: false,
          error: reason
        })
    end
  end

  # Private functions

  defp ensure_upload_directory do
    upload_path = Application.app_dir(:alphafold3_gateway, @upload_dir)
    File.mkdir_p!(upload_path)
  end

  defp validate_and_save_upload(%{"file" => upload}) when is_map(upload) do
    with :ok <- validate_file_size(upload),
         :ok <- validate_file_extension(upload.filename),
         {:ok, saved_path, file_id} <- save_uploaded_file(upload) do
      
      file_info = %{
        id: file_id,
        filename: Path.basename(saved_path),
        original_filename: upload.filename,
        size: File.stat!(saved_path).size,
        mime_type: upload.content_type || "application/octet-stream",
        path: saved_path,
        uploaded_at: DateTime.utc_now() |> DateTime.to_iso8601()
      }

      {:ok, file_info}
    else
      {:error, reason} -> {:error, reason}
    end
  end

  defp validate_and_save_upload(_params) do
    {:error, "No file uploaded. Please provide a file in the 'file' field."}
  end

  defp validate_file_size(%{path: path}) do
    case File.stat(path) do
      {:ok, %{size: size}} when size <= @max_file_size ->
        :ok

      {:ok, %{size: size}} ->
        {:error, "File too large. Maximum size is #{format_bytes(@max_file_size)}, got #{format_bytes(size)}"}

      {:error, reason} ->
        {:error, "Failed to read file: #{inspect(reason)}"}
    end
  end

  defp validate_file_extension(filename) do
    extension = Path.extname(filename) |> String.downcase()

    if extension in @allowed_extensions do
      :ok
    else
      {:error, "Invalid file type '#{extension}'. Allowed types: #{Enum.join(@allowed_extensions, ", ")}"}
    end
  end

  defp save_uploaded_file(upload) do
    file_id = generate_file_id()
    extension = Path.extname(upload.filename)
    filename = "#{file_id}#{extension}"
    
    upload_dir = Application.app_dir(:alphafold3_gateway, @upload_dir)
    destination = Path.join(upload_dir, filename)

    case File.cp(upload.path, destination) do
      :ok ->
        {:ok, destination, file_id}

      {:error, reason} ->
        Logger.error("Failed to save uploaded file: #{inspect(reason)}")
        {:error, "Failed to save file"}
    end
  end

  defp generate_file_id do
    timestamp = System.system_time(:millisecond)
    random = :crypto.strong_rand_bytes(8) |> Base.url_encode64(padding: false)
    "#{timestamp}_#{random}"
  end

  defp format_bytes(bytes) when bytes < 1024, do: "#{bytes} B"
  defp format_bytes(bytes) when bytes < 1_048_576, do: "#{Float.round(bytes / 1024, 2)} KB"
  defp format_bytes(bytes) when bytes < 1_073_741_824, do: "#{Float.round(bytes / 1_048_576, 2)} MB"
  defp format_bytes(bytes), do: "#{Float.round(bytes / 1_073_741_824, 2)} GB"
end
