defmodule AlphaFold3GatewayWeb.SequenceController do
  use AlphaFold3GatewayWeb, :controller
  require Logger

  @valid_amino_acids ~w(A C D E F G H I K L M N P Q R S T V W Y)
  @valid_nucleotides ~w(A T G C U)

  @doc """
  Validates protein or nucleotide sequences
  POST /api/sequences/validate
  Body: %{"sequence" => "ACDEFGH...", "type" => "protein" | "nucleotide"}
  """
  def validate(conn, params) do
    :telemetry.execute(
      [:alphafold3_gateway, :sequence, :validation],
      %{count: 1},
      %{status: :started}
    )

    case validate_sequence(params) do
      {:ok, validation_result} ->
        :telemetry.execute(
          [:alphafold3_gateway, :sequence, :validation],
          %{count: 1},
          %{status: :success}
        )

        conn
        |> put_status(:ok)
        |> json(%{
          valid: validation_result.valid,
          sequence_type: validation_result.type,
          length: validation_result.length,
          errors: validation_result.errors,
          warnings: validation_result.warnings
        })

      {:error, reason} ->
        :telemetry.execute(
          [:alphafold3_gateway, :sequence, :validation],
          %{count: 1},
          %{status: :error}
        )

        Logger.warning("Sequence validation failed: #{inspect(reason)}")

        conn
        |> put_status(:bad_request)
        |> json(%{error: reason})
    end
  end

  @doc """
  Analyzes protein or nucleotide sequences for various properties
  POST /api/sequences/analyze
  Body: %{"sequence" => "ACDEFGH...", "type" => "protein" | "nucleotide"}
  """
  def analyze(conn, params) do
    :telemetry.execute(
      [:alphafold3_gateway, :sequence, :analysis],
      %{count: 1},
      %{status: :started}
    )

    case analyze_sequence(params) do
      {:ok, analysis_result} ->
        :telemetry.execute(
          [:alphafold3_gateway, :sequence, :analysis],
          %{count: 1},
          %{status: :success}
        )

        conn
        |> put_status(:ok)
        |> json(analysis_result)

      {:error, reason} ->
        :telemetry.execute(
          [:alphafold3_gateway, :sequence, :analysis],
          %{count: 1},
          %{status: :error}
        )

        Logger.warning("Sequence analysis failed: #{inspect(reason)}")

        conn
        |> put_status(:bad_request)
        |> json(%{error: reason})
    end
  end

  # Private functions

  defp validate_sequence(%{"sequence" => sequence, "type" => type}) when is_binary(sequence) do
    sequence = String.upcase(String.trim(sequence))
    
    cond do
      String.length(sequence) == 0 ->
        {:error, "Sequence cannot be empty"}

      String.length(sequence) > 100_000 ->
        {:error, "Sequence too long (maximum 100,000 residues)"}

      type not in ["protein", "nucleotide"] ->
        {:error, "Invalid sequence type. Must be 'protein' or 'nucleotide'"}

      true ->
        validate_sequence_characters(sequence, type)
    end
  end

  defp validate_sequence(_params) do
    {:error, "Missing required fields: 'sequence' and 'type'"}
  end

  defp validate_sequence_characters(sequence, type) do
    valid_chars = if type == "protein", do: @valid_amino_acids, else: @valid_nucleotides
    sequence_chars = sequence |> String.graphemes() |> Enum.uniq()
    
    invalid_chars = Enum.reject(sequence_chars, fn char -> char in valid_chars end)
    
    case invalid_chars do
      [] ->
        {:ok, %{
          valid: true,
          type: type,
          length: String.length(sequence),
          errors: [],
          warnings: check_sequence_warnings(sequence, type)
        }}

      chars ->
        {:ok, %{
          valid: false,
          type: type,
          length: String.length(sequence),
          errors: ["Invalid #{type} characters found: #{Enum.join(chars, ", ")}"],
          warnings: []
        }}
    end
  end

  defp check_sequence_warnings(sequence, _type) do
    warnings = []
    
    warnings = if String.length(sequence) < 10 do
      ["Sequence is very short (< 10 residues)" | warnings]
    else
      warnings
    end

    warnings = if String.length(sequence) > 50_000 do
      ["Very long sequence (> 50,000 residues) may take significant time to process" | warnings]
    else
      warnings
    end

    warnings
  end

  defp analyze_sequence(%{"sequence" => sequence, "type" => type}) when is_binary(sequence) do
    case validate_sequence(%{"sequence" => sequence, "type" => type}) do
      {:ok, %{valid: true}} ->
        sequence = String.upcase(String.trim(sequence))
        
        analysis = if type == "protein" do
          analyze_protein_sequence(sequence)
        else
          analyze_nucleotide_sequence(sequence)
        end

        {:ok, Map.merge(analysis, %{
          sequence_type: type,
          length: String.length(sequence),
          molecular_weight: calculate_molecular_weight(sequence, type)
        })}

      {:ok, %{valid: false, errors: errors}} ->
        {:error, "Invalid sequence: #{Enum.join(errors, ", ")}"}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp analyze_sequence(_params) do
    {:error, "Missing required fields: 'sequence' and 'type'"}
  end

  defp analyze_protein_sequence(sequence) do
    chars = String.graphemes(sequence)
    total = length(chars)
    
    composition = Enum.frequencies(chars)
    
    hydrophobic = count_residues(chars, ~w(A V I L M F W P))
    hydrophilic = count_residues(chars, ~w(S T N Q))
    charged = count_residues(chars, ~w(D E K R))
    aromatic = count_residues(chars, ~w(F W Y))

    %{
      composition: composition,
      properties: %{
        hydrophobic_percentage: Float.round(hydrophobic / total * 100, 2),
        hydrophilic_percentage: Float.round(hydrophilic / total * 100, 2),
        charged_percentage: Float.round(charged / total * 100, 2),
        aromatic_percentage: Float.round(aromatic / total * 100, 2)
      },
      amino_acid_counts: %{
        total: total,
        unique: map_size(composition)
      }
    }
  end

  defp analyze_nucleotide_sequence(sequence) do
    chars = String.graphemes(sequence)
    total = length(chars)
    
    composition = Enum.frequencies(chars)
    
    gc_count = count_residues(chars, ~w(G C))
    at_count = count_residues(chars, ~w(A T U))

    %{
      composition: composition,
      properties: %{
        gc_content: Float.round(gc_count / total * 100, 2),
        at_content: Float.round(at_count / total * 100, 2)
      },
      nucleotide_counts: %{
        total: total,
        unique: map_size(composition)
      }
    }
  end

  defp count_residues(chars, residue_list) do
    Enum.count(chars, fn char -> char in residue_list end)
  end

  defp calculate_molecular_weight(sequence, "protein") do
    # Average molecular weight per amino acid residue (approximate)
    avg_weight = 110.0
    weight = String.length(sequence) * avg_weight
    Float.round(weight, 2)
  end

  defp calculate_molecular_weight(sequence, "nucleotide") do
    # Average molecular weight per nucleotide (approximate)
    avg_weight = 330.0
    weight = String.length(sequence) * avg_weight
    Float.round(weight, 2)
  end
end
