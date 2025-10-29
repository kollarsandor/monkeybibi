using HTTP
using JSON3

function setup_modal_credentials()
    modal_token_id = get(ENV, "MODAL_TOKEN_ID", "")
    modal_token_secret = get(ENV, "MODAL_TOKEN_SECRET", "")
    
    if isempty(modal_token_id) || isempty(modal_token_secret)
        error("❌ MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables must be set!")
    end
    
    println("✅ Modal credentials found")
    
    modal_dir = joinpath(homedir(), ".modal")
    mkpath(modal_dir)
    
    credentials = Dict(
        "token_id" => modal_token_id,
        "token_secret" => modal_token_secret
    )
    
    credentials_file = joinpath(modal_dir, "credentials.json")
    open(credentials_file, "w") do f
        JSON3.write(f, credentials)
    end
    
    println("✅ Modal credentials saved to $credentials_file")
    
    return true
end

function install_modal()
    println("=" * 80)
    println("📦 Installing Modal Python package...")
    println("=" * 80)
    
    run(`python3 -m pip install --upgrade pip`)
    run(`python3 -m pip install modal`)
    
    println("✅ Modal installed successfully")
    
    return true
end

function authenticate_modal()
    println("=" * 80)
    println("🔐 Authenticating with Modal.com...")
    println("=" * 80)
    
    modal_token_id = ENV["MODAL_TOKEN_ID"]
    modal_token_secret = ENV["MODAL_TOKEN_SECRET"]
    
    run(`python3 -c "import modal; modal.config.Config.set_credentials('$modal_token_id', '$modal_token_secret')"`)
    
    println("✅ Modal authentication complete")
    
    return true
end

function create_modal_secrets()
    println("=" * 80)
    println("🔐 Creating Modal secrets...")
    println("=" * 80)
    
    modal_token_id = ENV["MODAL_TOKEN_ID"]
    modal_token_secret = ENV["MODAL_TOKEN_SECRET"]
    
    secret_creation_script = """
import modal
from modal import Secret

app = modal.App("alphafold3-secrets")

try:
    secret = Secret.from_dict({
        "MODAL_TOKEN_ID": "$modal_token_id",
        "MODAL_TOKEN_SECRET": "$modal_token_secret"
    })
    secret.create("modal-credentials", overwrite=True)
    print("✅ Modal secret 'modal-credentials' created successfully")
except Exception as e:
    print(f"ℹ️  Secret creation: {str(e)}")
"""
    
    secret_script_path = joinpath(pwd(), "create_modal_secret.py")
    open(secret_script_path, "w") do f
        write(f, secret_creation_script)
    end
    
    run(`python3 $secret_script_path`)
    
    rm(secret_script_path)
    
    println("✅ Modal secrets configured")
    
    return true
end

function upload_main_jl_to_modal()
    println("=" * 80)
    println("📤 Preparing main.jl for Modal upload...")
    println("=" * 80)
    
    main_jl_path = joinpath(pwd(), "main.jl")
    
    if !isfile(main_jl_path)
        error("❌ main.jl not found in current directory!")
    end
    
    main_jl_content = read(main_jl_path, String)
    
    ENV["MAIN_JL_CONTENT"] = main_jl_content
    
    println("✅ main.jl content loaded ($(length(main_jl_content)) bytes)")
    
    return true
end

function download_datasets_on_modal(datasets::String="all")
    println("=" * 80)
    println("📥 Starting dataset download on Modal.com...")
    println("=" * 80)
    
    action = datasets == "all" ? "download_all" : "download_$(datasets)"
    
    cmd = `python3 modal_training.py $action`
    
    println("Running: $cmd")
    
    run(cmd)
    
    println("✅ Dataset download completed")
    
    return true
end

function start_training_on_modal(;
    epochs::Int=100,
    batch_size::Int=8,
    learning_rate::Float64=0.0001
)
    println("=" * 80)
    println("🚀 Starting AlphaFold3 Training on Modal.com")
    println("=" * 80)
    println("Configuration:")
    println("  Epochs: $epochs")
    println("  Batch Size: $batch_size")
    println("  Learning Rate: $learning_rate")
    println("  GPUs: 8x H100")
    println("=" * 80)
    
    cmd = `python3 modal_training.py train --epochs $epochs --batch-size $batch_size --learning-rate $learning_rate`
    
    println("\nRunning: $cmd")
    
    run(cmd)
    
    println("\n" * "=" * 80)
    println("✅ TRAINING COMPLETED SUCCESSFULLY!")
    println("=" * 80)
    
    return true
end

function run_full_training_pipeline(;
    epochs::Int=100,
    batch_size::Int=8,
    learning_rate::Float64=0.0001,
    download_data::Bool=true
)
    println("=" * 80)
    println("🧬 AlphaFold3 Full Training Pipeline on Modal.com")
    println("=" * 80)
    
    try
        println("\n📋 Step 1/7: Installing Modal...")
        install_modal()
        
        println("\n📋 Step 2/7: Setting up Modal credentials...")
        setup_modal_credentials()
        
        println("\n📋 Step 3/7: Authenticating with Modal...")
        authenticate_modal()
        
        println("\n📋 Step 4/7: Creating Modal secrets...")
        create_modal_secrets()
        
        println("\n📋 Step 5/7: Uploading main.jl...")
        upload_main_jl_to_modal()
        
        if download_data
            println("\n📋 Step 6/7: Downloading datasets on Modal...")
            download_datasets_on_modal("all")
        else
            println("\n📋 Step 6/7: Skipping dataset download (already downloaded)")
        end
        
        println("\n📋 Step 7/7: Starting training on Modal...")
        start_training_on_modal(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        println("\n" * "=" * 80)
        println("✅ FULL TRAINING PIPELINE COMPLETED!")
        println("=" * 80)
        
        return true
        
    catch e
        println("\n" * "=" * 80)
        println("❌ ERROR: Training pipeline failed!")
        println("=" * 80)
        println("Error: $e")
        println(stacktrace(catch_backtrace()))
        
        return false
    end
end

println("=" * 80)
println("🧬 AlphaFold3 Modal Training Control Script Loaded")
println("=" * 80)
println()
println("📖 Available Functions:")
println()
println("  1. run_full_training_pipeline()")
println("     - Teljes training pipeline futtatása Modal.com-on")
println("     - Paraméterek:")
println("       * epochs::Int=100")
println("       * batch_size::Int=8")
println("       * learning_rate::Float64=0.0001")
println("       * download_data::Bool=true")
println()
println("  2. download_datasets_on_modal(datasets=\"all\")")
println("     - Datasetek letöltése Modal.com-ra")
println("     - Opciók: \"all\", \"pdb\", \"alphafold\", \"uniprot\", \"mgnify\", \"empiar\", \"bmrb\"")
println()
println("  3. start_training_on_modal()")
println("     - Training indítása Modal.com-on")
println("     - Paraméterek:")
println("       * epochs::Int=100")
println("       * batch_size::Int=8")
println("       * learning_rate::Float64=0.0001")
println()
println("=" * 80)
println()
println("🚀 Gyors start:")
println("   julia> run_full_training_pipeline(epochs=100, batch_size=8)")
println()
println("=" * 80)
