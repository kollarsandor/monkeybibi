import modal
import subprocess
import os

app = modal.App("alphafold-unison")

image = (
    modal.Image.debian_slim()
    .apt_install("curl", "git", "build-essential", "wget", "ca-certificates", "redis-tools")
    .run_commands(
        "wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.5-linux-x86_64.tar.gz",
        "tar -xzf julia-1.10.5-linux-x86_64.tar.gz",
        "mv julia-1.10.5 /opt/julia",
        "ln -s /opt/julia/bin/julia /usr/local/bin/julia",
        "rm julia-1.10.5-linux-x86_64.tar.gz"
    )
    .pip_install("dagshub", "boto3", "requests", "redis", "hiredis")
    .run_commands(
        "julia -e 'using Pkg; Pkg.add([\"CUDA\", \"Flux\", \"NearestNeighbors\", \"JSON3\", \"Downloads\", \"Statistics\", \"LinearAlgebra\", \"Random\", \"Printf\", \"SIMD\", \"BenchmarkTools\", \"ThreadsX\"]); Pkg.precompile()'"
    )
    .add_local_file("dragonfly_config.py", "/root/dragonfly_config.py")
)

volume = modal.Volume.from_name("alphafold-data", create_if_missing=True)

@app.function(
    gpu="B200:8",
    image=image,
    volumes={"/data": volume},
    timeout=86400,
    memory=262144,
    cpu=48,
    secrets=[
        modal.Secret.from_name("dragonfly-credentials"),
    ]
)
def train():
    import subprocess
    import os
    import json
    from datetime import datetime
    import sys
    
    sys.path.append("/root")
    
    from dragonfly_config import DragonflyClient

    print("📂 Reloading volume to ensure latest files...")
    volume.reload()
    
    print("📋 Listing files in /data...")
    for root, dirs, files in os.walk("/data"):
        for file in files:
            filepath = os.path.join(root, file)
            size = os.path.getsize(filepath)
            print(f"  ✓ {filepath} ({size:,} bytes)")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    env["JULIA_NUM_THREADS"] = "48"
    
    dragonfly_client = None
    
    try:
        print("🔌 Connecting to Dragonfly DB (Azure Australia East)...")
        dragonfly_client = DragonflyClient()
        print("✅ Dragonfly DB connected successfully!")
    except Exception as e:
        print(f"⚠️  Dragonfly connection failed: {e}")
        print("⚠️  Continuing without Dragonfly...")

    print("🚀 Starting AlphaFold training on 8x B200 GPUs...")
    
    start_time = datetime.now()
    
    training_command = ["julia", "/data/main.jl"]
    print("📝 Using Julia training script directly")
    
    result = subprocess.run(
        training_command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise Exception(f"Training failed with exit code {result.returncode}")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("✅ Training completed successfully!")
    print(f"⏱️  Training duration: {duration:.2f} seconds ({duration/3600:.2f} hours)")
    print("💾 Saving trained model...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_info = {
        "timestamp": timestamp,
        "gpu_count": 8,
        "status": "completed",
        "epochs": 100,
        "duration_seconds": duration,
        "model_path": "/data/checkpoints",
        "dragonfly_enabled": dragonfly_client is not None,
        "training_command": " ".join(training_command)
    }
    
    with open(f"/data/model_info_{timestamp}.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    if dragonfly_client:
        try:
            dragonfly_client.save_model_checkpoint(
                epoch=100,
                model_path="/data/checkpoints",
                metadata=model_info
            )
            
            dragonfly_client.save_training_metrics({
                "timestamp": timestamp,
                "duration": str(duration),
                "gpu_count": "8",
                "status": "completed",
                "cloud_provider": "Modal",
                "gpu_type": "B200"
            })
            
            print("✅ Training metadata saved to Dragonfly DB!")
        except Exception as e:
            print(f"⚠️  Failed to save to Dragonfly: {e}")
        finally:
            dragonfly_client.close()
    
    volume.commit()
    
    print(f"✅ Model saved to /data/checkpoints")
    print(f"📊 Model info saved to /data/model_info_{timestamp}.json")

    return model_info

@app.function(volumes={"/data": volume})
def download_model(local_path: str = "trained_models"):
    import os
    import shutil
    
    print(f"📥 Downloading trained model to {local_path}...")
    
    os.makedirs(local_path, exist_ok=True)
    
    checkpoints_dir = "/data/checkpoints"
    if os.path.exists(checkpoints_dir):
        for item in os.listdir(checkpoints_dir):
            src = os.path.join(checkpoints_dir, item)
            dst = os.path.join(local_path, item)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                print(f"  ✅ Downloaded: {item}")
    
    model_infos = [f for f in os.listdir("/data") if f.startswith("model_info_")]
    for info_file in model_infos:
        src = os.path.join("/data", info_file)
        dst = os.path.join(local_path, info_file)
        shutil.copy2(src, dst)
        print(f"  ✅ Downloaded: {info_file}")
    
    print(f"✅ Model downloaded to {local_path}/")
    return {"status": "downloaded", "path": local_path}

@app.local_entrypoint()
def main():
    print("=" * 80)
    print("  🧬 AlphaFold 3 Training on Modal B200 x8 GPUs")
    print("=" * 80)
    
    result = train.remote()
    print(f"\n✅ Training result: {result}")
    
    print("\n📥 Downloading trained model...")
    download_result = download_model.remote()
    print(f"✅ Download result: {download_result}")
    
    print("\n" + "=" * 80)
    print("  ✅ TRAINING COMPLETE - Model saved and downloaded!")
    print("=" * 80)
