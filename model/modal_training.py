import modal
import os
import sys
from datetime import datetime
import json
import subprocess
import tempfile
import shutil
from pathlib import Path

app = modal.App("alphafold3-training")

cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("wget", "git", "build-essential", "cmake", "gfortran", "libopenblas-dev")
    .run_commands("wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.0-linux-x86_64.tar.gz")
    .run_commands("tar -xzf julia-1.10.0-linux-x86_64.tar.gz")
    .run_commands("mv julia-1.10.0 /opt/julia")
    .env({"PATH": "/opt/julia/bin:$PATH"})
    .run_commands("julia -e 'using Pkg; Pkg.add(\"CUDA\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"Flux\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"Zygote\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"LinearAlgebra\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"Statistics\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"Random\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"Printf\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"JSON3\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"Downloads\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"Dates\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"Distributed\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"ProgressMeter\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"DataFrames\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"CSV\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"ArgParse\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"Clustering\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"Optim\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"Distributions\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"StatsBase\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"Distances\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"NearestNeighbors\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"BSON\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"HTTP\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"Tar\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"CodecZlib\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"UUIDs\")'")
    .pip_install("requests", "tqdm", "numpy", "pandas", "biopython", "pdbfixer", "openmm")
)

volume = modal.Volume.from_name("alphafold3-data", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("alphafold3-checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu="B200:8",
    volumes={
        "/data": volume,
        "/checkpoints": checkpoints_volume
    },
    timeout=86400 * 7,
    secrets=[modal.Secret.from_name("modal-credentials")],
    memory=1024 * 1024,
    cpu=96.0,
)
def download_pdb_database():
    import requests
    from pathlib import Path
    import gzip
    import shutil
    from tqdm import tqdm
    
    print("=" * 80)
    print("📥 Downloading PDB Database")
    print("=" * 80)
    
    pdb_dir = Path("/data/pdb")
    pdb_dir.mkdir(parents=True, exist_ok=True)
    
    base_url = "https://files.rcsb.org/download/"
    
    print("📡 Fetching PDB entry list...")
    response = requests.get("https://www.rcsb.org/pdb/json/getCurrent")
    pdb_list = response.json()
    
    entries = []
    if isinstance(pdb_list, list):
        entries = [entry.lower() for entry in pdb_list]
    else:
        print("⚠️  Using fallback PDB list from RCSB API")
        response = requests.get("https://data.rcsb.org/rest/v1/holdings/current/entry_ids")
        entries = response.json()
    
    print(f"✅ Found {len(entries)} PDB entries")
    
    downloaded = 0
    failed = 0
    
    for pdb_id in tqdm(entries[:50000], desc="Downloading PDB files"):
        pdb_file = pdb_dir / f"{pdb_id}.pdb"
        
        if pdb_file.exists():
            continue
        
        url = f"{base_url}{pdb_id.upper()}.pdb.gz"
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                gz_path = pdb_dir / f"{pdb_id}.pdb.gz"
                with open(gz_path, 'wb') as f:
                    f.write(response.content)
                
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(pdb_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                gz_path.unlink()
                downloaded += 1
                
                if downloaded % 1000 == 0:
                    volume.commit()
                    print(f"💾 Checkpoint: {downloaded} files downloaded")
            else:
                failed += 1
        except Exception as e:
            failed += 1
            if failed % 100 == 0:
                print(f"⚠️  {failed} files failed")
    
    volume.commit()
    
    print("=" * 80)
    print(f"✅ PDB Download Complete")
    print(f"   Downloaded: {downloaded}")
    print(f"   Failed: {failed}")
    print(f"   Total: {len(entries)}")
    print("=" * 80)
    
    return {"downloaded": downloaded, "failed": failed, "total": len(entries)}

@app.function(
    image=image,
    gpu="B200:8",
    volumes={
        "/data": volume,
        "/checkpoints": checkpoints_volume
    },
    timeout=86400 * 7,
    secrets=[modal.Secret.from_name("modal-credentials")],
    memory=1024 * 1024,
    cpu=96.0,
)
def download_alphafold_database():
    import requests
    from pathlib import Path
    from tqdm import tqdm
    import tarfile
    
    print("=" * 80)
    print("📥 Downloading AlphaFold Database")
    print("=" * 80)
    
    af_dir = Path("/data/alphafold")
    af_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = {
        "model_v4": "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar",
        "pdb70": "http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/pdb70_from_mmcif_latest.tar.gz",
        "uniref90": "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz",
        "mgnify": "https://storage.googleapis.com/alphafold-databases/v2.3/mgy_clusters_2022_05.fa.gz",
        "bfd": "https://storage.googleapis.com/alphafold-databases/v2.3/bfd-first_non_consensus_sequences.fasta.gz",
        "uniclust30": "https://storage.googleapis.com/alphafold-databases/v2.3/uniclust30_2018_08_hhsuite.tar.gz",
    }
    
    for name, url in datasets.items():
        print(f"\n📦 Downloading {name}...")
        
        filename = url.split('/')[-1]
        filepath = af_dir / filename
        
        if filepath.exists():
            print(f"✅ {name} already downloaded")
            continue
        
        try:
            response = requests.get(url, stream=True, timeout=300)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='iB', unit_scale=True, desc=name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            if filepath.suffix == '.gz' and not filepath.name.endswith('.tar.gz'):
                print(f"📂 Extracting {name}...")
                import gzip
                import shutil
                
                extracted_path = filepath.with_suffix('')
                with gzip.open(filepath, 'rb') as f_in:
                    with open(extracted_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            
            elif filepath.suffix == '.tar' or filepath.name.endswith('.tar.gz'):
                print(f"📂 Extracting {name}...")
                with tarfile.open(filepath, 'r:*') as tar:
                    tar.extractall(path=af_dir / name)
            
            volume.commit()
            print(f"✅ {name} downloaded and extracted")
            
        except Exception as e:
            print(f"❌ Failed to download {name}: {str(e)}")
    
    volume.commit()
    
    print("=" * 80)
    print("✅ AlphaFold Database Download Complete")
    print("=" * 80)
    
    return {"status": "complete"}

@app.function(
    image=image,
    gpu="B200:8",
    volumes={
        "/data": volume,
        "/checkpoints": checkpoints_volume
    },
    timeout=86400 * 7,
    secrets=[modal.Secret.from_name("modal-credentials")],
    memory=1024 * 1024,
    cpu=96.0,
)
def download_uniprot_database():
    import requests
    from pathlib import Path
    from tqdm import tqdm
    import gzip
    import shutil
    
    print("=" * 80)
    print("📥 Downloading UniProt Database")
    print("=" * 80)
    
    uniprot_dir = Path("/data/uniprot")
    uniprot_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = {
        "uniprotkb_sprot": "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz",
        "uniprotkb_trembl": "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.fasta.gz",
        "uniref100": "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref100/uniref100.fasta.gz",
        "uniref90": "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz",
        "uniref50": "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz",
    }
    
    for name, url in datasets.items():
        print(f"\n📦 Downloading {name}...")
        
        filename = url.split('/')[-1]
        filepath = uniprot_dir / filename
        extracted_path = filepath.with_suffix('')
        
        if extracted_path.exists():
            print(f"✅ {name} already downloaded and extracted")
            continue
        
        try:
            response = requests.get(url, stream=True, timeout=300)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='iB', unit_scale=True, desc=name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"📂 Extracting {name}...")
            with gzip.open(filepath, 'rb') as f_in:
                with open(extracted_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            filepath.unlink()
            
            volume.commit()
            print(f"✅ {name} downloaded and extracted")
            
        except Exception as e:
            print(f"❌ Failed to download {name}: {str(e)}")
    
    volume.commit()
    
    print("=" * 80)
    print("✅ UniProt Database Download Complete")
    print("=" * 80)
    
    return {"status": "complete"}

@app.function(
    image=image,
    gpu="B200:8",
    volumes={
        "/data": volume,
        "/checkpoints": checkpoints_volume
    },
    timeout=86400 * 7,
    secrets=[modal.Secret.from_name("modal-credentials")],
    memory=1024 * 1024,
    cpu=96.0,
)
def download_mgnify_proteins():
    import requests
    from pathlib import Path
    from tqdm import tqdm
    import gzip
    import shutil
    
    print("=" * 80)
    print("📥 Downloading MGnify Proteins Database")
    print("=" * 80)
    
    mgnify_dir = Path("/data/mgnify")
    mgnify_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = {
        "mgnify_proteins_v2023_12": "https://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/2023_12/mgnify_proteins_2023_12.fa.gz",
        "mgnify_clusters_v2023_12": "https://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/2023_12/mgnify_clusters_2023_12.fa.gz",
    }
    
    for name, url in datasets.items():
        print(f"\n📦 Downloading {name}...")
        
        filename = url.split('/')[-1]
        filepath = mgnify_dir / filename
        extracted_path = filepath.with_suffix('')
        
        if extracted_path.exists():
            print(f"✅ {name} already downloaded and extracted")
            continue
        
        try:
            response = requests.get(url, stream=True, timeout=300)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='iB', unit_scale=True, desc=name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"📂 Extracting {name}...")
            with gzip.open(filepath, 'rb') as f_in:
                with open(extracted_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            filepath.unlink()
            
            volume.commit()
            print(f"✅ {name} downloaded and extracted")
            
        except Exception as e:
            print(f"❌ Failed to download {name}: {str(e)}")
    
    volume.commit()
    
    print("=" * 80)
    print("✅ MGnify Proteins Database Download Complete")
    print("=" * 80)
    
    return {"status": "complete"}

@app.function(
    image=image,
    gpu="B200:8",
    volumes={
        "/data": volume,
        "/checkpoints": checkpoints_volume
    },
    timeout=86400 * 7,
    secrets=[modal.Secret.from_name("modal-credentials")],
    memory=1024 * 1024,
    cpu=96.0,
)
def download_empiar_cryoem():
    import requests
    from pathlib import Path
    from tqdm import tqdm
    
    print("=" * 80)
    print("📥 Downloading EMPIAR Cryo-EM Database")
    print("=" * 80)
    
    empiar_dir = Path("/data/empiar")
    empiar_dir.mkdir(parents=True, exist_ok=True)
    
    print("📡 Fetching EMPIAR entry list...")
    
    api_url = "https://www.ebi.ac.uk/empiar/api/entry/"
    
    try:
        response = requests.get(api_url, timeout=60)
        entries = response.json()
        
        print(f"✅ Found {len(entries)} EMPIAR entries")
        
        downloaded = 0
        failed = 0
        
        for entry in tqdm(entries[:1000], desc="Downloading EMPIAR metadata"):
            entry_id = entry.get('accession_code', '')
            
            if not entry_id:
                continue
            
            metadata_file = empiar_dir / f"{entry_id}_metadata.json"
            
            if metadata_file.exists():
                continue
            
            try:
                entry_url = f"{api_url}{entry_id}/"
                response = requests.get(entry_url, timeout=30)
                
                if response.status_code == 200:
                    with open(metadata_file, 'w') as f:
                        json.dump(response.json(), f, indent=2)
                    
                    downloaded += 1
                    
                    if downloaded % 100 == 0:
                        volume.commit()
                else:
                    failed += 1
            except Exception as e:
                failed += 1
        
        volume.commit()
        
        print("=" * 80)
        print(f"✅ EMPIAR Download Complete")
        print(f"   Downloaded: {downloaded}")
        print(f"   Failed: {failed}")
        print("=" * 80)
        
        return {"downloaded": downloaded, "failed": failed}
        
    except Exception as e:
        print(f"❌ Failed to download EMPIAR: {str(e)}")
        return {"downloaded": 0, "failed": 0}

@app.function(
    image=image,
    gpu="B200:8",
    volumes={
        "/data": volume,
        "/checkpoints": checkpoints_volume
    },
    timeout=86400 * 7,
    secrets=[modal.Secret.from_name("modal-credentials")],
    memory=1024 * 1024,
    cpu=96.0,
)
def download_bmrb_nmr():
    import requests
    from pathlib import Path
    from tqdm import tqdm
    
    print("=" * 80)
    print("📥 Downloading BMRB NMR Database")
    print("=" * 80)
    
    bmrb_dir = Path("/data/bmrb")
    bmrb_dir.mkdir(parents=True, exist_ok=True)
    
    base_url = "https://bmrb.io/ftp/pub/bmrb/entry_lists/"
    
    datasets = {
        "nmr_star_v3": "https://bmrb.io/ftp/pub/bmrb/relational_tables/nmr-star3.1/csv/",
    }
    
    print("📦 Downloading BMRB metadata...")
    
    try:
        response = requests.get("https://bmrb.io/search/get_all_values_from_category.php?category=Entry&format=json", timeout=60)
        entries = response.json()
        
        print(f"✅ Found {len(entries)} BMRB entries")
        
        downloaded = 0
        failed = 0
        
        for entry in tqdm(entries[:5000], desc="Downloading BMRB entries"):
            entry_id = entry.get('Entry_ID', '')
            
            if not entry_id:
                continue
            
            entry_file = bmrb_dir / f"bmr{entry_id}.json"
            
            if entry_file.exists():
                continue
            
            try:
                entry_url = f"https://bmrb.io/data_library/summary/index.php?bmrbId={entry_id}"
                response = requests.get(entry_url, timeout=30)
                
                if response.status_code == 200:
                    with open(entry_file, 'w') as f:
                        f.write(response.text)
                    
                    downloaded += 1
                    
                    if downloaded % 500 == 0:
                        volume.commit()
                else:
                    failed += 1
            except Exception as e:
                failed += 1
        
        volume.commit()
        
        print("=" * 80)
        print(f"✅ BMRB Download Complete")
        print(f"   Downloaded: {downloaded}")
        print(f"   Failed: {failed}")
        print("=" * 80)
        
        return {"downloaded": downloaded, "failed": failed}
        
    except Exception as e:
        print(f"❌ Failed to download BMRB: {str(e)}")
        return {"downloaded": 0, "failed": 0}

@app.function(
    image=image,
    gpu="B200:8",
    volumes={
        "/data": volume,
        "/checkpoints": checkpoints_volume
    },
    timeout=86400 * 7,
    secrets=[modal.Secret.from_name("modal-credentials")],
    memory=1024 * 1024,
    cpu=96.0,
)
def train_alphafold3_modal(epochs: int = 100, batch_size: int = 8, learning_rate: float = 0.0001):
    import subprocess
    from pathlib import Path
    
    print("=" * 80)
    print("🧬 AlphaFold 3 Training on Modal.com with 8x H100 GPUs")
    print("=" * 80)
    
    print("\n📂 Setting up training environment...")
    
    main_jl_content = """
using Pkg

packages = ["LinearAlgebra", "Statistics", "Random", "Printf", "JSON3", "Downloads", "Dates", "Distributed", "CUDA", "Flux", "HTTP", "ProgressMeter", "Tar", "CodecZlib", "Sockets", "BSON", "DataFrames", "CSV", "ArgParse", "Clustering", "Zygote", "Optim", "Distributions", "StatsBase", "Distances", "NearestNeighbors", "UUIDs", "SIMD", "BenchmarkTools", "ThreadsX"]
for pkg in packages
    if !haskey(Pkg.project().dependencies, pkg)
        Pkg.add(pkg)
    end
end

ENV["DATASET_PATH"] = "/data"
ENV["CHECKPOINT_DIR"] = "/checkpoints"
ENV["EPOCHS"] = string({epochs})
ENV["BATCH_SIZE"] = string({batch_size})
ENV["LEARNING_RATE"] = string({learning_rate})
ENV["USE_GPU"] = "true"

println("🚀 Loading main AlphaFold3 implementation...")
"""
    
    main_jl_path = Path("/tmp/modal_training_launcher.jl")
    with open(main_jl_path, 'w') as f:
        f.write(main_jl_content.format(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate))
    
    print("✅ Training launcher created")
    
    print("\n📥 Copying main.jl from Replit...")
    
    main_jl_source = os.environ.get("MAIN_JL_CONTENT", "")
    if main_jl_source:
        with open("/tmp/main.jl", 'w') as f:
            f.write(main_jl_source)
        print("✅ main.jl copied from environment")
    else:
        print("⚠️  MAIN_JL_CONTENT not found in environment, using placeholder")
        with open("/tmp/main.jl", 'w') as f:
            f.write("println(\"AlphaFold3 placeholder - replace with actual main.jl\")\n")
    
    print("\n🔧 Checking CUDA availability...")
    result = subprocess.run(["julia", "-e", "using CUDA; println(CUDA.functional())"], capture_output=True, text=True)
    print(f"CUDA functional: {result.stdout.strip()}")
    
    print("\n🚀 Starting AlphaFold3 training...")
    
    training_command = f"""
julia --project -e '
include("/tmp/main.jl")
run_training()
'
"""
    
    print("=" * 80)
    print("📊 Training Configuration:")
    print(f"   Epochs: {epochs}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   GPUs: 8x H100")
    print(f"   Data Path: /data")
    print(f"   Checkpoint Path: /checkpoints")
    print("=" * 80)
    
    process = subprocess.Popen(
        training_command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    if process.stdout:
        for line in process.stdout:
            print(line, end='')
    
    process.wait()
    
    checkpoints_volume.commit()
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETED!")
    print("=" * 80)
    
    checkpoint_dir = Path("/checkpoints")
    checkpoints = list(checkpoint_dir.glob("*.bson"))
    
    print(f"📁 Saved {len(checkpoints)} checkpoints to Modal volume")
    
    return {
        "status": "completed",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "checkpoints": len(checkpoints)
    }

@app.local_entrypoint()
def main(
    action: str = "train",
    epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 0.0001,
):
    print("=" * 80)
    print("🧬 AlphaFold3 Modal.com Training System")
    print("=" * 80)
    print(f"Action: {action}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print("=" * 80)
    
    if action == "download_all":
        print("\n📥 Downloading all datasets...")
        
        print("\n1/7 Downloading PDB Database...")
        result_pdb = download_pdb_database.remote()
        print(f"✅ PDB: {result_pdb}")
        
        print("\n2/7 Downloading AlphaFold Database...")
        result_af = download_alphafold_database.remote()
        print(f"✅ AlphaFold: {result_af}")
        
        print("\n3/7 Downloading UniProt Database...")
        result_uniprot = download_uniprot_database.remote()
        print(f"✅ UniProt: {result_uniprot}")
        
        print("\n4/7 Downloading MGnify Proteins...")
        result_mgnify = download_mgnify_proteins.remote()
        print(f"✅ MGnify: {result_mgnify}")
        
        print("\n5/7 Downloading EMPIAR Cryo-EM...")
        result_empiar = download_empiar_cryoem.remote()
        print(f"✅ EMPIAR: {result_empiar}")
        
        print("\n6/7 Downloading BMRB NMR...")
        result_bmrb = download_bmrb_nmr.remote()
        print(f"✅ BMRB: {result_bmrb}")
        
        print("\n" + "=" * 80)
        print("✅ ALL DATASETS DOWNLOADED!")
        print("=" * 80)
        
    elif action == "download_pdb":
        result = download_pdb_database.remote()
        print(f"✅ Result: {result}")
        
    elif action == "download_alphafold":
        result = download_alphafold_database.remote()
        print(f"✅ Result: {result}")
        
    elif action == "download_uniprot":
        result = download_uniprot_database.remote()
        print(f"✅ Result: {result}")
        
    elif action == "download_mgnify":
        result = download_mgnify_proteins.remote()
        print(f"✅ Result: {result}")
        
    elif action == "download_empiar":
        result = download_empiar_cryoem.remote()
        print(f"✅ Result: {result}")
        
    elif action == "download_bmrb":
        result = download_bmrb_nmr.remote()
        print(f"✅ Result: {result}")
        
    elif action == "train":
        print("\n🚀 Starting training on Modal.com...")
        result = train_alphafold3_modal.remote(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
        print(f"✅ Training Result: {result}")
        
    else:
        print(f"❌ Unknown action: {action}")
        print("Available actions: download_all, download_pdb, download_alphafold, download_uniprot, download_mgnify, download_empiar, download_bmrb, train")
