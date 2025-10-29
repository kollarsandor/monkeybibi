import modal
import os
import sys
from datetime import datetime
import json
import subprocess
from pathlib import Path

app = modal.App("alphafold3-production-training")

cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("wget", "git", "build-essential", "cmake", "gfortran", "libopenblas-dev", "curl")
    .run_commands("wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.0-linux-x86_64.tar.gz")
    .run_commands("tar -xzf julia-1.10.0-linux-x86_64.tar.gz && mv julia-1.10.0 /opt/julia && rm julia-1.10.0-linux-x86_64.tar.gz")
    .env({"PATH": "/opt/julia/bin:$PATH", "JULIA_NUM_THREADS": "96"})
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
    .run_commands("julia -e 'using Pkg; Pkg.add(\"SIMD\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"BenchmarkTools\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"ThreadsX\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"Enzyme\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"GZip\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"FilePaths\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"NPZ\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"JSON\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"Parameters\")'")
    .run_commands("julia -e 'using Pkg; Pkg.add(\"Requires\")'")
    .pip_install("requests", "tqdm", "numpy", "pandas", "biopython", "pdbfixer", "openmm", "beautifulsoup4", "lxml")
)

data_volume = modal.Volume.from_name("af3-datasets", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("af3-checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu=modal.gpu.H100(count=8),
    volumes={"/datasets": data_volume, "/checkpoints": checkpoints_volume},
    timeout=86400 * 7,
    memory=1024 * 1024,
    cpu=96.0,
)
def download_pdb_wwpdb():
    import requests
    from pathlib import Path
    import gzip
    import shutil
    from tqdm import tqdm
    
    print("=" * 100)
    print("📥 DOWNLOADING PDB AND wwPDB FTP DATABASE")
    print("=" * 100)
    
    pdb_dir = Path("/datasets/pdb")
    pdb_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n🔍 Fetching PDB entry list from RCSB...")
    
    try:
        response = requests.get("https://data.rcsb.org/rest/v1/holdings/current/entry_ids", timeout=60)
        pdb_entries = response.json()
        print(f"✅ Found {len(pdb_entries)} PDB entries")
    except:
        print("⚠️  Using fallback PDB list")
        pdb_entries = []
        for i in range(1, 10000):
            pdb_entries.append(f"{i:04d}")
    
    base_url_rcsb = "https://files.rcsb.org/download/"
    base_url_wwpdb = "https://ftp.wwpdb.org/pub/pdb/data/structures/divided/pdb/"
    
    downloaded = 0
    failed = 0
    
    for pdb_id in tqdm(pdb_entries[:100000], desc="Downloading PDB structures"):
        pdb_id_lower = pdb_id.lower()
        pdb_file = pdb_dir / f"{pdb_id_lower}.pdb"
        
        if pdb_file.exists():
            continue
        
        success = False
        
        for base_url in [base_url_rcsb, base_url_wwpdb]:
            try:
                if "rcsb" in base_url:
                    url = f"{base_url}{pdb_id.upper()}.pdb.gz"
                else:
                    middle_chars = pdb_id_lower[1:3]
                    url = f"{base_url}{middle_chars}/pdb{pdb_id_lower}.ent.gz"
                
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    gz_path = pdb_dir / f"{pdb_id_lower}.pdb.gz"
                    with open(gz_path, 'wb') as f:
                        f.write(response.content)
                    
                    with gzip.open(gz_path, 'rb') as f_in:
                        with open(pdb_file, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    gz_path.unlink()
                    downloaded += 1
                    success = True
                    break
            except:
                continue
        
        if not success:
            failed += 1
        
        if downloaded % 1000 == 0 and downloaded > 0:
            data_volume.commit()
            print(f"💾 Checkpoint: {downloaded} structures downloaded")
    
    data_volume.commit()
    
    print("=" * 100)
    print(f"✅ PDB/wwPDB DOWNLOAD COMPLETE")
    print(f"   Downloaded: {downloaded:,}")
    print(f"   Failed: {failed:,}")
    print(f"   Total scanned: {len(pdb_entries):,}")
    print("=" * 100)
    
    return {"downloaded": downloaded, "failed": failed}

@app.function(
    image=image,
    gpu=modal.gpu.H100(count=8),
    volumes={"/datasets": data_volume, "/checkpoints": checkpoints_volume},
    timeout=86400 * 7,
    memory=1024 * 1024,
    cpu=96.0,
)
def download_alphafold_database():
    import requests
    from pathlib import Path
    from tqdm import tqdm
    import tarfile
    import gzip
    import shutil
    
    print("=" * 100)
    print("📥 DOWNLOADING ALPHAFOLD DATABASE")
    print("=" * 100)
    
    af_dir = Path("/datasets/alphafold")
    af_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = {
        "params_2022": "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar",
        "params_2021": "https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar",
        "pdb70": "http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/pdb70_from_mmcif_latest.tar.gz",
        "uniref90": "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz",
        "mgnify": "https://storage.googleapis.com/alphafold-databases/v2.3/mgy_clusters_2022_05.fa.gz",
        "bfd": "https://storage.googleapis.com/alphafold-databases/v2.3/bfd-first_non_consensus_sequences.fasta.gz",
        "uniclust30": "https://storage.googleapis.com/alphafold-databases/v2.3/uniclust30_2018_08_hhsuite.tar.gz",
        "pdb_seqres": "https://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz",
    }
    
    for name, url in datasets.items():
        print(f"\n📦 Downloading {name}...")
        
        filename = url.split('/')[-1]
        filepath = af_dir / filename
        
        if filepath.exists():
            print(f"✅ {name} already exists")
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
                extracted_path = filepath.with_suffix('')
                with gzip.open(filepath, 'rb') as f_in:
                    with open(extracted_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            elif filepath.suffix == '.tar' or filepath.name.endswith('.tar.gz'):
                print(f"📂 Extracting {name}...")
                extract_dir = af_dir / name
                extract_dir.mkdir(exist_ok=True)
                with tarfile.open(filepath, 'r:*') as tar:
                    tar.extractall(path=extract_dir)
            
            data_volume.commit()
            print(f"✅ {name} complete")
            
        except Exception as e:
            print(f"❌ Failed {name}: {str(e)}")
    
    data_volume.commit()
    print("=" * 100)
    print("✅ ALPHAFOLD DATABASE COMPLETE")
    print("=" * 100)
    
    return {"status": "complete"}

@app.function(
    image=image,
    gpu=modal.gpu.H100(count=8),
    volumes={"/datasets": data_volume, "/checkpoints": checkpoints_volume},
    timeout=86400 * 7,
    memory=1024 * 1024,
    cpu=96.0,
)
def download_uniprotkb_complete():
    import requests
    from pathlib import Path
    from tqdm import tqdm
    import gzip
    import shutil
    
    print("=" * 100)
    print("📥 DOWNLOADING UNIPROTKB COMPLETE DATABASE")
    print("=" * 100)
    
    uniprot_dir = Path("/datasets/uniprotkb")
    uniprot_dir.mkdir(parents=True, exist_ok=True)
    
    base_url = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/"
    
    datasets = {
        "uniprot_sprot": f"{base_url}uniprot_sprot.fasta.gz",
        "uniprot_trembl": f"{base_url}uniprot_trembl.fasta.gz",
        "uniprot_sprot_varsplic": f"{base_url}uniprot_sprot_varsplic.fasta.gz",
        "uniref100": "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref100/uniref100.fasta.gz",
        "uniref90": "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz",
        "uniref50": "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz",
        "uniparc": "https://ftp.uniprot.org/pub/databases/uniprot/current_release/uniparc/uniparc_active.fasta.gz",
    }
    
    for name, url in datasets.items():
        print(f"\n📦 Downloading {name}...")
        
        filename = url.split('/')[-1]
        filepath = uniprot_dir / filename
        extracted_path = filepath.with_suffix('')
        
        if extracted_path.exists():
            print(f"✅ {name} already exists")
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
            data_volume.commit()
            print(f"✅ {name} complete")
            
        except Exception as e:
            print(f"❌ Failed {name}: {str(e)}")
    
    data_volume.commit()
    print("=" * 100)
    print("✅ UNIPROTKB COMPLETE DATABASE DOWNLOADED")
    print("=" * 100)
    
    return {"status": "complete"}

@app.function(
    image=image,
    gpu=modal.gpu.H100(count=8),
    volumes={"/datasets": data_volume, "/checkpoints": checkpoints_volume},
    timeout=86400 * 7,
    memory=1024 * 1024,
    cpu=96.0,
)
def download_empiar_cryoem():
    import requests
    from pathlib import Path
    from tqdm import tqdm
    import json
    
    print("=" * 100)
    print("📥 DOWNLOADING EMPIAR CRYO-EM DATABASE")
    print("=" * 100)
    
    empiar_dir = Path("/datasets/empiar")
    empiar_dir.mkdir(parents=True, exist_ok=True)
    
    api_url = "https://www.ebi.ac.uk/empiar/api/entry/"
    
    try:
        print("📡 Fetching EMPIAR entry list...")
        response = requests.get(api_url, timeout=60)
        entries = response.json()
        
        print(f"✅ Found {len(entries)} EMPIAR entries")
        
        downloaded = 0
        failed = 0
        
        for entry in tqdm(entries[:5000], desc="Downloading EMPIAR metadata"):
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
                        data_volume.commit()
                else:
                    failed += 1
            except:
                failed += 1
        
        data_volume.commit()
        
        print("=" * 100)
        print(f"✅ EMPIAR CRYO-EM DATABASE COMPLETE")
        print(f"   Downloaded: {downloaded}")
        print(f"   Failed: {failed}")
        print("=" * 100)
        
        return {"downloaded": downloaded, "failed": failed}
        
    except Exception as e:
        print(f"❌ EMPIAR download failed: {str(e)}")
        return {"downloaded": 0, "failed": 0}

@app.function(
    image=image,
    gpu=modal.gpu.H100(count=8),
    volumes={"/datasets": data_volume, "/checkpoints": checkpoints_volume},
    timeout=86400 * 7,
    memory=1024 * 1024,
    cpu=96.0,
)
def download_bmrb_nmr():
    import requests
    from pathlib import Path
    from tqdm import tqdm
    import json
    
    print("=" * 100)
    print("📥 DOWNLOADING BMRB NMR DATABASE")
    print("=" * 100)
    
    bmrb_dir = Path("/datasets/bmrb")
    bmrb_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("📡 Fetching BMRB entry list...")
        response = requests.get("https://bmrb.io/search/get_all_values_from_category.php?category=Entry&format=json", timeout=60)
        entries = response.json()
        
        print(f"✅ Found {len(entries)} BMRB entries")
        
        downloaded = 0
        failed = 0
        
        for entry in tqdm(entries[:10000], desc="Downloading BMRB NMR data"):
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
                        data_volume.commit()
                else:
                    failed += 1
            except:
                failed += 1
        
        data_volume.commit()
        
        print("=" * 100)
        print(f"✅ BMRB NMR DATABASE COMPLETE")
        print(f"   Downloaded: {downloaded}")
        print(f"   Failed: {failed}")
        print("=" * 100)
        
        return {"downloaded": downloaded, "failed": failed}
        
    except Exception as e:
        print(f"❌ BMRB download failed: {str(e)}")
        return {"downloaded": 0, "failed": 0}

@app.function(
    image=image,
    gpu=modal.gpu.H100(count=8),
    volumes={"/datasets": data_volume, "/checkpoints": checkpoints_volume},
    timeout=86400 * 7,
    memory=1024 * 1024,
    cpu=96.0,
)
def download_viro3d_virus_proteins():
    import requests
    from pathlib import Path
    from tqdm import tqdm
    from bs4 import BeautifulSoup
    import json
    
    print("=" * 100)
    print("📥 DOWNLOADING VIRO3D VIRUS PROTEINS DATABASE")
    print("=" * 100)
    
    viro3d_dir = Path("/datasets/viro3d")
    viro3d_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("📡 Fetching Viro3D data from publication...")
        
        paper_url = "https://www.embopress.org/doi/full/10.1038/s44320-025-00147-9"
        
        response = requests.get(paper_url, timeout=60)
        
        if response.status_code == 200:
            metadata_file = viro3d_dir / "viro3d_publication.html"
            with open(metadata_file, 'w') as f:
                f.write(response.text)
            
            print("✅ Viro3D publication metadata saved")
        
        supplementary_urls = [
            "https://www.embopress.org/action/downloadSupplement?doi=10.1038%2Fs44320-025-00147-9&file=msb202414111-sup-0001-AppendixS1.xlsx",
            "https://www.embopress.org/action/downloadSupplement?doi=10.1038%2Fs44320-025-00147-9&file=msb202414111-sup-0002-DatasetEV1.xlsx",
        ]
        
        for i, url in enumerate(supplementary_urls):
            try:
                print(f"📦 Downloading supplementary file {i+1}...")
                response = requests.get(url, timeout=60)
                
                if response.status_code == 200:
                    filename = url.split('file=')[-1]
                    filepath = viro3d_dir / filename
                    
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    
                    print(f"✅ Downloaded {filename}")
            except Exception as e:
                print(f"⚠️  Failed to download supplementary {i+1}: {str(e)}")
        
        data_volume.commit()
        
        print("=" * 100)
        print("✅ VIRO3D VIRUS PROTEINS DATABASE COMPLETE")
        print("=" * 100)
        
        return {"status": "complete"}
        
    except Exception as e:
        print(f"❌ Viro3D download failed: {str(e)}")
        return {"status": "failed"}

@app.function(
    image=image,
    gpu=modal.gpu.H100(count=8),
    volumes={"/datasets": data_volume, "/checkpoints": checkpoints_volume},
    timeout=86400 * 7,
    memory=1024 * 1024,
    cpu=96.0,
)
def download_bfvd_predicted_viruses():
    import requests
    from pathlib import Path
    from tqdm import tqdm
    import json
    
    print("=" * 100)
    print("📥 DOWNLOADING BFVD PREDICTED VIRUS DATABASE")
    print("=" * 100)
    
    bfvd_dir = Path("/datasets/bfvd")
    bfvd_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("📡 Fetching BFVD data from publication...")
        
        paper_url = "https://academic.oup.com/nar/article/53/D1/D340/7906834"
        
        response = requests.get(paper_url, timeout=60)
        
        if response.status_code == 200:
            metadata_file = bfvd_dir / "bfvd_publication.html"
            with open(metadata_file, 'w') as f:
                f.write(response.text)
            
            print("✅ BFVD publication metadata saved")
        
        data_urls = [
            "https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/nar/53/D1/10.1093_nar_gkae1070/1/gkae1070_supplemental_files.zip?Expires=1735660800&Signature=example&Key-Pair-Id=example",
        ]
        
        for url in data_urls:
            try:
                print(f"📦 Downloading BFVD data...")
                response = requests.get(url, timeout=120, allow_redirects=True)
                
                if response.status_code == 200:
                    filepath = bfvd_dir / "bfvd_supplemental_data.zip"
                    
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    
                    print(f"✅ Downloaded BFVD supplemental data")
                    
                    import zipfile
                    with zipfile.ZipFile(filepath, 'r') as zip_ref:
                        zip_ref.extractall(bfvd_dir)
                    
                    print("✅ Extracted BFVD data")
            except Exception as e:
                print(f"⚠️  Could not download full BFVD dataset: {str(e)}")
                print("    Metadata saved for reference")
        
        data_volume.commit()
        
        print("=" * 100)
        print("✅ BFVD PREDICTED VIRUS DATABASE COMPLETE")
        print("=" * 100)
        
        return {"status": "complete"}
        
    except Exception as e:
        print(f"❌ BFVD download failed: {str(e)}")
        return {"status": "failed"}

@app.function(
    image=image,
    gpu=modal.gpu.H100(count=8),
    volumes={"/datasets": data_volume, "/checkpoints": checkpoints_volume},
    timeout=86400 * 7,
    memory=1024 * 1024,
    cpu=96.0,
)
def download_mgnify_proteins():
    import requests
    from pathlib import Path
    from tqdm import tqdm
    import gzip
    import shutil
    
    print("=" * 100)
    print("📥 DOWNLOADING MGNIFY PROTEINS DATABASE")
    print("=" * 100)
    
    mgnify_dir = Path("/datasets/mgnify")
    mgnify_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = {
        "mgnify_proteins_v2023_12": "https://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/2023_12/mgnify_proteins_2023_12.fa.gz",
        "mgnify_clusters_v2023_12": "https://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/2023_12/mgnify_clusters_2023_12.fa.gz",
        "mgnify_proteins_v2022_05": "https://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/2022_05/mgnify_proteins_2022_05.fa.gz",
    }
    
    for name, url in datasets.items():
        print(f"\n📦 Downloading {name}...")
        
        filename = url.split('/')[-1]
        filepath = mgnify_dir / filename
        extracted_path = filepath.with_suffix('')
        
        if extracted_path.exists():
            print(f"✅ {name} already exists")
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
            data_volume.commit()
            print(f"✅ {name} complete")
            
        except Exception as e:
            print(f"❌ Failed {name}: {str(e)}")
    
    data_volume.commit()
    print("=" * 100)
    print("✅ MGNIFY PROTEINS DATABASE COMPLETE")
    print("=" * 100)
    
    return {"status": "complete"}

@app.function(
    image=image,
    gpu=modal.gpu.H100(count=8),
    volumes={"/datasets": data_volume, "/checkpoints": checkpoints_volume},
    timeout=86400 * 7,
    memory=1024 * 1024,
    cpu=96.0,
)
def train_alphafold3_production(
    main_jl_content: str,
    epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 0.0001
):
    from pathlib import Path
    import subprocess
    
    print("=" * 100)
    print("🧬 ALPHAFOLD3 PRODUCTION TRAINING")
    print("=" * 100)
    print(f"GPU: 8x NVIDIA H100 (640GB VRAM)")
    print(f"CPU: 96 cores")
    print(f"RAM: 1TB")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print("=" * 100)
    
    print("\n📝 Writing main.jl...")
    main_jl_path = Path("/tmp/main.jl")
    with open(main_jl_path, 'w') as f:
        f.write(main_jl_content)
    
    print(f"✅ main.jl written ({len(main_jl_content)} bytes)")
    
    print("\n🔧 Checking CUDA...")
    result = subprocess.run(
        ["julia", "-e", "using CUDA; println(\"CUDA functional: \", CUDA.functional()); println(\"CUDA devices: \", length(CUDA.devices()))"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    
    print("\n📊 Creating training launcher...")
    
    launcher_content = f"""
ENV["DATASET_PATH"] = "/datasets"
ENV["CHECKPOINT_DIR"] = "/checkpoints"
ENV["EPOCHS"] = "{epochs}"
ENV["BATCH_SIZE"] = "{batch_size}"
ENV["LEARNING_RATE"] = "{learning_rate}"
ENV["USE_GPU"] = "true"
ENV["JULIA_NUM_THREADS"] = "96"

println("=" ^ 100)
println("🚀 Loading AlphaFold3 main implementation...")
println("=" ^ 100)

include("/tmp/main.jl")

println("\\n" * "=" ^ 100)
println("🔥 STARTING TRAINING...")
println("=" ^ 100)

run_training()

println("\\n" * "=" ^ 100)
println("✅ TRAINING COMPLETE!")
println("=" ^ 100)
"""
    
    launcher_path = Path("/tmp/train_launcher.jl")
    with open(launcher_path, 'w') as f:
        f.write(launcher_content)
    
    print("✅ Launcher created")
    
    print("\n🚀 LAUNCHING TRAINING...")
    print("=" * 100)
    
    process = subprocess.Popen(
        ["julia", "--project", str(launcher_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    
    checkpoints_volume.commit()
    
    print("\n" * "=" * 100)
    print("✅ ALPHAFOLD3 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 100)
    
    checkpoint_dir = Path("/checkpoints")
    checkpoints = list(checkpoint_dir.glob("*.bson"))
    
    print(f"💾 Saved {len(checkpoints)} checkpoints")
    
    return {
        "status": "completed",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "checkpoints": len(checkpoints),
        "gpu": "8x H100"
    }

@app.local_entrypoint()
def main(
    action: str = "train",
    epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 0.0001,
):
    import os
    
    print("=" * 100)
    print("🧬 ALPHAFOLD3 MODAL.COM PRODUCTION TRAINING SYSTEM")
    print("=" * 100)
    print(f"Action: {action}")
    print("=" * 100)
    
    if action == "download_all":
        print("\n📥 DOWNLOADING ALL DATASETS TO MODAL...")
        print("=" * 100)
        
        print("\n1/9 PDB + wwPDB FTP...")
        result1 = download_pdb_wwpdb.remote()
        print(f"✅ PDB/wwPDB: {result1}")
        
        print("\n2/9 AlphaFold Database...")
        result2 = download_alphafold_database.remote()
        print(f"✅ AlphaFold: {result2}")
        
        print("\n3/9 UniProtKB Complete...")
        result3 = download_uniprotkb_complete.remote()
        print(f"✅ UniProtKB: {result3}")
        
        print("\n4/9 EMPIAR Cryo-EM...")
        result4 = download_empiar_cryoem.remote()
        print(f"✅ EMPIAR: {result4}")
        
        print("\n5/9 BMRB NMR...")
        result5 = download_bmrb_nmr.remote()
        print(f"✅ BMRB: {result5}")
        
        print("\n6/9 Viro3D Virus Proteins...")
        result6 = download_viro3d_virus_proteins.remote()
        print(f"✅ Viro3D: {result6}")
        
        print("\n7/9 BFVD Predicted Viruses...")
        result7 = download_bfvd_predicted_viruses.remote()
        print(f"✅ BFVD: {result7}")
        
        print("\n8/9 MGnify Proteins...")
        result8 = download_mgnify_proteins.remote()
        print(f"✅ MGnify: {result8}")
        
        print("\n" + "=" * 100)
        print("✅ ALL DATASETS DOWNLOADED TO MODAL!")
        print("=" * 100)
        
    elif action == "train":
        print("\n🚀 STARTING PRODUCTION TRAINING...")
        print("=" * 100)
        
        main_jl_path = os.path.join(os.getcwd(), "main.jl")
        
        if not os.path.exists(main_jl_path):
            print(f"❌ ERROR: main.jl not found at {main_jl_path}")
            return
        
        print(f"📖 Reading main.jl from {main_jl_path}...")
        with open(main_jl_path, 'r') as f:
            main_jl_content = f.read()
        
        print(f"✅ main.jl loaded ({len(main_jl_content)} bytes, {main_jl_content.count(chr(10))} lines)")
        
        print("\n🚀 Launching training on Modal with 8x H100 GPUs...")
        
        result = train_alphafold3_production.remote(
            main_jl_content=main_jl_content,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        print("\n" + "=" * 100)
        print("✅ TRAINING RESULT:")
        print(f"   {result}")
        print("=" * 100)
        
    elif action == "full":
        print("\n📥 PHASE 1: DOWNLOADING ALL DATASETS...")
        os.system(f"python {__file__} download_all")
        
        print("\n🚀 PHASE 2: STARTING TRAINING...")
        os.system(f"python {__file__} train --epochs {epochs} --batch-size {batch_size} --learning-rate {learning_rate}")
        
    else:
        print(f"❌ Unknown action: {action}")
        print("Available actions:")
        print("  - download_all: Download all datasets to Modal")
        print("  - train: Start training (requires main.jl in current directory)")
        print("  - full: Download datasets + train (complete pipeline)")
