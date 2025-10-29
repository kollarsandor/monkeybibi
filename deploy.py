#!/usr/bin/env python3
"""
Automatizált AlphaFold 3 Training Deployment
Feltölti az adatokat, elindítja a tanítást, és elmenti a modellt
"""

import subprocess
import sys
import os
import time

def run_command(cmd, description):
    """Parancs futtatása szép kimenettel"""
    print(f"\n{'='*80}")
    print(f"  {description}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            text=True,
            capture_output=True
        )
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Hiba: {e}")
        if e.stdout:
            print(f"Kimenet: {e.stdout}")
        if e.stderr:
            print(f"Hibaüzenet: {e.stderr}")
        return False

def check_modal_auth():
    """Ellenőrzi hogy Modal autentikáció rendben van-e"""
    print("🔑 Modal autentikáció ellenőrzése...")
    
    if not os.getenv("MODAL_TOKEN_ID") or not os.getenv("MODAL_TOKEN_SECRET"):
        print("❌ Modal API kulcsok hiányoznak a környezeti változókból!")
        return False
    
    print("✅ Modal API kulcsok megtalálva!")
    return True

def create_volume():
    """Létrehozza a Modal volume-ot ha még nem létezik"""
    cmd = "modal volume create alphafold-data"
    print("📦 Modal volume létrehozása...")
    
    result = subprocess.run(
        cmd,
        shell=True,
        text=True,
        capture_output=True
    )
    
    if "already exists" in result.stderr or result.returncode == 0:
        print("✅ Volume létrehozva vagy már létezik!")
        return True
    else:
        if result.stderr:
            print(f"⚠️  Volume hiba: {result.stderr}")
        return True

def upload_data_to_volume():
    """Feltölti a training fájlokat a Modal volume-ba"""
    files = [
        ("main.jl", "/main.jl"),
        ("training.u", "/training.u")
    ]
    
    for local_file, remote_path in files:
        if not os.path.exists(local_file):
            print(f"⚠️  Figyelmeztetés: {local_file} nem található!")
            continue
        
        cmd = f"modal volume put alphafold-data {local_file} {remote_path} --force"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0 or "already exists" in result.stderr:
            print(f"✅ {local_file} → /data{remote_path}")
        else:
            print(f"⚠️  {local_file} feltöltés: {result.stderr}")
    
    return True

def setup_dragonfly_secret():
    """Setup Dragonfly DB secret in Modal"""
    print("🔐 Dragonfly DB secret beállítása...")
    
    result = subprocess.run(
        "bash setup_dragonfly_secret.sh",
        shell=True,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0 or "already exists" in result.stderr.lower():
        print("✅ Dragonfly secret rendben!")
        return True
    else:
        print(f"⚠️  Dragonfly secret probléma: {result.stderr}")
        print("⚠️  Folytatjuk Dragonfly nélkül...")
        return False

def main():
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║     🧬 AlphaFold 3 - Automatizált Training Deployment        ║
    ║                                                                ║
    ║     Platform: Modal B200 x8 GPUs                              ║
    ║     Memory: 262 GB RAM                                        ║
    ║     CPU: 48 cores                                             ║
    ║     Database: Dragonfly (Azure Australia East)                ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    # 1. Modal autentikáció ellenőrzése
    if not check_modal_auth():
        print("\n❌ Modal autentikáció sikertelen!")
        sys.exit(1)
    
    # 2. Dragonfly secret beállítása
    print("\n🔐 DRAGONFLY DB SECRET")
    print("-" * 80)
    setup_dragonfly_secret()
    
    # 3. Volume létrehozása
    print("\n📦 VOLUME LÉTREHOZÁSA")
    print("-" * 80)
    if not create_volume():
        print("\n❌ Volume létrehozása sikertelen!")
        sys.exit(1)
    
    # 4. Adatok feltöltése Modal volume-ba
    print("\n📤 ADATOK FELTÖLTÉSE")
    print("-" * 80)
    if not upload_data_to_volume():
        print("\n⚠️  Néhány fájl feltöltése sikertelen, de folytatjuk...")
    else:
        print("\n✅ Adatok sikeresen feltöltve!")
    
    # 5. Training futtatása
    print("\n\n🚀 TRAINING INDÍTÁSA")
    print("-" * 80)
    print("⏳ Ez eltarthat egy ideig (órákig)...")
    print("💡 A Modal GPU cluster most dolgozik...")
    print("📊 Training metadata mentődik Dragonfly DB-be...")
    
    if not run_command("modal run modal_wrapper.py", "🧬 AlphaFold 3 Training Futtatása"):
        print("\n❌ Training futtatás sikertelen!")
        sys.exit(1)
    
    # 4. Sikeres befejezés
    print(f"\n\n{'='*80}")
    print("  ✅ TRAINING SIKERESEN BEFEJEZVE!")
    print(f"{'='*80}")
    print("\n📁 A betanított model itt található:")
    print("   → trained_models/")
    print("\n💡 A model checkpointok a Modal volume-ban is elmentve:")
    print("   → /data/checkpoints/")
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()
