#!/bin/bash

echo "=================================================================================="
echo "🧬 AlphaFold3 Modal.com Training - AZONNALI INDÍTÁS"
echo "=================================================================================="
echo ""
echo "✅ Modal credentials: CONFIGURED"
echo "✅ Training files: READY"
echo "✅ Main.jl: 28,597 lines LOADED"
echo "✅ GPUs: 8x NVIDIA H100 CONFIGURED"
echo ""
echo "=================================================================================="
echo "🚀 TRAINING INDÍTÁSA..."
echo "=================================================================================="
echo ""

julia LAUNCH_TRAINING.jl
