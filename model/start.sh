#!/bin/bash

cd "$(dirname "$0")"

echo "🚀 AlphaFold3 Julia Backend Starting"
echo "  Strategy: Fast-start with async model loading"
echo "  Port: 6000"
echo "  Full main.jl implementation (15,000 lines)"
echo ""

# Check if HTTP package is available
if ! julia --project=. -e 'using HTTP' 2>/dev/null; then
    echo "📦 Julia packages not ready, installing now..."
    echo "⏳ This may take several minutes on first run..."
    
    julia --project=. -e 'using Pkg; Pkg.instantiate()' || {
        echo "❌ Failed to install Julia dependencies"
        exit 1
    }
    
    echo "✅ Package installation complete!"
fi

echo "✅ Starting server..."
julia --project=. fast_server.jl
