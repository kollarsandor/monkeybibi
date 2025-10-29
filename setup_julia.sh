#!/bin/bash

echo "=========================================="
echo "🔧 JULIA TELJES TELEPÍTÉS - EGYSZER ÉS MINDENKORRA"
echo "=========================================="

cd model || exit 1

# Check if already installed
if [ -f ".julia_setup_complete" ]; then
    echo "✅ Julia csomagok már telepítve vannak!"
    echo "   Ha újra szeretnéd telepíteni, töröld a .julia_setup_complete fájlt"
    exit 0
fi

echo ""
echo "📦 1. Project.toml tisztítása PyCall nélkül..."

# Remove PyCall dependency (causes problems)
cat > Project.toml << 'EOF'
[deps]
ArgParse = "c7e460c6-2fb9-53a9-8c5b-16f535851c63"
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
Clustering = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
CodecZlib = "944b1d66-785c-5afd-91f1-9de20f533193"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Downloads = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
NearestNeighbors = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
Optim = "429524aa-4258-5aef-a3af-852621145aeb"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Profile = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
ProgressMeter = "92933f4c-e287-5a05-a399-4b506db050ca"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SIMD = "fdea26ae-647d-5447-a871-4b548cad5224"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
Tar = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
ThreadsX = "ac1d9e8a-700a-412c-b207-f0111f4b6c0d"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
EOF

echo "✅ Project.toml frissítve (PyCall eltávolítva)"

echo ""
echo "📦 2. Manifest.toml törlése (tiszta újrakezdés)..."
rm -f Manifest.toml
echo "✅ Manifest.toml törölve"

echo ""
echo "📦 3. Julia csomagok telepítése (ez kb. 2-3 percet vesz igénybe)..."
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.resolve(); Pkg.precompile()' || {
    echo "❌ Hiba a telepítés során"
    exit 1
}

echo ""
echo "📦 4. HTTP csomag explicit telepítése..."
julia --project=. -e 'using Pkg; Pkg.add("HTTP"); Pkg.precompile()' || {
    echo "⚠️  HTTP csomag telepítési hiba (lehet már telepítve van)"
}

echo ""
echo "📦 5. Precompilation - gyorsítás következő indításokhoz..."
julia --project=. -e 'using Pkg; Pkg.precompile()' || {
    echo "⚠️  Precompilation nem sikerült teljesen (nem kritikus)"
}

echo ""
echo "📦 6. Telepítési marker létrehozása..."
touch .julia_setup_complete
echo "$(date)" >> .julia_setup_complete

echo ""
echo "=========================================="
echo "✅ TELEPÍTÉS KÉSZ!"
echo "=========================================="
echo ""
echo "A Julia backend mostantól minden indításkor:"
echo "  ✅ NEM fog újratelepíteni semmit"
echo "  ✅ Gyorsan elindul"
echo "  ✅ Használja az előre lefordított csomagokat"
echo ""
echo "Ha problémád van, töröld a .julia_setup_complete fájlt és futtasd újra!"
echo ""
