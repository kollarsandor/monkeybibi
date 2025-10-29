#!/usr/bin/env bash

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate

pip install --upgrade pip
pip install modal dagshub boto3 requests

echo "Modal CLI installed. Run: modal setup"
echo "Then: modal volume put alphafold-data main.jl /data/main.jl"
echo "Then: modal volume put alphafold-data training.u /data/training.u"
echo "Finally: modal run modal_wrapper.py"
