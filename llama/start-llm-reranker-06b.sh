#!/bin/bash
# Qwen3-Reranker-0.6B (Q4_K_M, Voodisss) — cross-encoder reranker
# Port 8184 | Les 3 flags sont OBLIGATOIRES : --reranking --pooling rank --embedding
# Sans les 3 → "This server does not support reranking"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"

export LD_LIBRARY_PATH="$(dirname "$LLAMA_CPP_BIN"):$LD_LIBRARY_PATH"

if command -v ss &> /dev/null && ss -tln | grep -q :8184; then
    echo "⚠️  Port 8184 déjà occupé. pkill -f Qwen3-Reranker"
    exit 1
fi

cd "$(dirname "$LLAMA_CPP_BIN")/.."
exec "$LLAMA_CPP_BIN" \
  -m "${GGUF_DIR}/Qwen3-Reranker-0.6B-Q4_K_M.gguf" \
  --reranking \
  --pooling rank \
  --embedding \
  --n-gpu-layers 0 \
  --threads 6 \
  --ctx-size 1024 \
  -ub 1024 \
  --cache-ram 0 \
  --host 127.0.0.1 \
  --port 8184 \
  --no-mmap \
  -ctk q8_0 \
  -ctv q8_0
