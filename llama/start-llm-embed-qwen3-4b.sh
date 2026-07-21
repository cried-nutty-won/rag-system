SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"

export LD_LIBRARY_PATH="$(dirname "$LLAMA_CPP_BIN"):$LD_LIBRARY_PATH"

if command -v ss &> /dev/null && ss -tln | grep -q :8181; then
    echo "⚠️  Port 8181 déjà occupé. pkill -f Qwen3-Embedding"
    exit 1
fi

cd "$(dirname "$LLAMA_CPP_BIN")/.."
exec "$LLAMA_CPP_BIN" \
  -m "${GGUF_DIR}/Qwen3-Embedding-4B-Q4_K_M.gguf \
  --embedding \
  --pooling last \
  --n-gpu-layers 0 \
  --threads 4 \
  --ctx-size 8192 \
  -ub 8192 \
  --parallel 1 \
  --host 127.0.0.1 \
  --port 8181 \
  --no-mmap \
  -ctk q8_0 \
  -ctv q8_0 \
  > /tmp/llm-embed-06b.log 2>&1
