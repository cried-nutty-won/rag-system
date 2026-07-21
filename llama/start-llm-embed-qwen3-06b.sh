#!/bin/bash
# Qwen3-Embedding-0.6B — serveur d'embedding
# Port 8181 | pooling last (hidden state du token [EOS])
# Dimensions : 2560 | Multilingue 100+ langues | ctx 8192

export LD_LIBRARY_PATH=/home/ksoinan/llama-cpp-turboquant/build-cpu/bin:$LD_LIBRARY_PATH

if command -v ss &> /dev/null && ss -tln | grep -q :8181; then
    echo "⚠️  Port 8181 déjà occupé. pkill -f Qwen3-Embedding"
    exit 1
fi

cd ~/llama-cpp-turboquant
exec ./build-cpu/bin/llama-server \
  -m /home/ksoinan/wijdha/library/GGUF/rag/Qwen3-Embedding-0.6B-Q8_0.gguf \
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

  
