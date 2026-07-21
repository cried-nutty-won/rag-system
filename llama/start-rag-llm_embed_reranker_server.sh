#!/bin/bash
# start-rag-llm_embed_reranker_server.sh — Démarre toute la stack RAG
# Usage: start-rag-llm_embed_reranker_server.sh [--with-0.6b-rerank]
set -euo pipefail

echo "=== Démarrage stack RAG ==="

# 1. Embedding Qwen3-0.6B-q8 (port 8181)
if ss -tln 2>/dev/null | grep -q :8181; then
    echo "  ✓ Embedding déjà actif (8181)"
else
    echo "  → Lancement embedding Qwen3-0.6B..."
    ~/scripts/llm/rag/start-llm-embed-qwen3-06b.sh &
    sleep 4
fi

# 2. Reranker Qwen3-0.6B-q4 (port 8184)
if ss -tln 2>/dev/null | grep -q :8184; then
    echo "  ✓ Reranker 0.6B déjà actif (8184)"
else
    echo "  → Lancement reranker 0.6B..."
    ~/scripts/llm/rag/start-llm-reranker-06b.sh &
    sleep 3
fi

# 3. Serveur RAG Python (port 8182)
if ss -tln 2>/dev/null | grep -q :8182; then
    echo "  ✓ RAG déjà actif (8182)"
else
    echo "  → Lancement serveur RAG..."
    ~/.venv/main/bin/python3 /home/ksoinan/scripts/rag/rag_server_rerank.py \
        > /tmp/rag_server_rerank.log 2>&1 &
    sleep 2
fi

echo ""
echo "=== Stack prête ==="
echo "  Santé : curl -s http://127.0.0.1:8182/health | jq ."
echo "  Test  : rag void \"ma question\" 3"
echo "  Logs  : tail -f /tmp/rag_server_rerank.log"
