#!/usr/bin/env bash
# search_vault.sh — CLI pour le serveur RAG+Reranker
# Usage: search_vault.sh [vault] "question" [top_k] [--no-rerank]
set -euo pipefail

# Parsing args
RERANK="true"
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--no-rerank" ]]; then
        RERANK="false"
    else
        ARGS+=("$arg")
    fi
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

if [[ "${1:-}" =~ ^(void|linux|browsing|terminal|llm|images|telephone|obsidian|all)$ ]]; then
    VAULT="$1"; QUERY="${2:-}"; TOP_K="${3:-5}"
else
    VAULT="void"; QUERY="${1:-}"; TOP_K="${2:-3}"
fi

if [[ -z "$QUERY" ]]; then
    echo "Usage: search_vault.sh [void|linux|browsing|terminal|llm|images|telephone|obsidian|all] \"question\" [top_k] [--no-rerank]" >&2
    exit 1
fi

RAG_URL="http://127.0.0.1:8182/search"

if ! curl -sf http://127.0.0.1:8182/health > /dev/null 2>&1; then
    echo "⚠️  Serveur RAG (port 8182) hors ligne." >&2
    exit 1
fi

JSON_PAYLOAD=$(jq -n \
    --arg v "$VAULT" --arg q "$QUERY" \
    --argjson k "$TOP_K" --argjson r "$RERANK" \
    '{vault: $v, query: $q, top_k: $k, rerank: $r}')

HTTP_CODE=$(curl -s -o /tmp/rag_response.json -w "%{http_code}" \
    -X POST "$RAG_URL" -H "Content-Type: application/json" -d "$JSON_PAYLOAD")

if [[ "$HTTP_CODE" -ne 200 ]]; then
    echo "⚠️  Erreur HTTP $HTTP_CODE" >&2
    cat /tmp/rag_response.json >&2
    exit 1
fi

ERROR=$(jq -r '.error // empty' /tmp/rag_response.json)
[[ -n "$ERROR" ]] && { echo "⚠️  $ERROR" >&2; exit 1; }

COUNT=$(jq -r '.count // 0' /tmp/rag_response.json)
VAULT_NAME=$(jq -r '.vault // "?"' /tmp/rag_response.json)
RERANKED=$(jq -r '.reranked // false' /tmp/rag_response.json)
ELAPSED=$(jq -r '.elapsed_ms // "?"' /tmp/rag_response.json)
VAULT_TITLE=$(echo "$VAULT_NAME" | tr '[:lower:]' '[:upper:]')

[[ "$COUNT" -eq 0 ]] && { echo "Aucun résultat dans $VAULT_NAME."; exit 0; }

if [[ "$RERANKED" == "true" ]]; then
    echo "=== CONTEXTE $VAULT_TITLE ($COUNT extraits | reranké | ${ELAPSED}ms) ==="
else
    echo "=== CONTEXTE $VAULT_TITLE ($COUNT extraits | RRF seul | ${ELAPSED}ms) ==="
fi

jq -r '.results[] | "\u001b[36m\(.source)\u001b[0m \u001b[33m| \(.confidence)% | rerank: \(.rerank_score // "N/A")\u001b[0m\n\u001b[97m\(.path)\u001b[0m\n\(.text)\n"' /tmp/rag_response.json
echo "=== FIN CONTEXTE $VAULT_TITLE ==="
