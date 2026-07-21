# RAG — Quick Start Guide

## Startup

| Command | Action |
|---------|--------|
| `llmers` | Full stack (embedding + reranker + RAG server) |
| `llmes` | Embedding + RAG server (no reranker) |
| `llme` | Embedding only (port 8181) |
| `llmr` | Reranker only (port 8184) |
| `rs` | Python RAG server only (port 8182) |
| `rst` | `tail -f` RAG server logs |
| `rc` | Health check all 3 services |
| `rsk` | Kill Python RAG server only |
| `stop` | Kill all llama-server processes |

## Search

| Command | Mode | Latency |
|---------|------|---------|
| `rag void "my question"` | RRF only (fast) | ~20 ms |
| `ragr void "my question"` | RRF + Reranker (precise) | ~10-20 s |
| `rag all "my question" 5` | All vaults, top 5 | ~130 ms |
| `rag obsidian "my question"` | Obsidian vaults only | ~130 ms |
| `ragr linux "nftables config" 10` | Linux vault, top 10, reranked | ~10-20 s |

## Available Vaults

`void` · `linux` · `browsing` · `terminal` · `llm` · `images` · `telephone` · `obsidian` · `all`

- `obsidian`: all Obsidian vaults (without external vaults)
- `all`: all vaults (Obsidian + external)

## Health Check

| Command | Action |
|---------|--------|
| `rc` | Health check all 3 services (embedding, reranker, RAG) |
| `curl -s http://127.0.0.1:8182/health \| jq .` | Full details |

## Isolated Reranker Test

```bash
curl -s http://127.0.0.1:8184/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"query":"nftables","documents":["nftables linux network filter","nice weather today"]}' | jq .
```

## Shutdown

| Command | Action |
|---------|--------|
| `stop` | Kill all llama-server + purge memory |
| `rsk` | Kill Python RAG server only |

## After Modifying an Obsidian File

```bash
rsk
rs
```

The cache automatically detects modified chunks. Only new chunks are re-embedded.

## After Changing the Embedding Model

```bash
rm -f ~/.rag/*__qwen3-embed-06b*
rsk
rs
```

## Logs

| Command | Action |
|---------|--------|
| `rst` | `tail -f` RAG server logs |
| `tail -f /tmp/llm-embed-06b.log` | Embedding logs |

## Current Parameters

| Parameter | Value |
|-----------|-------|
| RERANK_CANDIDATES | 18 |
| Default top_k | 5 |
| alpha_ratio (embeddable threshold) | 0.36 |
| Embedding ctx-size | 8192 |
| Reranker ctx-size | 1024 |
| KV cache | q8_0 (both) |
| Batch size (-ub) | 8192 (embed) / 1024 (reranker) |
| Host prompt cache | disabled (`--cache-ram 0`) |
```
