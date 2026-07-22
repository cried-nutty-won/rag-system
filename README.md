# RAG Hybrid + Reranker

# introduction

## What is RAG?

**RAG** (Retrieval-Augmented Generation) is an architecture that enables an LLM to answer questions by retrieving relevant documents from a knowledge base and injecting them into the context, rather than relying solely on its internal parameters.

Concretely, when you (or an LLM agent) ask a question:

1. The system **retrieves** relevant passages from your documents (Obsidian notes, technical docs, procedures)
2. It **injects** only those passages into the LLM's context
3. The LLM **answers** with source citations

Without RAG, an LLM must either rely on its frozen training knowledge (often outdated or hallucinated) or ingest entire documents into its context window (expensive and slow).

### Why use RAG?

| Benefit | Without RAG | With RAG |
|---------|-------------|----------|
| **Response speed** | LLM must process full context (~30k tokens) | Indexed search in ~20ms, only top-5 sent to LLM |
| **Token consumption** | Entire corpus injected (thousands of pages) | 5-10 relevant chunks (~2000 tokens) |
| **Ecological impact** | Maximum GPU/CPU compute per query | Compute proportional to actual relevance |
| **Privacy** | Often requires cloud APIs (OpenAI, etc.) | 100% local, no data leaves your machine |
| **Accuracy** | Frequent hallucinations on specific facts | Sourced answers, verifiable in your documents |

---

## What is a Reranker?

A **reranker** (or cross-encoder) is a second-stage retrieval model that reads the query and each candidate document **jointly**, then outputs a precise relevance score. Unlike the embedding model (bi-encoder) which encodes query and document separately into vectors, the reranker processes both inputs together through the full transformer stack, capturing fine-grained semantic interactions that vector similarity misses.

In this pipeline, the reranker receives the top-N candidates from the RRF fusion step and reorders them by true relevance:

```
RRF Top-18 Candidates
        │
        ▼
[Reranker Qwen3-0.6B]
  reads (query + document) jointly
  → P(yes) via cls.output.weight
        │
        ▼
Final Ranked Results
```

### Why use a Reranker?

| Aspect | Without Reranker (RRF only) | With Reranker |
|--------|-----------------------------|---------------|
| **Precision** | Good for obvious matches, weak on nuanced queries | Captures subtle semantic relationships, paraphrases, and domain-specific terminology |
| **False positives** | BM25 promotes documents with matching keywords but irrelevant content | Cross-encoder reads the full pair and rejects keyword-matching noise |
| **Short queries** | 2-3 word queries produce ambiguous embeddings → poor ranking | Joint encoding compensates for query brevity by leveraging document context |
| **Score interpretability** | RRF scores are arbitrary ranks, not comparable across queries | Reranker outputs calibrated P(yes) probabilities (0.0–1.0) |
| **Latency cost on CPU with 0.6B models** | ~20 ms | 12 s for 18 candidates |
| **Latency cost on GPU with 0.6B models** | ~3 ms | 1 s for 100 candidates |
| **Latency cost on GPU with 4B models** | ~30 ms | 3 s for 100 candidates |
| **Token cost to LLM** | May send irrelevant chunks, wasting context | Only the most relevant chunks reach the LLM → fewer tokens, better answers |

The reranker is the single largest quality improvement in the pipeline. In Dave Ebbelaar's FinanceQA benchmark, adding a reranker improved NDCG@10 by **+12 points** over hybrid retrieval alone. The latency cost is structural (one full forward pass per candidate), but the precision gain eliminates hallucinations and irrelevant context injection downstream.

### When to skip the reranker

- Interactive exploration where speed matters more than precision (`rag` alias)
- Queries with highly specific keywords where BM25 alone suffices
- Resource-constrained environments where +10s latency is unacceptable
- The server automatically falls back to pure RRF if the reranker is offline

---

## Who is it for?

- **Humans**: fast search across Obsidian notes, technical documentation, meeting transcripts
- **LLM Agents**: an agent can call RAG as a tool (`tool calling`) to consult your knowledge base before answering, without stuffing its context window

---

## Hardware Compatibility

### This RAG stack runs on **any hardware supported by llama.cpp**:

| Backend | Status | Notes |
|---------|--------|-------|
| CPU (x86, ARM, RISC-V) | Full support | AVX2/AVX512/NEON auto-detected |
| NVIDIA GPU (CUDA) | Full support | `--n-gpu-layers all` for max speed |
| Apple Silicon (Metal) | Full support | Unified memory, no VRAM limits |
| AMD GPU (HIP/Vulkan) | Supported | Via llama.cpp Vulkan backend |
| Intel GPU (SYCL) | Supported | Via llama.cpp SYCL backend |

No GPU required — runs entirely on CPU if needed. GPU acceleration is optional and speeds up both embedding and reranking proportionally.

---

### Other Backends

| Scenario | Recommended Backend | Why |
|----------|--------------------|----|
| Multi-user, GPU cluster (DGX, etc.) | vLLM | Native batching, PagedAttention, concurrent sessions |
| High-throughput production | SGLang | RadixAttention prefix caching, optimized scheduler |
| Quick prototyping, embedding only | Ollama | Zero-config model management |
| Mixed: embedding on GPU + reranker on CPU | llama.cpp + vLLM | Each backend serves what it does best |

> **Note:** Ollama does not support reranking. When using Ollama for embeddings disable reranking (`--no-rerank`).

---
---

# Installation

### Prerequisites

- **llama.cpp** compiled with CPU support (or CUDA/Metal/Vulkan for GPU acceleration) or vLLM, SGLang , ollama
- **Python 3.10+** with a virtual environment
- **GGUF models**:
  - Embedding: `Qwen3-Embedding-0.6B-Q8_0.gguf` ([official Qwen](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF))
  - Reranker: `Qwen3-Reranker-0.6B-Q4_K_M.gguf` (**must be from [Voodisss](https://huggingface.co/Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp)** — community GGUFs are broken, see [llama.cpp #16407](https://github.com/ggml-org/llama.cpp/issues/16407))

### Step 1: Clone and configure

```bash
git clone https://github.com/cried-nutty-won/rag-system.git
cd rag-system
cp config.sh.example config.sh
# Edit config.sh with your actual paths
```
Edit `config.sh` to match your environment:

```bash
LLAMA_CPP_BIN="$HOME/llama-cpp-turboquant/build-cpu/bin/llama-server"
GGUF_DIR="$HOME/models/GGUF/rag"
OBSIDIAN_DIR="$HOME/obsidian"
VENV_PYTHON="$HOME/.venv/main/bin/python3"
RAG_SCRIPTS_DIR="$(pwd)/server"
LLAMA_SCRIPTS_DIR="$(pwd)/llama"
LOG_DIR="/tmp"
```

### Step 2: Python dependencies

```bash
python3 -m venv ~/.venv/main
~/.venv/main/bin/pip install numpy requests rank_bm25
```

### Step 3: Download models

```bash
mkdir -p $GGUF_DIR

# Embedding (official Qwen GGUF)
huggingface-cli download Qwen/Qwen3-Embedding-0.6B-GGUF \
  Qwen3-Embedding-0.6B-Q8_0.gguf --local-dir $GGUF_DIR

# Reranker (Voodisss ONLY — do NOT use other sources)
huggingface-cli download Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp \
  Qwen3-Reranker-0.6B-Q4_K_M.gguf --local-dir $GGUF_DIR
```

### Step 4: Configure vaults

Edit `server/rag_server_rerank.py` and update `VAULTS_CONFIG` with your Obsidian or documentation vault paths:

```python
VAULTS_CONFIG = {
    "void":     {"path": os.path.join(OBSIDIAN_DIR, "001 Void 000")},
    "linux":    {"path": os.path.join(OBSIDIAN_DIR, "000 linux 000")},
    # Add your vaults here
}
```

### Step 5: First launch

```bash
# Start the full stack (embedding + reranker + RAG server)
bash llama/start-rag-llm_embed_reranker_server.sh

# Wait for indexing (~5-15 min on first run, instant on subsequent runs via cache)
# Verify health:
curl -s http://127.0.0.1:8182/health | jq .
```

Expected output (exemple) :

```json
{
  "status": "ok",
  "mode": "hybrid+reranker",
  "embedding_model": "qwen3-embed-06b",
  "reranker_model": "Qwen3-Reranker-0.6B",
  "vaults": ["void", "linux", "..."],
  "total_chunks": 3218,
  "port": 8182
}
```

### Step 6: Shell aliases

Add to your `~/.config/fish/config.fish` or `~/.bashrc`:

```
alias llmers='bash /path/to/rag-system/llama/start-rag-llm_embed_reranker_server.sh &'
alias rag='bash /path/to/rag-system/server/search_vault.sh --no-rerank'
alias ragr='bash /path/to/rag-system/server/search_vault.sh'
```

### Troubleshooting installation

| Problem | Solution |
|---------|----------|
| Reranker scores ~`1e-28` | Wrong GGUF source. Re-download from [Voodisss](https://huggingface.co/Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp) |
| `"This server does not support reranking"` | Missing flags. Ensure `--reranking --pooling rank --embedding` are all present |
| Port already in use | `pkill -f llama-server && pkill -f rag_server_rerank` then restart |
| OOM on startup | Add `--cache-ram 0` to both llama-server scripts (disables 8 GiB host prompt cache) |
| Slow first indexing | Normal. Subsequent starts use cached embeddings (instant) |

---
---

## 1. Architecture

### Search Pipeline

```
User Query
       │
       ├──→ [Embedding Qwen3-0.6B] ──→ 1024d Vector ──→ Cosine similarity ──→ Vector ranking
       │                                                                              │
       └──→ [FR Tokenization] ──→ BM25Okapi ──→ BM25 ranking                        │
                                                          │                           │
                                                          └───── RRF (k=60) ──────────┘
                                                                      │
                                                              Top 18 candidates
                                                                      │
                                                          [Reranker Qwen3-0.6B]
                                                           (cross-encoder)
                                                                      │
                                                              Final results
```

### Ports

| Port | Service | Model | Critical Flags |
|------|---------|-------|----------------|
| 8181 | Embedding (bi-encoder) | Qwen3-Embedding-0.6B-Q8_0 | `--embedding --pooling last` |
| 8184 | Reranker (cross-encoder) | Qwen3-Reranker-0.6B-Q4_K_M | `--reranking --pooling rank --embedding` |
| 8182 | RAG Server (Python) | — | — |

### Hardware (exemple)

- from 8 GB unified RAM
- Linux CPU only (`--n-gpu-layers 0`) => remove this flag for GPU use according to your hardware
- llama.cpp custom build: `$LLAMA_CPP_BIN`

---

## 2. File Paths

### Startup Scripts (llama.cpp)

```
$LLAMA_SCRIPTS_DIR/
├── start-llm-embed-qwen3-06b.sh      # Embedding 0.6B Q8_0 (port 8181) — ACTIVE
├── start-llm-embed-qwen3-4b.sh       # Embedding 4B Q4_K_M (port 8181) — alternative
├── start-llm-reranker-06b.sh         # Reranker 0.6B Q4_K_M (port 8184) — ACTIVE
├── start-rag-llm_embed_reranker_server.sh  # Stack: embed + reranker + RAG
└── start-rag-llm_embed_server.sh     # Stack: embed + RAG (no reranker)
```

### RAG Server and CLI

```
$RAG_SCRIPTS_DIR/
├── rag_server_rerank.py              # Main RAG server (port 8182)
├── search_vault.sh                   # CLI client (called by fish aliases)
└── test_tokens.sh                    # Max tokens measurement script per vault
```

### Fish Wrappers (exemple : add alias rag='path to rag.sh' in your config.fish file) to use the shortcut rag

```
$REPO_DIR/fish/
├── rag.sh                            # → search_vault.sh (aliases `rag` and `ragr`)
├── rc.sh                             # → health check all 3 ports
├── rs.sh                             # → launches rag_server_rerank.py in background
├── rsk.sh                            # → pkill -f rag_server_rerank.py
├── rst.sh                            # → tail -f RAG server logs
├── llmers.sh                         # → start-rag-llm_embed_reranker_server.sh &
├── llmes.sh                          # → start-rag-llm_embed_server.sh &
├── llmr.sh                           # → start-llm-reranker-06b.sh &
└── llme.sh                           # → start-llm-embed-qwen3-06b.sh &
```

### GGUF Models (exemple)

```
$GGUF_DIR/
├── Qwen3-Embedding-0.6B-Q8_0.gguf   # 610 MB — active embedding
├── Qwen3-Embedding-4B-Q4_K_M.gguf   # 2.4 GB — alternative embedding
├── Qwen3-Reranker-0.6B-Q4_K_M.gguf  # 379 MB — active reranker
├── Qwen3-Reranker-0.6B.Q8_0.gguf    # 610 MB — alternative reranker
├── Qwen3-Reranker-4B-Q4_K_M.gguf    # 2.4 GB — unused (too slow on CPU)
└── nomic-nofr/nomic-embed-text-v1.5.Q8_0.gguf  # legacy
```

### Obsidian Vaults (exemple)

```
$OBSIDIAN_DIR/
├── 000 linux 000/                    # vault "linux"     — 59 chunks, max 422 tokens
├── 001 Void 000/                     # vault "void"      — 1377 chunks, max 9708 tokens
├── 002 browsing 000/                 # vault "browsing"  — 238 chunks, max 940 tokens
├── 003 Terminal 000/                 # vault "terminal"  — 151 chunks, max 3141 tokens
├── 004 llm 000/                      # vault "llm"       — 1080 chunks, max 4606 tokens
├── 005 images 000/                   # vault "images"    — 78 chunks, max 522 tokens
└── 006 telephone/                    # vault "telephone" — 218 chunks, max 3463 tokens
```

### Embedding Cache (exemple)

```
~/.rag/
├── void_cache__qwen3-embed-06b.json
├── linux_cache__qwen3-embed-06b.json
├── browsing_cache__qwen3-embed-06b.json
├── terminal_cache__qwen3-embed-06b.json
├── llm_cache__qwen3-embed-06b.json
├── images_cache__qwen3-embed-06b.json
└── telephone_cache__qwen3-embed-06b.json
```

### llama.cpp Binary

```
$LLAMA_CPP_BIN
```

### Python Environment

```
$VENV_PYTHON
```

---

## 3. Validated Final Configuration

### Embedding Script (`start-llm-embed-qwen3-06b.sh`)

```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"

export LD_LIBRARY_PATH="$(dirname "$LLAMA_CPP_BIN"):$LD_LIBRARY_PATH"

if command -v ss &> /dev/null && ss -tln | grep -q :8181; then
    echo "⚠️  Port 8181 already in use. pkill -f Qwen3-Embedding"
    exit 1
fi

cd "$(dirname "$LLAMA_CPP_BIN")/.."
exec "$LLAMA_CPP_BIN" \
  -m "${GGUF_DIR}/Qwen3-Embedding-0.6B-Q8_0.gguf" \
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
  --cache-ram 0 \
  -ctk q8_0 \
  -ctv q8_0 \
  > "${LOG_DIR}/llm-embed-06b.log" 2>&1
```

### Reranker Script (`start-llm-reranker-06b.sh`)

```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"

export LD_LIBRARY_PATH="$(dirname "$LLAMA_CPP_BIN"):$LD_LIBRARY_PATH"

if command -v ss &> /dev/null && ss -tln | grep -q :8184; then
    echo "⚠️  Port 8184 already in use. pkill -f Qwen3-Reranker"
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
```

### RAG Server Parameters (`rag_server_rerank.py`)

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `RERANK_CANDIDATES` | **18** | Maximum stable before crash (30+ fails). ~580ms/candidate. |
| `RRF_K` | 60 | Standard constant (Cormack et al. 2009) |
| `DEFAULT_TOP_K` | 5 | Default number of results |
| `MIN_CONFIDENCE` | 50.0 | Minimum threshold in RRF-only mode |
| `MAX_CHARS` | 3000 | Text truncation before embedding |
| `EMBEDDING_MODEL_ID` | `"qwen3-embed-06b"` | Cache identifier |
| `alpha_ratio` | **0.36** | Threshold for `is_embeddable()` filter |
| `LLAMA_EMBED_URL` | `http://127.0.0.1:8181/embedding` | Embedding endpoint |
| `LLAMA_RERANK_URL` | `http://127.0.0.1:8184/v1/rerank` | Reranker endpoint (**/v1/rerank**, NOT /reranking) |

---

## 4. Detailed Operation

### 4.1 Indexing (at server startup)

1. Recursive traversal of each vault (`os.walk`)
2. Reading `.md` files
3. Chunking by Markdown headers (`#`, `##`, `###`) via `chunk_by_markdown()`
4. Anti-noise filtering via `is_embeddable()`:
   - Rejects chunks starting with ` ``` `, `<`, `|`
   - Rejects if >25% of lines are commands/logs
   - Rejects if alphabetic ratio <36%
   - Rejects if printable character ratio <95%
5. For each chunk:
   - If text is in cache → vector loaded from JSON
   - Otherwise → call to embedding API (port 8181) → vector computed and cached
6. Building BM25 index (`BM25Okapi`) on French tokens
7. In-memory storage: `{id, source, path, text, vector, tokens}`

### 4.2 Search (per query)

**Step 1 — Vector (bi-encoder):**
- Query is embedded via `POST /embedding` (port 8181)
- Cosine similarity against all vectors in the vault
- Ranking by decreasing score

**Step 2 — BM25 (sparse):**
- French tokenization of the query (regex with accents)
- BM25 score against vault index
- Ranking by decreasing score

**Step 3 — Reciprocal Rank Fusion:**
- Formula: `score(doc) = Σ 1/(k + rank)` with k=60
- Merges both rankings into one
- Raw scores (incomparable) are ignored; only rank matters

**Step 4 — Reranker (cross-encoder):**
- Top 18 RRF candidates are sent to the reranker
- Each document is prefixed with its filename: `[filename.md]\n{text[:1400]}`
- Reranker evaluates each (query, doc) pair jointly via chat template:
  ```
  <|im_start|>system
  Judge whether the Document meets the requirements based on the Query
  and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
  <|im_start|>user
  <Instruct>: Given a web search query, retrieve relevant passages that answer the query
  <Query>: {query}
  <Document>: [{filename}]\n{text}<|im_end|>
  <|im_start|>assistant
  <think>
  </think>
  ```
- Classifier `cls.output.weight` projects final hidden state to P(yes)/P(no)
- Relevance score: `relevance_score = P(yes)` (0.0 → 1.0)
- If reranker is unreachable → fallback to pure RRF (transparent)

### 4.3 Cache

- Key: the **exact text** of the chunk
- Value: the embedding vector (list of floats)
- Naming: `{vault}_cache__{model_id}.json`
- The model_id (`qwen3-embed-06b`) allows model changes without collision
- Modified file → new chunks → cache miss → automatic re-embedding
- Unchanged file → cache hit → instant

### 4.4 API Endpoints

**`GET /health`** (port 8182):
```json
{
  "status": "ok",
  "mode": "hybrid+reranker",
  "embedding_model": "qwen3-embed-06b",
  "reranker_model": "Qwen3-Reranker-0.6B",
  "vaults": ["void", "linux", ...],
  "total_chunks": 3218,
  "port": 8182
}
```

**`POST /search`** (port 8182):
```json
// Request
{"vault": "void", "query": "my question", "top_k": 5, "rerank": true}

// Response
{
  "query": "my question",
  "vault": "void",
  "count": 5,
  "reranked": true,
  "elapsed_ms": 12200,
  "results": [
    {
      "source": "file.md",
      "path": "$OBSIDIAN_DIR/.../file.md",
      "confidence": 99.9,
      "rerank_score": 0.9999,
      "semantic_score": 0.847,
      "bm25_score": 12.34,
      "rrf_score": 0.03279,
      "text": "chunk content..."
    }
  ]
}
```

**`POST /v1/rerank`** (port 8184):
```json
// Request
{"query": "...", "documents": ["doc1", "doc2"], "top_n": 3}

// Response
{"results": [{"index": 0, "relevance_score": 0.98}, ...]}
```

**`POST /embedding`** (port 8181):
```json
// Request
{"content": "text to embed"}

// Response
{"embedding": [0.012, -0.034, ...]}  // 1024 dimensions
```

---

## 5. Technical Choices and Justification

### 5.1 Qwen3-Embedding-0.6B vs Nomic v1.5 vs Qwen3-4B

| Criteria | Nomic v1.5 | Qwen3-0.6B Q8_0 | Qwen3-4B Q4_K_M |
|---------|-----------|-----------------|-----------------|
| French | Weak (English-centric) | Native (100+ languages) | Native |
| Dimensions | 768 | 1024 | 2560 |
| Reranker family | None | Qwen3-Reranker ✅ | Qwen3-Reranker ✅ |
| RAM | 150 MB | 650 MB | 2.8 GB |
| Query speed | ~15 ms | ~60 ms | ~250 ms |
| MTEB multilingual | — | 64.33 | 69.45 |

**Choice: Qwen3-0.6B Q8_0.** French language gain and consistency with reranker take priority. The 4B was too slow for interactive use alongside the LLM agent. The reranker compensates for the quality gap between 0.6B and 4B.

### 5.2 Reranker 0.6B Q4_K_M vs Q8_0 vs 4B

Voodisss benchmark (MTEB AskUbuntuDupQuestions, 0.6B):

| Quant | Size | Δ NDCG@10 |
|-------|------|-----------|
| F16 | 1.12 GB | baseline |
| Q8_0 | 610 MB | -0.2% |
| **Q4_K_M** | **379 MB** | **-0.3%** |
| Q4_0 | 360 MB | -2.0% |
| Q2_K | 280 MB | -28.7% |

**Choice: Q4_K_M.** Official sweet spot: 3× smaller than F16, 0.3% loss. The 4B is unusable on CPU with the LLM agent (too slow, ~30-40s for 18 candidates).

### 5.3 Pooling

| Model | Pooling | Justification |
|-------|---------|---------------|
| Embedding | `--pooling last` | Qwen3 blog: "hidden state vector corresponding to the final [EOS] token" |
| Reranker | `--pooling rank` | Activates yes/no classifier (`cls.output.weight`). Mandatory. |

**Note:** Voodisss guide indicates `pooling = mean` for embedding. Official Qwen3 documentation (blog + README) explicitly says `last` ([EOS] token). Official docs take precedence.

### 5.4 Embedding ctx-size = 8192 — Reranker ctx-size = 1024

Max tokens measurement per vault (exact Qwen3 tokenizer):

| Vault | Max tokens | Covered by 8192? |
|-------|-----------|-----------------|
| void | 9,708 | ❌ (1 outlier file: video transcription) |
| llm | 4,606 | ✅ |
| telephone | 3,463 | ✅ |
| terminal | 3,141 | ✅ |
| browsing | 940 | ✅ |
| images | 522 | ✅ |
| linux | 422 | ✅ |

**Choice: ctx=8192 and ub=8192.** Covers 99% of chunks without truncation. The outlier file (9708 tokens) is cleanly truncated (beginning preserved). ctx/ub-size does not affect quality as long as input fits — empty slots are never used.

**Reranker ctx-size = 1024 and ub=1024.** Covers 100% of cases. The reranker only processes **one pair** (query + document) at a time, i.e. ~515 tokens max.

### 5.5 KV Cache Quantization: q8_0

```bash
-ctk q8_0    # key quantization (K)
-ctv q8_0    # value quantization (V)
```

| KV Quant | KV Cache RAM | Quality Loss |
|----------|-------------|--------------|
| f16 (default) | 56 KB/token | baseline |
| **q8_0** | **28 KB/token** | **<0.1%** |
| q4_0 | 14 KB/token | Notable (avoid) |

**Choice: q8_0.** Half the RAM, nearly identical quality. Free performance gain.

### 5.6 RERANK_CANDIDATES = 18

Empirical tests (M1 Pro CPU, ~580ms/candidate):

| Candidates | Latency | Reranker | Quality |
|-----------|---------|----------|---------|
| 5 | 2,655 ms | ✅ | Good (but limited discrimination) |
| 10 | 6,235 ms | ✅ | Acceptable |
| **18** | **~10,400 ms** | **✅** | **Optimal (stable max)** |
| 21 | ~12,200 ms | ✅ | Marginal gain over 18 |
| 30 | 20,501 ms | ✅ (with --timeout 120) | Same as 21 |
| 50 | 20,497 ms | ❌ crash → RRF fallback | — |

**Choice: 18.** Good balance between latency and quality. Literature recommends 20-50 candidates (for GPU rerankers at 4ms/doc). With 18 candidates at 580ms/doc on CPU, we cover the equivalent of 50 GPU candidates.

### 5.7 Filename Inclusion in Reranker

```python
rerank_docs = [f"[{c['source']}]\n{c['text'][:1400]}" for c in candidate_chunks]
```

Without the filename, the reranker cannot match a query that is literally the filename. With the filename, the cross-encoder sees the exact match and scores 0.9999 instead of 0.9895. Tested and validated.

### 5.8 Voodisss GGUF (Mandatory for Reranker)

Community Qwen3-Reranker GGUFs are **broken** (llama.cpp #16407). They are missing:
- The `cls.output.weight` tensor (yes/no classifier)
- The `pooling_type=RANK` metadata
- The reranking chat template

Result: garbage scores (`4.5e-23`). Only **Voodisss** GGUFs (converted with the official `convert_hf_to_gguf.py`) work.

The Qwen3 reranker is a **generative reranker**: the model produces logits, `cls.output.weight` (tensor `[hidden_dim, 2]`) projects the final hidden state to P(yes) and P(no), then softmax → `relevance_score = P(yes)`.

### 5.9 Host Prompt Cache Disabled (`--cache-ram 0`)

llama.cpp PR #16391 introduced host-memory prompt caching with a **default of 8 GiB**. For embedding/reranking servers where prompts are **never reused**, this is pure waste. dvcdsys/code-index documented the problem in production: RSS went from 365 MB to 11.3 GB before OOM kill. With `--cache-ram 0`, it plateaus at ~900 MB under the same load.

### 5.10 Instruction-Aware (1-5% gain)

Both models support custom instructions. Qwen3 recommends:
- Write instructions in **English** (even for multilingual use)
- Default instruction: `"Given a web search query, retrieve relevant passages that answer the query"`
- Measured gain: 1% to 5% depending on tasks

Currently, the default instruction is used (automatically injected by the reranker's chat template). Customization possible later.

---

## 6. Methodology (Inspired by Dave Ebbelaar)

The pipeline follows the architecture presented in "Hybrid Retrieval from Scratch" (2026):

1. **BM25**: captures exact terms, identifiers, rare words. Misses paraphrases.
2. **Dense embeddings**: captures semantic meaning. Misses exact terms.
3. **RRF**: merges both rankings by rank (not by score). BM25 and cosine scores are incomparable; rank is not.
4. **Reranker**: reorders candidates by jointly reading query + document. This is the step that provides the most qualitative gain (NDCG +12 points in the video's FinanceQA benchmark).

**What is NOT implemented (compared to the video):**
- NDCG@10 evaluation with ground truth
- LLM-generated evaluation dataset
- Systematic configuration comparison

---

## 7. Setup from Scratch

### Prerequisites

- llama.cpp compiled (CPU build): `$LLAMA_CPP_BIN`
- Python 3 with venv: `~/.venv/main/`
- Python packages: `numpy`, `requests`, `rank_bm25`
- GGUF models downloaded (Voodisss for reranker, official Qwen for embedding)

### Installation

```bash
# 1. Create venv and install dependencies
python3 -m venv ~/.venv/main
~/.venv/main/bin/pip install numpy requests rank_bm25

# 2. Copy config.sh.example to config.sh and adapt paths
cp config.sh.example config.sh
# Edit config.sh with your paths

# 3. Place scripts (see section 2 for paths)
chmod +x llama/*.sh
chmod +x server/*.py
chmod +x fish/*.sh

# 4. Configure fish aliases (see config.fish)

# 5. First launch (full indexing)
llmers
# Wait for indexing to complete (~5-15 min depending on vault size)
# Verify:
curl -s http://127.0.0.1:8182/health | jq .
```

### Adding a New Vault

1. Add entry in `VAULTS_CONFIG` in `rag_server_rerank.py`:
```python
"docs": {
    "path": os.path.join(OBSIDIAN_DIR, "docs_techniques"),
},
```

2. Add vault name to the regex in `search_vault.sh`:
```bash
if [[ "${1:-}" =~ ^(void|linux|browsing|terminal|llm|images|telephone|docs|obsidian|all)$ ]]; then
```

3. Restart the server:
```bash
rsk
rs
```

---

## 8. Quick Usage Guide

See `doc/00 Quick Start Guide.md` for the condensed version.

---

## 9. Debugging

### RAG Server Not Responding (port 8182)

```bash
ps aux | grep rag_server_rerank
tail -50 $LOG_DIR/rag_server_rerank.log

# Common error: IndentationError after editing
$VENV_PYTHON -c "import py_compile; py_compile.compile('$RAG_SCRIPTS_DIR/rag_server_rerank.py', doraise=True)"
```

### Embedding Not Responding (port 8181)

```bash
ps aux | grep llama-server | grep 8181
tail -20 $LOG_DIR/llm-embed-06b.log

# Common error: script points to wrong model
cat $LLAMA_SCRIPTS_DIR/start-llm-embed-qwen3-06b.sh | grep "^\s*-m"
```

### Reranker Not Responding (port 8184)

```bash
ps aux | grep llama-server | grep 8184

# Direct test
curl -s http://127.0.0.1:8184/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"query":"test","documents":["test document"]}' | jq .

# If scores ~1e-28 → broken GGUF (bad conversion)
# If "This server does not support reranking" → missing flags
# Verify the 3 mandatory flags:
cat $LLAMA_SCRIPTS_DIR/start-llm-reranker-06b.sh | grep -E "reranking|pooling|embedding"
```

### "Connection refused" on 8181 from RAG

Embedding is not running. The RAG server needs 8181 to be active **before** receiving requests.

```bash
llme    # start embedding
# wait 3-5s
rag void "test"
```

### Suspicious Reranker Scores (1e-28)

GGUF is badly converted. Missing `cls.output.weight`. Solution: re-download from Voodisss.

### Abnormal Latency (>20s)

```bash
# Check which model is loaded
ps aux | grep llama-server | grep -v grep

# If 4B is loaded instead of 0.6B → kill and restart with correct script
pkill -f llama-server
llme    # 0.6B
llmr    # reranker 0.6B
```

### Measure Max Tokens per Vault

```bash
$RAG_SCRIPTS_DIR/test_tokens.sh
```

Requires `transformers` installed: `$VENV_PYTHON -m pip install transformers`

---

## 10. Cleanup

### Delete Cache (Full Re-indexing)

```bash
rm -f ~/.rag/*__qwen3-embed-06b*
```

### Delete Old Cache (Model Change)

```bash
rm -f ~/.rag/*__nomic*
rm -f ~/.rag/*__qwen3-embed-4b*
```

### Kill All RAG Processes

```bash
pkill -f llama-server
pkill -f rag_server_rerank
```

### Check Cache Size

```bash
du -sh ~/.rag/
```

---

## 11. Estimated RAM with ai agent

| Component | Configuration | RAM |
|-----------|--------------|-----|
| Embedding 0.6B Q8_0 | ctx 8192, ub 8192, KV q8_0, parallel 1 | ~794 MB |
| Reranker 0.6B Q4_K_M | ctx 1024, ub 1024, KV q8_0, parallel 1 | ~687 MB |
| Python Server (numpy + BM25) | 3218 chunks | ~200 MB |
| **Total RAG Stack** | | **~1.68 GB** |
| LLM Qwen3.6-35B-A3B Q4_K_XL | | ~20.4 GB |
| System | | ~5 GB |
| **Total System** | | **~27 GB / 32 GB** |

---

## 12. Known Limitations

- **No hot-reload**: modifying an Obsidian file requires server restart (`rs`)
- **Reranker latency**: ~10s on CPU for 18 candidates (~580ms/candidate). Structural (cross-encoder = 1 forward pass per pair)
- **RERANK_CANDIDATES > 21**: reranker crashes (timeout or memory saturation). 21 is the stable maximum.
- **1 outlier file**: "Is Void Linux Good - With Jake from @JakeLinux.md" (9708 tokens) is truncated at 8192 tokens. Negligible impact (1 chunk out of 3218).
- **No NDCG evaluation**: no ground truth to objectively measure quality
- **is_embeddable() filter**: may reject legitimate technical chunks (25% code line threshold)
- **Orphan cache**: deleted/modified chunks remain in JSON (no garbage collection)
- **Single-threaded HTTP**: Python server (`http.server`) handles only one request at a time
- **ctx-size does not depend on chunk count**: it is determined by the max length of an individual chunk, not by corpus size

---

## 13. Possible Evolutions

- [ ] Switch to `http.server.ThreadingHTTPServer` for concurrent requests
- [ ] Add `POST /reindex` endpoint to re-scan a vault without restarting
- [ ] Build evaluation dataset (query→chunk pairs) to measure NDCG@10
- [ ] Add custom instructions to reranker (1-5% gain according to Qwen)
      Default instruction: "Given a web search query, retrieve relevant passages that answer the query"
      For a FR technical vault: "Retrieve relevant technical documentation passages that answer the query about Linux system administration"
      Note: Qwen3 recommends writing instructions in English even for multilingual use
- [ ] Cache garbage collection (remove orphan entries)
- [ ] Support 4B model in "batch offline" mode for indexing large vaults
- [ ] Investigate why RERANK_CANDIDATES > 21 crashes the reranker (memory? internal timeout?)
- [ ] Test `--parallel 1` vs `--parallel 2` for reranker
- [ ] Explore llama.cpp model routing (`--models-preset models.ini`) to serve embedding + reranker on a single port
- [ ] Improve chunking for files without Markdown headers (transcriptions, long articles)
- [ ] Replace numpy cosine similarity with FAISS IndexHNSW beyond 50,000 chunks

---

## 14. References

- **Hybrid pipeline**: Dave Ebbelaar, "Hybrid Retrieval from Scratch" (2026)
  https://www.youtube.com/watch?v=XvKiTfd6Xvo
- **RRF**: Cormack, Clarke, Buettcher (2009) — "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
- **Qwen3-Embedding**: https://qwenlm.github.io/blog/qwen3-embedding/
  Paper: arXiv:2506.05176
- **Qwen3-Reranker HuggingFace**: https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
- **Qwen3-Reranker GGUF (Voodisss)**:
  https://huggingface.co/Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp
- **llama.cpp multi-model guide (Voodisss)**:
  https://gist.github.com/VooDisss/42bce4eb5c76d3c325633886c5e348ee
- **llama.cpp issue #16407** (broken reranker GGUFs):
  https://github.com/ggml-org/llama.cpp/issues/16407
- **llama.cpp PR #16391** (host-memory prompt caching):
  https://github.com/ggml-org/llama.cpp/pull/16391
- **llama.cpp**: https://github.com/ggml-org/llama.cpp

---

## 15. FAQ

**Q: Why not a vector database (Qdrant, ChromaDB, etc.)?**
A: With ~3000 chunks, an in-memory numpy array suffices. Cosine search on 3000 vectors of 1024d takes <1ms. A vector DB adds complexity without measurable gain at this scale. Beyond ~100,000 chunks, reconsider (FAISS IndexHNSW or Qdrant).

**Q: Why BM25 in addition to vector search?**
A: Vector search misses exact terms (command names, paths, identifiers). BM25 captures them. On technical vaults with many commands (`nftables`, `dracut`, `sfdisk`), BM25 is often more reliable than vector search alone.

**Q: Why is the reranker so slow on CPU?**
A: The cross-encoder does a FULL forward pass of the model for EACH (query, document) pair. Unlike the bi-encoder which encodes once and compares vectors, the cross-encoder re-processes everything from scratch. On CPU without GPU, a 0.6B forward pass takes ~580ms. This is structural.

**Q: Can I use RAG without reranker?**
A: Yes. `rag` (alias) forces `--no-rerank`. The server also works if the reranker is not running (automatic RRF fallback). Latency: ~20ms.

**Q: Can I use RAG without embedding?**
A: No. Embedding is required for vector search. Without the embedding server (port 8181), the RAG server returns a 500 error.

**Q: How to add a vault without re-indexing everything?**
A: Add the entry in VAULTS_CONFIG, restart the server. Existing vaults are loaded from cache (instant). Only the new vault is indexed from scratch.

**Q: Is the cache compatible between models?**
A: No. The filename includes the model identifier (`void_cache__qwen3-embed-06b.json`). Changing models creates a new cache file.

**Q: Why are all reranker scores >0.99?**
A: The Qwen3 reranker uses a yes/no classifier with softmax. On clearly relevant documents, P(yes) → 1.0. Discrimination happens on marginal documents (scores 0.3-0.8).

**Q: Does ctx-size depend on the number of chunks in the vault?**
A: No. ctx-size defines the max length of a single input. Whether the vault contains 3,000 or 300,000 chunks, the embedding always receives one chunk at a time (~910 tokens max) and the reranker always a query+document pair (~500 tokens max). ctx-size is determined by the length of the largest individual chunk, not by corpus size.

**Q: Why `--pooling last` and not `--pooling mean` for embedding?**
A: Official Qwen3 blog says: "The Embedding model processes a single text segment as input, extracting the semantic representation by utilizing the hidden state vector corresponding to the final [EOS] token." That's `last`. Voodisss guide indicates `mean`, but official Qwen3 documentation takes precedence.

---

## 16. Glossary

| Term | Definition |
|------|-----------|
| **Bi-encoder** | Encodes query and document separately → vectors → cosine similarity. Fast but loses nuances. |
| **Cross-encoder** | Encodes query + document jointly → relevance score. Slow but precise. |
| **Generative reranker** | The Qwen3 reranker uses a classifier (cls.output.weight) that projects the hidden state to P(yes)/P(no). Not a traditional cross-encoder. |
| **RRF** | Reciprocal Rank Fusion. Merges rankings by rank, not by score. Formula: 1/(k+rank). |
| **BM25** | Keyword search algorithm with TF-IDF weighting. Captures exact terms. |
| **Pooling last** | The embedding vector is the hidden state of the last [EOS] token. |
| **Pooling rank** | Classifier mode for the reranker. Extracts yes/no logits via cls.output.weight. |
| **KV cache** | Attention memory (keys + values) allocated per context token. Quantizable via -ctk/-ctv. |
| **Chunk** | Document fragment from Markdown splitting. Indexing unit. |
| **Vault** | Obsidian directory indexed as a distinct collection. |
| **GGUF** | File format for quantized models used by llama.cpp. |
| **Quantization** | Reducing weight precision (F16→Q8→Q4) to reduce RAM. |
| **MTEB** | Massive Text Embedding Benchmark. Reference ranking for embedding models. |
| **NDCG@K** | Normalized Discounted Cumulative Gain. Retrieval quality metric (0-1). |
| **MRL** | Matryoshka Representation Learning. Allows flexible embedding dimensions (32 to 2560 for Qwen3-4B). |
| **Instruction-aware** | Model's ability to adapt behavior based on a custom instruction (1-5% gain). |

---

Recommanded system (not mandatory) : Void Linux
