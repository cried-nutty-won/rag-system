#!/usr/bin/env python3
"""
Serveur RAG API Multi-Vaults + Reranker (Qwen3-Reranker-0.6B)
Pipeline : Vectoriel (Qwen3-Embedding-0.6B) + BM25 → RRF → Reranker cross-encoder

Écoute sur http://127.0.0.1:8182
Endpoints :
  - GET  /health       : Statut serveur + reranker
  - POST /search       : Recherche hybride + reranking
"""

import json
import os
import re
import sys
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import numpy as np
import requests
from rank_bm25 import BM25Okapi

# ═══════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════
LLAMA_EMBED_URL = "http://127.0.0.1:8181/embedding"
LLAMA_RERANK_URL = "http://127.0.0.1:8184/v1/rerank"
HOST = "127.0.0.1"
PORT = 8182
MAX_CHARS = 3000
RRF_K = 60
DEFAULT_TOP_K = 5
MIN_CONFIDENCE = 50.0
RERANK_CANDIDATES = 18

CACHE_DIR = os.path.expanduser("~/.rag")
os.makedirs(CACHE_DIR, exist_ok=True)

EMBEDDING_MODEL_ID = "qwen3-embed-06b"


def cache_path_for(base_cache_file, model_id):
    root, ext = os.path.splitext(base_cache_file)
    return f"{root}__{model_id}{ext}"


VAULTS_CONFIG = {
    "void": {
        "path": "/home/ksoinan/obsidian/001 Void 000",
        "cache_file": os.path.join(CACHE_DIR, "void_cache.json"),
    },
    "linux": {
        "path": "/home/ksoinan/obsidian/000 linux 000",
        "cache_file": os.path.join(CACHE_DIR, "linux_cache.json"),
    },
    "browsing": {
        "path": "/home/ksoinan/obsidian/002 browsing 000",
        "cache_file": os.path.join(CACHE_DIR, "browsing_cache.json"),
    },
    "terminal": {
        "path": "/home/ksoinan/obsidian/003 Terminal 000",
        "cache_file": os.path.join(CACHE_DIR, "terminal_cache.json"),
    },
    "llm": {
        "path": "/home/ksoinan/obsidian/004 llm 000",
        "cache_file": os.path.join(CACHE_DIR, "llm_cache.json"),
    },
    "images": {
        "path": "/home/ksoinan/obsidian/005 images 000",
        "cache_file": os.path.join(CACHE_DIR, "images_cache.json"),
    },
    "telephone": {
        "path": "/home/ksoinan/obsidian/006 telephone",
        "cache_file": os.path.join(CACHE_DIR, "telephone_cache.json"),
    },
}

VAULT_DATA = {}
MAX_RRF_SCORE = 2.0 / (RRF_K + 1)


# ═══════════════════════════════════════════════════════════
#  UTILITAIRES
# ═══════════════════════════════════════════════════════════
def is_embeddable(text):
    stripped = text.strip()
    if (
        not stripped
        or stripped.startswith("```")
        or stripped.startswith("<")
        or stripped.startswith("|")
    ):
        return False
    lines = stripped.split("\n")
    code_log_lines = sum(
        1
        for line in lines
        if line.strip().startswith(
            (
                "$", ">", "ksoinan@", "slot ", "srv ", "res ", "que ",
                "import ", "def ", "class ", "//", "/*", "{", "[",
            )
        )
    )
    if len(lines) > 0 and (code_log_lines / len(lines)) > 0.25:
        return False
    alpha_ratio = sum(c.isalpha() for c in stripped) / max(len(stripped), 1)
    if alpha_ratio < 0.36:
        return False
    printable_ratio = sum(c.isprintable() or c in "\n\t" for c in stripped) / max(
        len(stripped), 1
    )
    return printable_ratio >= 0.95


def clean_text(text):
    cleaned = re.sub(r"[^\x20-\x7E\u00A0-\uFFFF\n\t]", " ", text)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def chunk_by_markdown(content, min_len=50):
    chunks = []
    sections = re.split(r"^(#{1,3}\s+.+)$", content, flags=re.MULTILINE)
    current_section = ""
    for part in sections:
        if not part.strip():
            continue
        if re.match(r"^#{1,3}\s+.+", part.strip()):
            if len(current_section.strip()) >= min_len and is_embeddable(
                current_section.strip()
            ):
                chunks.append(clean_text(current_section.strip()))
            current_section = part + "\n"
        else:
            current_section += part + "\n"
    if len(current_section.strip()) >= min_len and is_embeddable(
        current_section.strip()
    ):
        chunks.append(clean_text(current_section.strip()))
    if not chunks:
        chunks = [
            clean_text(p)
            for p in content.split("\n\n")
            if len(p.strip()) > min_len and is_embeddable(p.strip())
        ]
    return chunks


def tokenize_french(text):
    return [
        t.lower()
        for t in re.findall(
            r"\b[a-zàâäéèêëïîôùûüÿçœæ0-9]{2,}\b", text.lower(), re.IGNORECASE
        )
    ]


# ═══════════════════════════════════════════════════════════
#  EMBEDDING (Qwen3-Embedding-0.6B via llama.cpp)
# ═══════════════════════════════════════════════════════════
def get_embedding(text):
    safe_text = clean_text(text)[:MAX_CHARS]
    if not safe_text:
        raise ValueError("Texte vide")
    response = requests.post(
        LLAMA_EMBED_URL, json={"content": safe_text}, timeout=30
    )
    response.raise_for_status()
    res = response.json()

    def extract_vector(obj):
        if isinstance(obj, list) and len(obj) > 0:
            if isinstance(obj[0], (int, float)):
                return obj
            if isinstance(obj[0], (list, dict)):
                return extract_vector(obj[0])
        elif isinstance(obj, dict):
            if "embedding" in obj:
                return extract_vector(obj["embedding"])
            if "data" in obj and isinstance(obj["data"], list) and len(obj["data"]) > 0:
                return extract_vector(obj["data"][0])
        return None

    vec = extract_vector(res)
    if (
        vec is None
        or not isinstance(vec, list)
        or not all(isinstance(x, (int, float)) for x in vec)
    ):
        raise ValueError(f"Structure embedding invalide : {str(res)[:200]}")
    return vec


# ═══════════════════════════════════════════════════════════
#  RERANKER (Qwen3-Reranker-0.6B via /v1/rerank)
# ═══════════════════════════════════════════════════════════
def rerank(query, documents, top_n=None):
    if not documents:
        return []

    payload = {"query": query, "documents": documents}
    if top_n:
        payload["top_n"] = top_n

    try:
        response = requests.post(LLAMA_RERANK_URL, json=payload, timeout=60)
        response.raise_for_status()
        res = response.json()

        if "results" in res:
            ranked = [
                (item["index"], item["relevance_score"])
                for item in res["results"]
            ]
            ranked.sort(key=lambda x: x[1], reverse=True)
            return ranked
        elif isinstance(res, list):
            ranked = [(item["index"], item["relevance_score"]) for item in res]
            ranked.sort(key=lambda x: x[1], reverse=True)
            return ranked
        else:
            print(f"[RERANK] Format inattendu : {list(res.keys()) if isinstance(res, dict) else type(res)}")
            return []
    except requests.exceptions.ConnectionError:
        print("[RERANK] ⚠️  Reranker injoignable (port 8184) — fallback RRF pur")
        return []
    except requests.exceptions.Timeout:
        print("[RERANK] ⚠️  Timeout reranker — fallback RRF pur")
        return []
    except Exception as e:
        print(f"[RERANK] Erreur : {e}")
        return []


# ═══════════════════════════════════════════════════════════
#  SIMILARITÉ & FUSION
# ═══════════════════════════════════════════════════════════
def cosine_similarity(a, b):
    a, b = (
        np.asarray(a, dtype=np.float32).flatten(),
        np.asarray(b, dtype=np.float32).flatten(),
    )
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else 0.0


def reciprocal_rank_fusion(rankings, k=60):
    rrf_scores = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, 1):
            rrf_scores[chunk_id] += 1.0 / (k + rank)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


# ═══════════════════════════════════════════════════════════
#  CACHE
# ═══════════════════════════════════════════════════════════
def load_cache(cache_file):
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def save_cache(cache_data, cache_file):
    serializable = {
        k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in cache_data.items()
    }
    temp_file = cache_file + ".tmp"
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False)
        os.replace(temp_file, cache_file)
    except Exception as e:
        print(f"Erreur sauvegarde cache {cache_file} : {e}")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ═══════════════════════════════════════════════════════════
#  INITIALISATION
# ═══════════════════════════════════════════════════════════
def initialize_rag():
    global VAULT_DATA
    print("=== RAG + Reranker | Embed: Qwen3-0.6B-Q8 | Rerank: Qwen3-0.6B-Q4 ===")

    for vault_name, config in VAULTS_CONFIG.items():
        print(f"  Vault '{vault_name}' ({config['path']})...")
        raw_chunks = []
        for root, _, files in os.walk(config["path"]):
            for file in files:
                if file.endswith(".md"):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                            for chunk_text in chunk_by_markdown(f.read()):
                                raw_chunks.append({
                                    "source": file,
                                    "path": filepath,
                                    "text": chunk_text,
                                })
                    except Exception:
                        continue

        resolved_cache_file = cache_path_for(config["cache_file"], EMBEDDING_MODEL_ID)
        cache = load_cache(resolved_cache_file)
        chunks = []
        skipped = 0
        new_embeddings = 0

        for i, chunk in enumerate(raw_chunks):
            text, source = chunk["text"], chunk["source"]
            chunk_id = f"{vault_name}::{source}::{i}"

            if (
                text in cache
                and isinstance(cache[text], list)
                and len(cache[text]) > 0
                and isinstance(cache[text][0], (int, float))
            ):
                vector = np.array(cache[text], dtype=np.float32)
            else:
                try:
                    vector = np.array(get_embedding(text), dtype=np.float32)
                    cache[text] = vector
                    new_embeddings += 1
                except Exception:
                    skipped += 1
                    continue

            chunks.append({
                "id": chunk_id,
                "source": source,
                "path": chunk["path"],
                "text": text,
                "vector": vector,
                "tokens": tokenize_french(text + " " + source),
            })

        if new_embeddings > 0:
            print(f"    → {len(chunks)} chunks ({new_embeddings} nouveaux embeddings, {skipped} ignorés)")
        else:
            print(f"    → {len(chunks)} chunks (cache)")

        save_cache(cache, resolved_cache_file)
        all_tokens = [c["tokens"] for c in chunks]
        VAULT_DATA[vault_name] = {"chunks": chunks, "bm25": BM25Okapi(all_tokens)}

    # Index global "all"
    all_chunks = []
    for v in VAULT_DATA.values():
        all_chunks.extend(v["chunks"])
    all_tokens = [c["tokens"] for c in all_chunks]
    VAULT_DATA["all"] = {"chunks": all_chunks, "bm25": BM25Okapi(all_tokens)}

    # Index "obsidian" (tous les vaults Obsidian, sans les futurs vaults externes)
    OBSIDIAN_VAULTS = ["void", "linux", "browsing", "terminal", "llm", "images", "telephone"]
    obsidian_chunks = []
    for name in OBSIDIAN_VAULTS:
        if name in VAULT_DATA:
            obsidian_chunks.extend(VAULT_DATA[name]["chunks"])
    obsidian_tokens = [c["tokens"] for c in obsidian_chunks]
    VAULT_DATA["obsidian"] = {"chunks": obsidian_chunks, "bm25": BM25Okapi(obsidian_tokens)}
    
    # Vérification reranker
    try:
        r = requests.post(
            LLAMA_RERANK_URL,
            json={"query": "test", "documents": ["test document"]},
            timeout=5,
        )
        if r.status_code == 200:
            res = r.json()
            if "results" in res and res["results"]:
                score = res["results"][0].get("relevance_score", -1)
                if score > 0.01:
                    print(f"  ✓ Reranker opérationnel (score test: {score:.4f})")
                else:
                    print(f"  ⚠️  Reranker répond mais scores suspects ({score}) — GGUF cassé ?")
            else:
                print("  ⚠️  Reranker répond mais format inattendu")
        else:
            print(f"  ⚠️  Reranker HTTP {r.status_code} — mode dégradé")
    except Exception:
        print("  ⚠️  Reranker non détecté (port 8184) — mode dégradé sans rerank")

    total = sum(len(v["chunks"]) for k, v in VAULT_DATA.items() if k != "all")
    print(f"✓ Serveur prêt : http://{HOST}:{PORT} | {len(VAULT_DATA)-1} vaults | {total} chunks")


# ═══════════════════════════════════════════════════════════
#  HANDLER HTTP
# ═══════════════════════════════════════════════════════════
class RAGRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(
            json.dumps(data, ensure_ascii=False, cls=NumpyEncoder).encode("utf-8")
        )

    def do_GET(self):
        if urlparse(self.path).path == "/health":
            real_vaults = [k for k in VAULT_DATA.keys() if k != "all"]
            total_chunks = sum(
                len(v["chunks"]) for k, v in VAULT_DATA.items() if k != "all"
            )
            reranker_ok = False
            try:
                r = requests.post(
                    LLAMA_RERANK_URL,
                    json={"query": "test", "documents": ["doc"]},
                    timeout=3,
                )
                reranker_ok = r.status_code == 200
            except Exception:
                pass

            self._send_json({
                "status": "ok",
                "mode": "hybrid+reranker",
                "embedding_model": EMBEDDING_MODEL_ID,
                "reranker_model": "Qwen3-Reranker-0.6B" if reranker_ok else "OFFLINE",
                "vaults": real_vaults,
                "total_chunks": total_chunks,
                "port": PORT,
            })
        else:
            self._send_json({"error": "GET /health ou POST /search"}, 404)

    def do_POST(self):
        if urlparse(self.path).path != "/search":
            self._send_json({"error": "POST /search uniquement"}, 404)
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self._send_json({"error": "Body JSON vide"}, 400)
                return

            data = json.loads(self.rfile.read(content_length).decode("utf-8"))
            query = data.get("query", "").strip()
            vault_name = data.get("vault", "void")
            top_k = max(1, min(int(data.get("top_k", DEFAULT_TOP_K)), 20))
            use_rerank = data.get("rerank", True)

            if not query:
                self._send_json({"error": "Champ 'query' manquant ou vide"}, 400)
                return
            if vault_name not in VAULT_DATA:
                self._send_json(
                    {"error": f"Vault '{vault_name}' inconnu. Disponibles : {list(VAULT_DATA.keys())}"},
                    400,
                )
                return

            t0 = time.time()
            print(f"[SEARCH] vault='{vault_name}' query='{query}' top_k={top_k} rerank={use_rerank}")

            vault = VAULT_DATA[vault_name]
            chunks = vault["chunks"]
            bm25_index = vault["bm25"]

            # ─── ÉTAPE 1 : Vectoriel (bi-encoder) ───
            query_vector = np.array(get_embedding(query), dtype=np.float32)
            vector_scores = [
                (cosine_similarity(query_vector, c["vector"]), c["id"]) for c in chunks
            ]
            vector_scores.sort(key=lambda x: x[0], reverse=True)
            vector_ranking = [cid for _, cid in vector_scores]

            # ─── ÉTAPE 2 : BM25 (sparse) ───
            query_tokens = tokenize_french(query)
            bm25_scores = bm25_index.get_scores(query_tokens)
            bm25_results = [
                (bm25_scores[i], chunks[i]["id"]) for i in range(len(bm25_scores))
            ]
            bm25_results.sort(key=lambda x: x[0], reverse=True)
            bm25_ranking = [cid for _, cid in bm25_results]

            # ─── ÉTAPE 3 : Reciprocal Rank Fusion (k=60) ───
            fused = reciprocal_rank_fusion([vector_ranking, bm25_ranking], k=RRF_K)

            # ─── ÉTAPE 4 : Reranker cross-encoder ───
            candidate_ids = [cid for cid, _ in fused[:RERANK_CANDIDATES]]
            candidate_chunks = []
            for cid in candidate_ids:
                chunk = next((c for c in chunks if c["id"] == cid), None)
                if chunk:
                    candidate_chunks.append(chunk)

            results = []
            reranker_used = False

            if use_rerank and candidate_chunks:
                rerank_docs = [
                    f"[{c['source']}]\n{c['text'][:1400]}" for c in candidate_chunks
                ]
                rerank_scores = rerank(query, rerank_docs, top_n=top_k)

                if rerank_scores:
                    reranker_used = True
                    for orig_idx, rel_score in rerank_scores:
                        if orig_idx < len(candidate_chunks):
                            chunk = candidate_chunks[orig_idx]
                            vec_s = next(
                                (s for s, cid in vector_scores if cid == chunk["id"]), 0
                            )
                            bm25_s = next(
                                (s for s, cid in bm25_results if cid == chunk["id"]), 0
                            )
                            rrf_s = next(
                                (sc for cid, sc in fused if cid == chunk["id"]), 0
                            )
                            results.append({
                                "source": chunk["source"],
                                "path": chunk["path"],
                                "confidence": round(min(rel_score * 100, 99.9), 1),
                                "rerank_score": round(rel_score, 4),
                                "semantic_score": round(vec_s, 3),
                                "bm25_score": round(bm25_s, 3),
                                "rrf_score": round(rrf_s, 5),
                                "text": chunk["text"],
                            })

            # Fallback : reranker indisponible → RRF pur
            if not results:
                CANDIDATE_WINDOW = 50
                for chunk_id, rrf_score in fused[:CANDIDATE_WINDOW]:
                    confidence = (rrf_score / MAX_RRF_SCORE) * 100
                    if confidence < MIN_CONFIDENCE:
                        continue
                    chunk = next((c for c in chunks if c["id"] == chunk_id), None)
                    if not chunk:
                        continue
                    vec_s = next((s for s, cid in vector_scores if cid == chunk_id), 0)
                    bm25_s = next((s for s, cid in bm25_results if cid == chunk_id), 0)
                    results.append({
                        "source": chunk["source"],
                        "path": chunk["path"],
                        "confidence": round(confidence, 1),
                        "rerank_score": None,
                        "semantic_score": round(vec_s, 3),
                        "bm25_score": round(bm25_s, 3),
                        "rrf_score": round(rrf_score, 5),
                        "text": chunk["text"],
                    })
                    if len(results) >= top_k:
                        break

            elapsed = time.time() - t0
            print(
                f"[RESULT] {len(results)} résultats | "
                f"reranker={'OUI' if reranker_used else 'NON'} | "
                f"{elapsed:.2f}s"
            )

            self._send_json({
                "query": query,
                "vault": vault_name,
                "count": len(results),
                "reranked": reranker_used,
                "elapsed_ms": round(elapsed * 1000),
                "results": results[:top_k],
            })

        except json.JSONDecodeError:
            self._send_json({"error": "JSON invalide"}, 400)
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            self._send_json({"error": f"Erreur interne: {str(e)}"}, 500)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        sys.stderr.write(f"[HTTP] {args[0]}\n")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    initialize_rag()
    try:
        server = HTTPServer((HOST, PORT), RAGRequestHandler)
        print(f"\n🚀 RAG+Reranker sur http://{HOST}:{PORT}")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Arrêt.")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"\n❌ Port {PORT} occupé. pkill -f rag_server_rerank.py")
        else:
            raise
