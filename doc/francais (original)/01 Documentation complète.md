`Documentation complète RAG Hybrid + Reranker`

# introduction

## Qu'est-ce que le RAG ?

**RAG** (Retrieval-Augmented Generation) est une architecture qui permet à un LLM de répondre à une question en s'appuyant sur des documents précis, récupérés dynamiquement dans une base de connaissances, plutôt que sur ses seuls paramètres internes.

Concrètement, quand vous (ou un agent LLM) posez une question :

1. Le système **recherche** les passages pertinents dans vos documents (notes Obsidian, documentation technique, procédures)
2. Il **injecte** uniquement ces passages dans le contexte du LLM
3. Le LLM **répond** en citant ses sources

Sans RAG, un LLM doit soit se fier à ses connaissances figées (souvent obsolètes ou hallucinées), soit ingérer des documents entiers dans son contexte (coûteux et lent).

### Pourquoi utiliser un RAG ?

| Bénéfice | Sans RAG | Avec RAG |
|----------|----------|----------|
| **Vitesse de réponse** | Le LLM doit parcourir tout le contexte (~30k tokens) | Recherche indexée en ~20ms, seul le top-5 est envoyé au LLM |
| **Consommation de tokens** | Corpus entier injecté (milliers de pages) | 5-10 chunks pertinents (~2000 tokens) |
| **Impact écologique** | Calcul GPU/CPU maximal à chaque requête | Calcul proportionnel à la pertinence réelle |
| **Confidentialité** | Nécessite souvent un API cloud (OpenAI, etc.) | 100% local, aucune donnée ne quitte votre machine |
| **Précision** | Hallucinations fréquentes sur des faits spécifiques | Réponses sourcées, vérifiables dans vos documents |

---

## Qu'est-ce qu'un Reranker ?

Un **reranker** (ou cross-encoder) est un modèle de seconde étape qui lit la requête et chaque document candidat **conjointement**, puis produit un score de pertinence précis. Contrairement au modèle d'embedding (bi-encoder) qui encode requête et document séparément en vecteurs, le reranker traite les deux entrées ensemble à travers toutes les couches du transformer, capturant des interactions sémantiques fines que la similarité vectorielle ne voit pas.

Dans ce pipeline, le reranker reçoit les top-N candidats issus de la fusion RRF et les réordonne par pertinence réelle :

```
Top-18 Candidats RRF
        │
        ▼
[Reranker Qwen3-0.6B]
  lit (requête + document) conjointement
  → P(yes) via cls.output.weight
        │
        ▼
Résultats Finaux Classés
```

### Pourquoi utiliser un Reranker ?

| Aspect | Sans Reranker (RRF seul) | Avec Reranker |
|--------|--------------------------|---------------|
| **Précision** | Bonne pour les correspondances évidentes, faible sur les requêtes nuancées | Capture les relations sémantiques subtiles, paraphrases et terminologie métier |
| **Faux positifs** | BM25 promeut des documents contenant les mots-clés mais au contenu non pertinent | Le cross-encoder lit la paire complète et rejette le bruit des correspondances lexicales |
| **Requêtes courtes** | Les requêtes de 2-3 mots produisent des embeddings ambigus → mauvais classement | L'encodage conjoint compense la brièveté de la requête en exploitant le contexte du document |
| **Interprétabilité des scores** | Les scores RRF sont des rangs arbitraires, non comparables entre requêtes | Le reranker produit des probabilités P(yes) calibrées (0.0–1.0) |
| **Coût en latence** | ~20 ms total | +10-18 s pour 18 candidats (~580ms/candidat sur CPU) |
| **Coût en tokens pour le LLM** | Peut envoyer des chunks non pertinents, gaspillant le contexte | Seuls les chunks les plus pertinents atteignent le LLM → moins de tokens, meilleures réponses |

Le reranker est l'amélioration de qualité la plus significative du pipeline. Dans le benchmark FinanceQA de Dave Ebbelaar, l'ajout d'un reranker a amélioré le NDCG@10 de **+12 points** par rapport au retrieval hybride seul. Le coût en latence est structurel (un forward pass complet par candidat), mais le gain en précision élimine les hallucinations et l'injection de contexte non pertinent en aval.

### Quand se passer du reranker

- Exploration interactive où la vitesse prime sur la précision (alias `rag`)
- Requêtes avec des mots-clés très spécifiques où BM25 seul suffit
- Environnements à ressources limitées où +10s de latence est inacceptable
- Le serveur bascule automatiquement en RRF pur si le reranker est hors ligne

---

### Pour qui ?

- **Humains** : recherche rapide dans vos notes Obsidian, documentation technique, transcripts de réunions
- **Agents LLM** : un agent peut appeler le RAG comme un outil (`tool calling`) pour consulter votre base de connaissances avant de répondre, sans bourrer son contexte

---

### Compatibilité matérielle

Cette stack RAG fonctionne sur **tout matériel supporté par llama.cpp** :

| Backend | Statut | Notes |
|---------|--------|-------|
| CPU (x86, ARM, RISC-V) | Support complet | AVX2/AVX512/NEON auto-détecté |
| GPU NVIDIA (CUDA) | Support complet | `--n-gpu-layers all` pour vitesse max |
| Apple Silicon (Metal) | Support complet | Mémoire unifiée, pas de limite VRAM |
| GPU AMD (HIP/Vulkan) | Supporté | Via le backend Vulkan de llama.cpp |
| GPU Intel (SYCL) | Supporté | Via le backend SYCL de llama.cpp |

Aucun GPU requis — fonctionne entièrement sur CPU si nécessaire. L'accélération GPU est optionnelle et accélère l'embedding et le reranking proportionnellement.

---

# 1. Architecture

- # Pipeline de recherche

```
Query utilisateur
       │
       ├──→ [Embedding Qwen3-0.6B] ──→ Vecteur 1024d ──→ Cosine similarity ──→ Ranking vectoriel
       │                                                                              │
       └──→ [Tokenization FR] ──→ BM25Okapi ──→ Ranking BM25                         │
                                                          │                           │
                                                          └───── RRF (k=60) ──────────┘
                                                                      │
                                                              Top 21 candidats
                                                                      │
                                                          [Reranker Qwen3-0.6B]
                                                           (cross-encoder)
                                                                      │
                                                              Résultats finaux
```

- # Ports

| Port | Service | Modèle | Flags critiques |
|------|---------|--------|-----------------|
| 8181 | Embedding (bi-encoder) | Qwen3-Embedding-0.6B-Q8_0 | `--embedding --pooling last` |
| 8184 | Reranker (cross-encoder) | Qwen3-Reranker-0.6B-Q4_K_M | `--reranking --pooling rank --embedding` |
| 8182 | Serveur RAG (Python) | — | — |

- # Matériel

- MacBook Pro M1 Pro, 10 cœurs (8P + 2E), 32 Go RAM unifiée
- Void Linux Asahi, CPU uniquement (`--n-gpu-layers 0`)
- llama.cpp custom build : `$LLAMA_CPP_BIN`

---

# 2. Chemins des fichiers

- # Scripts de démarrage (llama.cpp)

```
$LLAMA_SCRIPTS_DIR/
├── start-llm-embed-qwen3-06b.sh      # Embedding 0.6B Q8_0 (port 8181) — ACTIF
├── start-llm-embed-qwen3-4b.sh       # Embedding 4B Q4_K_M (port 8181) — alternative
├── start-llm-embed-text-v1.5.sh      # Nomic v1.5 (port 8181) — legacy
├── start-llm-reranker-06b.sh         # Reranker 0.6B Q4_K_M (port 8184) — ACTIF
├── start-llm-reranker-06b_q8.sh      # Reranker 0.6B Q8_0 (port 8184) — alternative
├── start-rag-stack.sh                # Stack : embed + reranker + RAG
└── start-rag-stack-rerank.sh         # Variante stack
```

- # Serveur RAG et CLI

```
$RAG_SCRIPTS_DIR/
├── rag_server_rerank.py              # Serveur RAG principal (port 8182)
├── search_vault.sh                   # Client CLI (appelé par les alias fish)
└── test.sh                           # Script de mesure des tokens max par vault
```

- # Wrappers fish

```
$REPO_DIR/fish/
├── rag.sh                            # → search_vault.sh (alias `rag` et `ragr`)
├── rc.sh                             # → rag curl des 3 ports 
├── rs.sh                             # → lance rag_server_rerank.py en background
├── rsk.sh                            # → pkill -f rag_server_rerank.py
├── llms.sh                           # → start-rag-stack.sh &
├── llmsr.sh                          # → start-rag-stack-rerank.sh &
├── llmr.sh                           # → start-llm-reranker-06b.sh &
└── llme.sh                           # → start-llm-embed-qwen3-06b.sh &
```

- # Modèles GGUF

```
$GGUF_DIR/
├── Qwen3-Embedding-0.6B-Q8_0.gguf   # 610 Mo — embedding actif
├── Qwen3-Embedding-4B-Q4_K_M.gguf   # 2,4 Go — embedding alternatif
├── Qwen3-Reranker-0.6B-Q4_K_M.gguf  # 379 Mo — reranker actif
├── Qwen3-Reranker-0.6B.Q8_0.gguf    # 610 Mo — reranker alternatif
├── Qwen3-Reranker-4B-Q4_K_M.gguf    # 2,4 Go — non utilisé (trop lent en CPU)
└── nomic-nofr/nomic-embed-text-v1.5.Q8_0.gguf  # legacy
```

- # Vaults Obsidian

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

- # Cache d'embeddings

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

- # Binaire llama.cpp

```
$LLAMA_CPP_BIN
```

- # Environnement Python

```
$VENV_PYTHON
```

---

# 3. Configuration finale validée

- # Script embedding (`start-llm-embed-qwen3-06b.sh`)

```bash
#!/bin/bash
export LD_LIBRARY_PATH=$(dirname "$LLAMA_CPP_BIN"):$LD_LIBRARY_PATH

if command -v ss &> /dev/null && ss -tln | grep -q :8181; then
    echo "⚠️  Port 8181 déjà occupé. pkill -f Qwen3-Embedding"
    exit 1
fi

cd ~/llama-cpp-turboquant
exec ./build-cpu/bin/llama-server \
  -m $GGUF_DIR/Qwen3-Embedding-0.6B-Q8_0.gguf \
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
```

- # Script reranker (`start-llm-reranker-06b.sh`)

```bash
#!/bin/bash
export LD_LIBRARY_PATH=$(dirname "$LLAMA_CPP_BIN"):$LD_LIBRARY_PATH

if command -v ss &> /dev/null && ss -tln | grep -q :8184; then
    echo "⚠️  Port 8184 déjà occupé. pkill -f Qwen3-Reranker"
    exit 1
fi

cd ~/llama-cpp-turboquant
exec ./build-cpu/bin/llama-server \
  -m $GGUF_DIR/Qwen3-Reranker-0.6B-Q4_K_M.gguf \
  --reranking \
  --pooling rank \
  --embedding \
  --n-gpu-layers 0 \
  --threads 6 \
  --ctx-size 1024 \
  -ub 1024 \
  --host 127.0.0.1 \
  --port 8184 \
  --no-mmap \
  -ctk q8_0 \
  -ctv q8_0
```

- # Paramètres du serveur RAG (`rag_server_rerank.py`)

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `RERANK_CANDIDATES` | **21** | Maximum stable avant crash (30+ échoue). ~580ms/candidat. |
| `RRF_K` | 60 | Constante standard (Cormack et al. 2009) |
| `DEFAULT_TOP_K` | 5 | Nombre de résultats par défaut |
| `MIN_CONFIDENCE` | 50.0 | Seuil minimum en mode RRF seul |
| `MAX_CHARS` | 3000 | Troncation du texte avant embedding |
| `EMBEDDING_MODEL_ID` | `"qwen3-embed-06b"` | Identifiant de cache |
| `LLAMA_EMBED_URL` | `http://127.0.0.1:8181/embedding` | Endpoint embedding |
| `LLAMA_RERANK_URL` | `http://127.0.0.1:8184/v1/rerank` | Endpoint reranker (**/v1/rerank**, PAS /reranking) |

---

# 4. Fonctionnement détaillé

- # 4.1 Indexation (au démarrage du serveur)

1. Parcours récursif de chaque vault (`os.walk`)
2. Lecture des fichiers `.md`
3. Chunking par headers Markdown (`#`, `#`, `- #`) via `chunk_by_markdown()`
4. Filtrage anti-bruit via `is_embeddable()` :
   - Rejette les chunks commençant par ` ``` `, `<`, `|`
   - Rejette si >25% des lignes sont des commandes/logs
   - Rejette si ratio alphabétique <40%
   - Rejette si ratio caractères imprimables <95%
5. Pour chaque chunk :
   - Si le texte est dans le cache → vecteur chargé depuis le JSON
   - Sinon → appel à l'API embedding (port 8181) → vecteur calculé et caché
6. Construction de l'index BM25 (`BM25Okapi`) sur les tokens français
7. Stockage en mémoire : `{id, source, path, text, vector, tokens}`

- # 4.2 Recherche (à chaque requête)

**Étape 1 — Vectoriel (bi-encoder) :**
- La query est embeddée via `POST /embedding` (port 8181)
- Cosine similarity contre tous les vecteurs du vault
- Ranking par score décroissant

**Étape 2 — BM25 (sparse) :**
- Tokenisation française de la query (regex avec accents)
- Score BM25 contre l'index du vault
- Ranking par score décroissant

**Étape 3 — Reciprocal Rank Fusion :**
- Formule : `score(doc) = Σ 1/(k + rank)` avec k=60
- Fusionne les deux rankings en un seul
- Les scores bruts (incomparables) sont ignorés ; seul le rang compte

**Étape 4 — Reranker (cross-encoder) :**
- Les 21 premiers candidats RRF sont envoyés au reranker
- Chaque document est préfixé par son nom de fichier : `[filename.md]\n{texte[:1400]}`
- Le reranker évalue chaque paire (query, doc) conjointement via le chat template :
  ```
  <|im_start|>system
  Judge whether the Document meets the requirements based on the Query
  and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
  <|im_start|>user
  <Instruct>: Given a web search query, retrieve relevant passages that answer the query
  <Query>: {query}
  <Document>: [{filename}]\n{texte}<|im_end|>
  <|im_start|>assistant
  <think>
  </think>
  ```
- Le classifieur `cls.output.weight` projette le hidden state final vers P(yes)/P(no)
- Score de pertinence : `relevance_score = P(yes)` (0.0 → 1.0)
- Si le reranker est injoignable → fallback RRF pur (transparent)

- # 4.3 Cache

- Clé : le **texte exact** du chunk
- Valeur : le vecteur d'embedding (liste de floats)
- Nommage : `{vault}_cache__{model_id}.json`
- Le model_id (`qwen3-embed-06b`) permet de changer de modèle sans collision
- Fichier modifié → nouveaux chunks → cache miss → ré-embedding automatique
- Fichier inchangé → cache hit → instantané

- # 4.4 Endpoints API

**`GET /health`** (port 8182) :
```json
{
  "status": "ok",
  "mode": "hybrid+reranker",
  "embedding_model": "qwen3-embed-06b",
  "reranker_model": "Qwen3-Reranker-0.6B",
  "vaults": ["void", "linux", ...],
  "total_chunks": 3146,
  "port": 8182
}
```

**`POST /search`** (port 8182) :
```json
// Requête
{"vault": "void", "query": "ma question", "top_k": 3, "rerank": true}

// Réponse
{
  "query": "ma question",
  "vault": "void",
  "count": 3,
  "reranked": true,
  "elapsed_ms": 12200,
  "results": [
    {
      "source": "fichier.md",
      "path": "$OBSIDIAN_DIR/.../fichier.md",
      "confidence": 99.9,
      "rerank_score": 0.9999,
      "semantic_score": 0.847,
      "bm25_score": 12.34,
      "rrf_score": 0.03279,
      "text": "contenu du chunk..."
    }
  ]
}
```

**`POST /v1/rerank`** (port 8184) :
```json
// Requête
{"query": "...", "documents": ["doc1", "doc2"], "top_n": 3}

// Réponse
{"results": [{"index": 0, "relevance_score": 0.98}, ...]}
```

**`POST /embedding`** (port 8181) :
```json
// Requête
{"content": "texte à embedder"}

// Réponse
{"embedding": [0.012, -0.034, ...]}  // 1024 dimensions
```

---

# 5. Choix techniques et justification

- # 5.1 Qwen3-Embedding-0.6B vs Nomic v1.5 vs Qwen3-4B

| Critère | Nomic v1.5 | Qwen3-0.6B Q8_0 | Qwen3-4B Q4_K_M |
|---------|-----------|-----------------|-----------------|
| Français | Faible (anglais-centric) | Natif (100+ langues) | Natif |
| Dimensions | 768 | 1024 | 2560 |
| Famille reranker | Aucune | Qwen3-Reranker ✅ | Qwen3-Reranker ✅ |
| RAM | 150 Mo | 650 Mo | 2,8 Go |
| Vitesse query | ~15 ms | ~60 ms | ~250 ms |
| MTEB multilingue | — | 64.33 | 69.45 |

**Choix : Qwen3-0.6B Q8_0.** Le gain en français et la cohérence avec le reranker priment. Le 4B était trop lent en usage interactif avec le LLM agent en parallèle. Le reranker compense l'écart de qualité entre 0.6B et 4B.

- # 5.2 Reranker 0.6B Q4_K_M vs Q8_0 vs 4B

Benchmark Voodisss (MTEB AskUbuntuDupQuestions, 0.6B) :

| Quant | Taille | Δ NDCG@10 |
|-------|--------|-----------|
| F16 | 1,12 Go | baseline |
| Q8_0 | 610 Mo | -0,2% |
| **Q4_K_M** | **379 Mo** | **-0,3%** |
| Q4_0 | 360 Mo | -2,0% |
| Q2_K | 280 Mo | -28,7% |

**Choix : Q4_K_M.** Sweet spot officiel : 3× plus petit que F16, perte de 0,3%. Le 4B est inutilisable en CPU avec le LLM agent (trop lent, ~30-40s pour 21 candidats).

- # 5.3 Pooling

| Modèle | Pooling | Justification |
|--------|---------|---------------|
| Embedding | `--pooling last` | Blog Qwen3 : "hidden state vector corresponding to the final [EOS] token" |
| Reranker | `--pooling rank` | Active le classifieur yes/no (`cls.output.weight`). Obligatoire. |

**Note :** Le guide Voodisss indique `pooling = mean` pour l'embedding. La documentation officielle Qwen3 (blog + README) dit explicitement `last` (token [EOS]). La doc officielle fait foi.

- # 5.4 embedding ctx-size = 8192 - reranker ctx-size = 1024

Mesure des tokens maximum par vault (tokenizer Qwen3 exact) :

| Vault     | Max tokens | Couvert par 8192 ?                          |
| --------- | ---------- | ------------------------------------------- |
| void      | 9 708      | ❌ (1 fichier outlier : transcription vidéo) |
| llm       | 4 606      | ✅                                           |
| telephone | 3 463      | ✅                                           |
| terminal  | 3 141      | ✅                                           |
| browsing  | 940        | ✅                                           |
| images    | 522        | ✅                                           |
| linux     | 422        | ✅                                           |

**Choix : ctx=8192 et ub=8192.** Couvre 99% des chunks sans troncature. Le fichier outlier (9708 tokens) est tronqué proprement (début préservé). Le ctx/ub-size n'affecte pas la qualité tant que l'input tient dedans — les slots vides ne sont jamais utilisés.

**reranker ctx-size = 1024 et ub=1024** Couvre 100% des cas.
Le reranker ne traite qu'**une paire** (query + document) à la fois, soit ~515 tokens max

- # 5.5 KV cache quantization : q8_0

```bash
-ctk q8_0    # quantification des clés (K)
-ctv q8_0    # quantification des valeurs (V)
```

| Quant KV     | RAM KV cache    | Perte qualité      |
| ------------ | --------------- | ------------------ |
| f16 (défaut) | 56 Ko/token     | baseline           |
| **q8_0**     | **28 Ko/token** | **<0,1%**          |
| q4_0         | 14 Ko/token     | Notable (à éviter) |
|              |                 |                    |

**Choix : q8_0.** Moitié moins de RAM, qualité quasi identique. Gratuit en performance.

- # 5.6 RERANK_CANDIDATES = 21

Tests empiriques (M1 Pro CPU, ~580ms/candidat) :

| Candidats | Latence | Reranker | Qualité |
|-----------|---------|----------|---------|
| 5 | 2 655 ms | ✅ | Bon (mais peu de discrimination) |
| 10 | 6 235 ms | ✅ | Correct |
| **21** | **~12 200 ms** | **✅** | **Optimal (max stable)** |
| 30 | 20 501 ms | ✅ (avec --timeout 120) | Identique à 21 |
| 50 | 20 497 ms | ❌ crash → fallback RRF | — |

**Choix : 21.** Maximum fonctionnel stable. Au-delà, le reranker sature (probablement mémoire ou timeout interne). La littérature recommande 20-50 candidats (pour des rerankers GPU à 4ms/doc). Avec 21 candidats à 580ms/doc en CPU, on couvre l'équivalent de 50 candidats GPU.

- # 5.7 Inclusion du filename dans le reranker

```python
rerank_docs = [f"[{c['source']}]\n{c['text'][:1400]}" for c in candidate_chunks]
```

Sans le filename, le reranker ne peut pas matcher une query qui est littéralement le nom du fichier. Avec le filename, le cross-encoder voit la correspondance exacte et score 0.9999 au lieu de 0.9895. Testé et validé.

- # 5.8 GGUF Voodisss (obligatoire pour le reranker)

Les GGUFs communautaires de Qwen3-Reranker sont **cassés** (llama.cpp #16407). Il leur manque :
- Le tenseur `cls.output.weight` (classifieur yes/no)
- Le metadata `pooling_type=RANK`
- Le chat template de reranking

Résultat : scores poubelles (`4.5e-23`). Seuls les GGUFs de **Voodisss** (convertis avec le `convert_hf_to_gguf.py` officiel) fonctionnent.

Le reranker Qwen3 est un **reranker génératif** : le modèle produit des logits, `cls.output.weight` (tenseur `[hidden_dim, 2]`) projette le hidden state final vers P(yes) et P(no), puis softmax → `relevance_score = P(yes)`.

- # 5.9 Instruction-aware (gain 1-5%)

Les deux modèles supportent des instructions personnalisées. Qwen3 recommande :
- Écrire les instructions en **anglais** (même pour un usage multilingue)
- Instruction par défaut : `"Given a web search query, retrieve relevant passages that answer the query"`
- Gain mesuré : 1% à 5% selon les tâches

Actuellement, l'instruction par défaut est utilisée (injectée automatiquement par le chat template du reranker). Personnalisation possible ultérieurement.

---

# 6. Méthodologie (inspirée de Dave Ebbelaar)

Le pipeline suit l'architecture présentée dans "Hybrid Retrieval from Scratch" (2026) :

1. **BM25** : capture les termes exacts, identifiants, mots rares. Rate les paraphrases.
2. **Dense embeddings** : capture le sens sémantique. Rate les termes exacts.
3. **RRF** : fusionne les deux rankings par le rang (pas par le score). Les scores BM25 et cosine sont incomparables ; le rang ne l'est pas.
4. **Reranker** : réordonne les candidats en lisant conjointement query + document. C'est l'étape qui apporte le plus de gain qualitatif (NDCG +12 points dans le benchmark FinanceQA de la vidéo).

**Ce qui n'est PAS implémenté (par rapport à la vidéo) :**
- Évaluation NDCG@10 avec ground truth
- Dataset d'évaluation généré par LLM
- Comparaison systématique des configurations

---

# 7. Mise en route depuis zéro

- # Prérequis

- llama.cpp compilé (build CPU) : `$LLAMA_CPP_BIN`
- Python 3 avec venv : `~/.venv/main/`
- Packages Python : `numpy`, `requests`, `rank_bm25`
- Modèles GGUF téléchargés (Voodisss pour le reranker, Qwen officiel pour l'embedding)

- # Installation

```bash
# 1. Créer le venv et installer les dépendances
python3 -m venv ~/.venv/main
~/.venv/main/bin/pip install numpy requests rank_bm25

# 2. Placer les scripts (voir section 2 pour les chemins)
chmod +x ~/scripts/llm/rag/*.sh
chmod +x ~/scripts/rag/*.py
chmod +x ~/scripts/fish/rag/*.sh

# 3. Configurer les alias fish (voir config.fish)

# 4. Premier lancement (indexation complète)
llms
# Attendre la fin de l'indexation (~5-15 min selon la taille des vaults)
# Vérifier :
curl -s http://127.0.0.1:8182/health | jq .
```

- # Ajouter un nouveau vault

1. Ajouter l'entrée dans `VAULTS_CONFIG` dans `rag_server_rerank.py` :
```python
"docs": {
    "path": "$HOME/docs_techniques",
    "cache_file": os.path.join(CACHE_DIR, "docs_cache.json"),
},
```

2. Ajouter le nom du vault dans la regex de `search_vault.sh` :
```bash
if [[ "${1:-}" =~ ^(void|linux|browsing|terminal|llm|images|telephone|docs|all)$ ]]; then
```

3. Relancer le serveur :
```bash
pkill -f rag_server_rerank
rs
```

---

# 8. Guide d'utilisation rapide

- # Démarrage

| Commande | Action |
|----------|--------|
| `llms` | Stack complète (embedding + reranker + serveur RAG) |
| `llme` | Embedding seul (port 8181) |
| `llmr` | Reranker seul (port 8184) |
| `rs` | Serveur RAG Python seul (port 8182) |

- # Recherche

| Commande | Mode | Latence |
|----------|------|---------|
| `rag void "ma question"` | RRF seul (rapide) | ~20 ms |
| `ragr void "ma question"` | RRF + Reranker (précis) | ~12 s |
| `rag all "ma question" 5` | Tous vaults, top 5 | ~130 ms |
| `ragr linux "config nftables" 10` | Vault linux, top 10, reranké | ~12 s |
| `ragr void "question" --no-rerank` | Force le mode RRF | ~20 ms |

- # Vérification

```bash
curl -s http://127.0.0.1:8181/health | jq .   # embedding
curl -s http://127.0.0.1:8184/health | jq .   # reranker
curl -s http://127.0.0.1:8182/health | jq .   # rag
```

- # Test reranker isolé

```bash
curl -s http://127.0.0.1:8184/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"query":"nftables","documents":["nftables filtre réseau linux","il fait beau"]}' | jq .
```

- # Arrêt

| Commande | Action |
|----------|--------|
| `stop` | Tue tous les llama-server |
| `pkill -f rag_server_rerank` | Tue le serveur RAG Python |

- # Après modification d'un fichier Obsidian

```bash
pkill -f rag_server_rerank
rs
```

Le cache détecte automatiquement les chunks modifiés. Seuls les nouveaux chunks sont ré-embeddés.

- # Après changement de modèle d'embedding

```bash
rm -f ~/.rag/*__qwen3-embed-06b*
pkill -f rag_server_rerank
rs
```

- # Logs

```bash
tail -f /tmp/rag_server_rerank.log   # serveur RAG
tail -f /tmp/llm-embed-06b.log       # embedding
```

---

# 9. Debuggage

- # Le serveur RAG ne répond pas (port 8182)

```bash
ps aux | grep rag_server_rerank
tail -50 /tmp/rag_server_rerank.log

# Erreur fréquente : IndentationError après édition
$VENV_PYTHON -c "import py_compile; py_compile.compile('$RAG_SCRIPTS_DIR/rag_server_rerank.py', doraise=True)"
```

- # L'embedding ne répond pas (port 8181)

```bash
ps aux | grep llama-server | grep 8181
tail -20 /tmp/llm-embed-06b.log

# Erreur fréquente : le script pointe vers le mauvais modèle
cat ~/scripts/llm/rag/start-llm-embed-qwen3-06b.sh | grep "^\s*-m"
```

- # Le reranker ne répond pas (port 8184)

```bash
ps aux | grep llama-server | grep 8184

# Test direct
curl -s http://127.0.0.1:8184/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"query":"test","documents":["document test"]}' | jq .

# Si scores ~1e-28 → GGUF cassé (mauvaise conversion)
# Si "This server does not support reranking" → flags manquants
# Vérifier les 3 flags obligatoires :
cat ~/scripts/llm/rag/start-llm-reranker-06b.sh | grep -E "reranking|pooling|embedding"
```

- # "Connection refused" sur 8181 depuis le RAG

L'embedding n'est pas lancé. Le serveur RAG a besoin que 8181 soit actif **avant** de recevoir des requêtes.

```bash
llme    # lancer l'embedding
# attendre 3-5s
rag void "test"
```

- # Scores reranker suspects (1e-28)

Le GGUF est mal converti. Il manque `cls.output.weight`. Solution : re-télécharger depuis Voodisss.

- # Latence anormale (>20s)

```bash
# Vérifier quel modèle est chargé
ps aux | grep llama-server | grep -v grep

# Si le 4B est chargé au lieu du 0.6B → tuer et relancer avec le bon script
pkill -f llama-server
llme    # 0.6B
llmr    # reranker 0.6B
```

- # Mesurer les tokens max par vault

```bash
$RAG_SCRIPTS_DIR/test.sh
```

Nécessite `transformers` installé : `~/.venv/main/bin/pip install transformers`

---

# 10. Nettoyage

- # Supprimer le cache (ré-indexation complète)

```bash
rm -f ~/.rag/*__qwen3-embed-06b*
```

- # Supprimer un ancien cache (changement de modèle)

```bash
rm -f ~/.rag/*__nomic*
rm -f ~/.rag/*__qwen3-embed-4b*
```

- # Tuer tous les processus RAG

```bash
pkill -f llama-server
pkill -f rag_server_rerank
```

- # Vérifier la taille du cache

```bash
du -sh ~/.rag/
```

---

# 11. RAM estimée

| Composant                     | Configuration                          | RAM                |
| ----------------------------- | -------------------------------------- | ------------------ |
| Embedding 0.6B Q8_0           | ctx 8192, ub 8192, KV q8_0, parallel 1 | ~794 Mo            |
| Reranker 0.6B Q4_K_M          | ctx 1024, ub 1024 KV q8_0, parallel 1  | ~687 Mo            |
| Serveur Python (numpy + BM25) | 3146 chunks                            | ~200 Mo            |
| **Total stack RAG**           |                                        | **~1,68 Go**       |
| LLM Qwen3.6-35B-A3B Q4_K_XL   |                                        | ~20,4Go            |
| macOS + système               |                                        | ~5 Go              |
| **Total système**             |                                        | **~27 Go / 32 Go** |
|                               |                                        |                    |

---

# 12. Limites connues

- **Pas de hot-reload** : modifier un fichier Obsidian nécessite un redémarrage du serveur (`rs`)
- **Latence reranker** : ~12s en CPU pour 21 candidats (~580ms/candidat). Structurel (cross-encoder = 1 forward pass par paire)
- **RERANK_CANDIDATES > 21** : le reranker crash (timeout ou saturation mémoire). 21 est le maximum stable.
- **1 fichier outlier** : "Is Void Linux Good - With Jake from @JakeLinux.md" (9708 tokens) est tronqué à 5120 tokens. Impact négligeable (1 chunk sur 3146).
- **Pas d'évaluation NDCG** : pas de ground truth pour mesurer objectivement la qualité
- **Filtre is_embeddable()** : peut rejeter des chunks techniques légitimes (seuil 25% de lignes code)
- **Cache orphelin** : les chunks supprimés/modifiés restent dans le JSON (pas de garbage collection)
- **Single-threaded HTTP** : le serveur Python (`http.server`) ne gère qu'une requête à la fois
- **Le ctx-size ne dépend pas du nombre de chunks** : il est déterminé par la longueur maximale d'un chunk individuel, pas par la taille du corpus

---

# 13. Évolutions possibles

- [ ] Passer à `http.server.ThreadingHTTPServer` pour les requêtes concurrentes
- [ ] Ajouter un endpoint `POST /reindex` pour re-scanner un vault sans redémarrer
- [ ] Construire un dataset d'évaluation (paires query→chunk) pour mesurer NDCG@10
- [ ] Ajouter des instructions personnalisées au reranker (gain 1-5% selon Qwen)
      Instruction par défaut : "Given a web search query, retrieve relevant passages that answer the query"
      Pour un vault technique FR : "Retrieve relevant technical documentation passages that answer the query about Linux system administration"
      Note : Qwen3 recommande d'écrire les instructions en anglais même pour un usage multilingue
- [ ] Garbage collection du cache (supprimer les entrées orphelines)
- [ ] Support du modèle 4B en mode "batch offline" pour l'indexation de gros vaults
- [ ] Investiguer pourquoi RERANK_CANDIDATES > 21 fait crasher le reranker (mémoire ? timeout interne ?)
- [ ] Tester `--parallel 1` vs `--parallel 2` pour le reranker
- [ ] Explorer le model routing llama.cpp (`--models-preset models.ini`) pour servir embedding + reranker sur un seul port
- [ ] Améliorer le chunking pour les fichiers sans headers Markdown (transcriptions, articles longs)
- [ ] Remplacer la cosine similarity numpy par FAISS IndexHNSW au-delà de 50 000 chunks

---

# 14. Références

- **Pipeline hybride** : Dave Ebbelaar, "Hybrid Retrieval from Scratch" (2026)
  https://www.youtube.com/watch?v=XvKiTfd6Xvo
- **RRF** : Cormack, Clarke, Buettcher (2009) — "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
- **Qwen3-Embedding** : https://qwenlm.github.io/blog/qwen3-embedding/
  Paper : arXiv:2506.05176
- **Qwen3-Reranker HuggingFace** : https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
- **Qwen3-Reranker GGUF (Voodisss)** :
  https://huggingface.co/Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp
- **Guide multi-modèles llama.cpp (Voodisss)** :
  https://gist.github.com/VooDisss/42bce4eb5c76d3c325633886c5e348ee
- **Issue llama.cpp #16407** (GGUFs reranker cassés) :
  https://github.com/ggml-org/llama.cpp/issues/16407
- **llama.cpp** : https://github.com/ggml-org/llama.cpp

---

# 15. FAQ

**Q : Pourquoi pas de vector database (Qdrant, ChromaDB, etc.) ?**
R : Avec ~3000 chunks, un tableau numpy en mémoire suffit. La recherche cosine sur 3000 vecteurs de 1024d prend <1ms. Une vector DB ajoute de la complexité sans gain mesurable à cette échelle. Au-delà de ~100 000 chunks, reconsidérer (FAISS IndexHNSW ou Qdrant).

**Q : Pourquoi BM25 en plus du vectoriel ?**
R : Le vectoriel rate les termes exacts (noms de commandes, chemins, identifiants). BM25 les capture. Sur des vaults techniques avec beaucoup de commandes (`nftables`, `dracut`, `sfdisk`), BM25 est souvent plus fiable que le vectoriel seul.

**Q : Pourquoi le reranker est-il si lent en CPU ?**
R : Le cross-encoder fait un forward pass COMPLET du modèle pour CHAQUE paire (query, document). Contrairement au bi-encoder qui encode une fois et compare des vecteurs, le cross-encoder re-traite tout depuis zéro. Sur CPU sans GPU, un forward pass de 0.6B prend ~580ms. C'est structurel.

**Q : Puis-je utiliser le RAG sans reranker ?**
R : Oui. `rag` (alias) force `--no-rerank`. Le serveur fonctionne aussi si le reranker n'est pas lancé (fallback RRF automatique). Latence : ~20ms.

**Q : Puis-je utiliser le RAG sans embedding ?**
R : Non. L'embedding est nécessaire pour la recherche vectorielle. Sans le serveur d'embedding (port 8181), le serveur RAG retourne une erreur 500.

**Q : Comment ajouter un vault sans tout ré-indexer ?**
R : Ajouter l'entrée dans VAULTS_CONFIG, relancer le serveur. Les vaults existants sont chargés depuis le cache (instantané). Seul le nouveau vault est indexé from scratch.

**Q : Le cache est-il compatible entre modèles ?**
R : Non. Le nom de fichier inclut l'identifiant modèle (`void_cache__qwen3-embed-06b.json`). Changer de modèle crée un nouveau fichier de cache.

**Q : Pourquoi les scores reranker sont-ils tous >0.99 ?**
R : Le reranker Qwen3 utilise un classifieur yes/no avec softmax. Sur des documents clairement pertinents, P(yes) → 1.0. La discrimination se fait sur les documents marginaux (scores 0.3-0.8).

**Q : Le ctx-size dépend-il du nombre de chunks dans le vault ?**
R : Non. Le ctx-size définit la longueur maximale d'un seul input. Que le vault contienne 3 000 ou 300 000 chunks, l'embedding reçoit toujours un chunk à la fois (~910 tokens max) et le reranker toujours une paire query+document (~500 tokens max). Le ctx-size est déterminé par la longueur du plus gros chunk individuel, pas par la taille du corpus.

**Q : Pourquoi `--pooling last` et pas `--pooling mean` pour l'embedding ?**
R : Le blog officiel Qwen3 dit : "The Embedding model processes a single text segment as input, extracting the semantic representation by utilizing the hidden state vector corresponding to the final [EOS] token." C'est `last`. Le guide Voodisss indique `mean`, mais la documentation officielle Qwen3 fait foi.

---

# 16. Glossaire

| Terme | Définition |
|-------|-----------|
| **Bi-encoder** | Encode query et document séparément → vecteurs → cosine similarity. Rapide mais perd les nuances. |
| **Cross-encoder** | Encode query + document conjointement → score de pertinence. Lent mais précis. |
| **Reranker génératif** | Le reranker Qwen3 utilise un classifieur (cls.output.weight) qui projette le hidden state vers P(yes)/P(no). Ce n'est pas un cross-encoder traditionnel. |
| **RRF** | Reciprocal Rank Fusion. Fusionne des rankings par le rang, pas par le score. Formule : 1/(k+rank). |
| **BM25** | Algorithme de recherche par mots-clés avec pondération TF-IDF. Capture les termes exacts. |
| **Pooling last** | Le vecteur d'embedding est le hidden state du dernier token [EOS]. |
| **Pooling rank** | Mode classifieur pour le reranker. Extrait les logits yes/no via cls.output.weight. |
| **KV cache** | Mémoire d'attention (clés + valeurs) allouée par token de contexte. Quantifiable via -ctk/-ctv. |
| **Chunk** | Fragment de document issu du découpage Markdown. Unité d'indexation. |
| **Vault** | Répertoire Obsidian indexé comme une collection distincte. |
| **GGUF** | Format de fichier pour les modèles quantifiés utilisés par llama.cpp. |
| **Quantization** | Réduction de la précision des poids (F16→Q8→Q4) pour réduire la RAM. |
| **MTEB** | Massive Text Embedding Benchmark. Classement de référence pour les modèles d'embedding. |
| **NDCG@K** | Normalized Discounted Cumulative Gain. Métrique de qualité de retrieval (0-1). |
| **MRL** | Matryoshka Representation Learning. Permet des dimensions d'embedding flexibles (32 à 2560 pour Qwen3-4B). |
| **Instruction-aware** | Capacité du modèle à adapter son comportement selon une instruction personnalisée (gain 1-5%). |

---

`un fichier n'apparait pas`
# 1. Le fichier "01 Documentation complète.md" n'est probablement pas indexé

La section "# 11. RAM estimée" est un **tableau Markdown** :

```
# 11. RAM estimée

| Composant | Configuration | RAM |
|-----------|--------------|-----|
| Embedding 0.6B Q8_0 | ctx 5120, KV q8_0 | ~794 Mo |
| Reranker 0.6B Q4_K_M | ctx 5120, KV q8_0 | ~687 Mo |
...
```

Le filtre `is_embeddable()` rejette probablement ce chunk à cause du **ratio alphabétique**. Les tableaux Markdown sont pleins de `|`, `-`, chiffres et espaces — le ratio de caractères alphabétiques tombe sous le seuil de 0.40.

**Vérifie :**

```bash
$VENV_PYTHON -c "
import sys
sys.path.insert(0, '$RAG_SCRIPTS_DIR')
from rag_server_rerank import is_embeddable, chunk_by_markdown

with open('$OBSIDIAN_DIR/004 llm 000/07 - rag/01 Documentation complète.md', 'r') as f:
    content = f.read()

chunks = chunk_by_markdown(content)
print(f'Chunks totaux : {len(chunks)}')
for i, c in enumerate(chunks):
    if 'RAM estimée' in c or 'ram estimée' in c.lower():
        alpha = sum(ch.isalpha() for ch in c) / max(len(c), 1)
        print(f'--- Chunk {i} (alpha={alpha:.3f}, embeddable={is_embeddable(c)}) ---')
        print(c[:200])
"
```

Si `embeddable=False` → confirmé, le filtre rejette le chunk.

**Fix : abaisser le seuil alpha de 0.40 à 0.36** dans `rag_server_rerank.py` :

```python
# Avant :
if alpha_ratio < 0.40:
    return False

# Après :
if alpha_ratio < 0.36:
    return False
```

Puis ré-indexer :

```bash
rm -f ~/.rag/*__qwen3-embed-06b*
pkill -f rag_server_rerank
rs
```

# 2. Seulement 3 résultats : le défaut est `top_k=3`

Dans `search_vault.sh` :

```bash
# Avant :
TOP_K="${3:-3}"

# Après :
TOP_K="${3:-7}"
```

donne 7 résultats
