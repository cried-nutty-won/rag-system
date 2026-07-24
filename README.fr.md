# RAG Hybride + Reranker  [→ en](README.md)

# introduction

## Qu'est-ce qu'un RAG ?

Le **RAG** (Retrieval-Augmented Generation — Génération Augmentée par Récupération) est une architecture qui permet à un LLM de répondre à des questions en récupérant des documents pertinents depuis une base de connaissances et en les injectant dans le contexte, plutôt que de se fier uniquement à ses paramètres internes.

Concrètement, quand vous (ou un agent LLM) posez une question :

1. Le système **récupère** les passages pertinents de vos documents (notes Obsidian, docs techniques, procédures)
2. Il **injecte** uniquement ces passages dans le contexte du LLM
3. Le LLM **répond** avec des citations de sources

Sans RAG, un LLM doit soit se fier à ses connaissances d'entraînement figées (souvent obsolètes ou hallucinées), soit ingérer des documents entiers dans sa fenêtre de contexte (coûteux et lent).

### Pourquoi utiliser le RAG ?

| Avantage | Sans RAG | Avec RAG |
|----------|----------|----------|
| **Vitesse de réponse** | Le LLM doit traiter tout le contexte (~30k tokens) | Recherche indexée en ~20ms, seuls les top-5 envoyés au LLM |
| **Consommation de tokens** | Corpus entier injecté (milliers de pages) | 5-10 chunks pertinents (~2000 tokens) |
| **Impact écologique** | Calcul GPU/CPU maximal par requête | Calcul proportionnel à la pertinence réelle |
| **Confidentialité** | Nécessite souvent des APIs cloud (OpenAI, etc.) | 100% local, aucune donnée ne quitte votre machine |
| **Précision** | Hallucinations fréquentes sur des faits spécifiques | Réponses sourcées, vérifiables dans vos documents |

---

## Qu'est-ce qu'un Reranker ?

Un **reranker** (ou cross-encoder) est un modèle de récupération de second stade qui lit la requête et chaque document candidat **conjointement**, puis produit un score de pertinence précis. Contrairement au modèle d'embedding (bi-encoder) qui encode requête et document séparément en vecteurs, le reranker traite les deux entrées ensemble à travers toute la pile transformer, capturant des interactions sémantiques fines que la similarité vectorielle manque.

Dans ce pipeline, le reranker reçoit les top-N candidats de l'étape de fusion RRF et les réordonne par pertinence réelle :

```
Top-18 Candidats RRF
        │
        ▼
[Reranker Qwen3-0.6B]
  lit (requête + document) conjointement
  → P(oui) via cls.output.weight
        │
        ▼
Résultats Classés Finaux
```

### Pourquoi utiliser un Reranker ?

| Aspect | Sans Reranker (RRF seul) | Avec Reranker |
|--------|--------------------------|---------------|
| **Précision** | Bon pour les correspondances évidentes, faible sur les requêtes nuancées | Capture les relations sémantiques subtiles, paraphrases et terminologie spécifique au domaine |
| **Faux positifs** | BM25 promeut des documents avec mots-clés correspondants mais contenu non pertinent | Le cross-encoder lit la paire complète et rejette le bruit de correspondance de mots-clés |
| **Requêtes courtes** | Les requêtes de 2-3 mots produisent des embeddings ambigus → mauvais classement | L'encodage conjoint compense la brièveté de la requête en exploitant le contexte du document |
| **Interprétabilité des scores** | Les scores RRF sont des rangs arbitraires, non comparables entre requêtes | Le reranker produit des probabilités P(oui) calibrées (0.0–1.0) |
| **Coût de latence sur CPU avec modèles 0.6B** | ~20 ms | 12 s pour 18 candidats |
| **Coût de latence sur GPU avec modèles 0.6B** | ~3 ms | 1 s pour 100 candidats |
| **Coût de latence sur GPU avec modèles 4B** | ~30 ms | 3 s pour 100 candidats |
| **Coût en tokens pour le LLM** | Peut envoyer des chunks non pertinents, gaspillant le contexte | Seuls les chunks les plus pertinents atteignent le LLM → moins de tokens, meilleures réponses |

Le reranker est l'amélioration qualitative la plus importante du pipeline. Dans le benchmark FinanceQA de Dave Ebbelaar, l'ajout d'un reranker a amélioré le NDCG@10 de **+12 points** par rapport à la récupération hybride seule. Le coût de latence est structurel (un forward pass complet par candidat), mais le gain de précision élimine les hallucinations et l'injection de contexte non pertinent en aval.

### Quand se passer du reranker

- Exploration interactive où la vitesse importe plus que la précision (alias `rag`)
- Requêtes avec des mots-clés très spécifiques où BM25 seul suffit
- Environnements à ressources limitées où +10s de latence est inacceptable
- Le serveur bascule automatiquement en RRF pur si le reranker est hors ligne

---

## Pour qui ?

- **Humains** : recherche rapide dans les notes Obsidian, documentation technique, transcriptions de réunions
- **Agents LLM** : un agent peut appeler le RAG comme outil (`tool calling`) pour consulter votre base de connaissances avant de répondre, sans saturer sa fenêtre de contexte

---

## Compatibilité Matérielle

### Cette pile RAG fonctionne sur **tout matériel supporté par llama.cpp** :

| Backend | Statut | Notes |
|---------|--------|-------|
| CPU (x86, ARM, RISC-V) | Support complet | AVX2/AVX512/NEON auto-détectés |
| GPU NVIDIA (CUDA) | Support complet | `--n-gpu-layers all` pour vitesse max |
| Apple Silicon (Metal) | Support complet | Mémoire unifiée, pas de limite VRAM |
| GPU AMD (HIP/Vulkan) | Supporté | Via le backend Vulkan de llama.cpp |
| GPU Intel (SYCL) | Supporté | Via le backend SYCL de llama.cpp |

Pas de GPU requis — fonctionne entièrement sur CPU si nécessaire. L'accélération GPU est optionnelle et accélère proportionnellement l'embedding et le reranking.

---

### Autres Backends

| Scénario | Backend Recommandé | Pourquoi |
|----------|-------------------|----------|
| Multi-utilisateurs, cluster GPU (DGX, etc.) | vLLM | Batching natif, PagedAttention, sessions concurrentes |
| Production à haut débit | SGLang | Cache de préfixe RadixAttention, scheduler optimisé |
| Prototypage rapide, embedding seul | Ollama | Gestion de modèles zéro-config |
| Mixte : embedding sur GPU + reranker sur CPU | llama.cpp + vLLM | Chaque backend sert ce qu'il fait de mieux |

> **Note :** Ollama ne supporte pas le reranking. Quand vous utilisez Ollama pour les embeddings, désactivez le reranking (`--no-rerank`).

---

### Résumé de Compatibilité des Backends

| Backend | Embedding | Reranking | Mode RAG |
|---|---|---|---|
| llamacpp | ✅ `POST /embedding` | ✅ `POST /v1/rerank` | Hybride + Reranker |
| vLLM | ✅ `POST /v1/embeddings` | ✅ `POST /v1/rerank` | Hybride + Reranker |
| sglang | ✅ `POST /v1/embeddings` | ✅ `POST /v1/rerank` | Hybride + Reranker |
| ollama | ✅ `POST /api/embeddings` | ❌ | Hybride (RRF seul) |

Le serveur RAG s'adapte automatiquement : si le reranker est injoignable, il bascule en RRF pur sans erreur.

#### Configuration du serveur RAG pour chaque backend

```bash
# llamacpp (défaut)
export LLAMA_EMBED_URL="http://127.0.0.1:8181/embedding"
export LLAMA_RERANK_URL="http://127.0.0.1:8184/v1/rerank"

# vLLM
export LLAMA_EMBED_URL="http://127.0.0.1:8000/v1/embeddings"
export LLAMA_RERANK_URL="http://127.0.0.1:8001/v1/rerank"

# sglang
export LLAMA_EMBED_URL="http://127.0.0.1:8000/v1/embeddings"
export LLAMA_RERANK_URL="http://127.0.0.1:8001/v1/rerank"

# ollama (embedding seul, pas de reranker)
export LLAMA_EMBED_URL="http://127.0.0.1:11434/api/embeddings"
# Pas d'URL reranker — bascule RRF automatique
```

---

## Choix des Modèles

| Matériel | Embedding | Reranker | Pourquoi |
|----------|-----------|----------|----------|
| CPU seul (8 Go RAM) | 0.6B Q8_0 | 0.6B Q4_K_M | Tient en RAM, latence interactive |
| GPU (6+ Go VRAM) | 4B Q4_K_M | 4B Q4_K_M | Meilleure qualité, ~3s pour 100 candidats |
| GPU (24+ Go VRAM) | 4B F16 | 4B F16 | Qualité maximale, pas de perte de quantification |

### Modèles GGUF

| Modèle | Quant | Taille | Matériel | MTEB |
|--------|-------|--------|----------|------|
| Qwen3-Embedding-0.6B | Q8_0 | 610 Mo | CPU ou GPU | 64.33 |
| Qwen3-Embedding-4B | Q4_K_M | 2.4 Go | GPU recommandé | 69.45 |
| Qwen3-Reranker-0.6B | Q4_K_M | 379 Mo | CPU ou GPU | 65.80 |
| Qwen3-Reranker-4B | Q4_K_M | 2.4 Go | GPU recommandé | 69.76 |

> **Note :** Les scores MTEB de l'embedding et du reranker ne sont **pas comparables** — ils évaluent des tâches différentes (récupération vectorielle vs. reclassement de paires). Le gain réel du reranker dans le pipeline est de **+12 points NDCG@10** (benchmark FinanceQA), pas la différence entre les deux scores MTEB ci-dessus.

- Embedding : [Qwen/Qwen3-Embedding-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF) ou [Qwen/Qwen3-Embedding-4B-GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-4B-GGUF) (officiel)
- Reranker : [Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp](https://huggingface.co/Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp) ou [Voodisss/Qwen3-Reranker-4B-GGUF-llama_cpp](https://huggingface.co/Voodisss/Qwen3-Reranker-4B-GGUF-llama_cpp) (**obligatoire** — les GGUF communautaires sont cassés, voir [llama.cpp #16407](https://github.com/ggml-org/llama.cpp/issues/16407))

---

### MTEB (Massive Text Embedding Benchmark)

Le MTEB est le benchmark de référence pour évaluer la qualité des modèles d'embedding. Il mesure la capacité d'un modèle à produire des vecteurs qui capturent le sens du texte, à travers **8 types de tâches** :

| Tâche | Ce qu'elle mesure | Exemple |
|-------|-------------------|---------|
| **Récupération** | Trouver le bon document parmi des milliers | "Quelle est la procédure nftables ?" → trouver le bon fichier |
| **Reranking** | Réordonner les candidats par pertinence | Classer 18 chunks du plus au moins pertinent |
| **Classification** | Catégoriser un texte | "Ce document parle-t-il de réseau ou de stockage ?" |
| **Clustering** | Regrouper des textes similaires | Grouper les notes par sujet |
| **STS** (Similarité Textuelle Sémantique) | Mesurer la similarité entre deux phrases | "pare-feu nftables" ≈ "règles de pare-feu nftables" |
| **Classification de paires** | Déterminer si deux textes sont liés | "Cette procédure correspond-elle à cette question ?" |
| **Extraction de bitexte** | Trouver la traduction correspondante | FR ↔ EN |
| **Résumé** | Évaluer la qualité d'un résumé | — |

Le score MTEB **Récupération** est le plus important pour le RAG : il mesure directement la capacité du modèle à trouver le bon document. Plus le score est élevé, moins le RAG a besoin du reranker pour compenser.

| Modèle | MTEB Multilingue | MTEB Récupération | Dimensions |
|--------|------------------|-------------------|------------|
| Qwen3-Embedding-0.6B | 64.33 | 64.64 | 1024 |
| Qwen3-Embedding-4B | 69.45 | 69.60 | 2560 |
| Qwen3-Embedding-8B | 70.58 | 70.88 | 4096 |

Le 0.6B est suffisant pour un RAG local avec reranker. Le 4B ajoute +5 points mais nécessite un GPU.

---
---

# Installation

### Installation Rapide (recommandée)

```bash
mkdir -p ~/rag
git clone https://github.com/cried-nutty-won/rag-system.git
cd ~/rag/rag-system
bash install.sh
```

L'installateur interactif gère tout :
- Détecte l'OS, la RAM, le GPU (NVIDIA, Apple Silicon, lspci)
- Détecte le shell (fish, bash, zsh, sh) et écrit les alias dans la bonne config
- Propose les modèles 0.6B (défaut) ou 4B (GPU uniquement — masqués sur CPU)
- Télécharge les GGUF depuis Qwen officiel + Voodisss
- Configure les vaults interactivement (Obsidian + documentation)
- Installe 10 raccourcis shell

Tester sans rien modifier : `bash install.sh --dry-run`

Pour désinstaller : `bash uninstall.sh`

---

### Installation Manuelle

#### Prérequis

- **llama.cpp** compilé avec support CPU (ou CUDA/Metal/Vulkan pour accélération GPU) ou vLLM, SGLang, ollama
- **Python 3.10+** avec un environnement virtuel
- **Modèles GGUF** :
  - Embedding : `Qwen3-Embedding-0.6B-Q8_0.gguf` ([Qwen officiel](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF))
  - Reranker : `Qwen3-Reranker-0.6B-Q4_K_M.gguf` (**doit provenir de [Voodisss](https://huggingface.co/Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp)** — les GGUF communautaires sont cassés, voir [llama.cpp #16407](https://github.com/ggml-org/llama.cpp/issues/16407))

### Étape 1 : Cloner et configurer

```bash
git clone https://github.com/cried-nutty-won/rag-system.git
cd rag-system
cp config.sh.example config.sh
# Éditer config.sh avec vos chemins réels
```
Éditer `config.sh` pour correspondre à votre environnement :

```bash
LLAMA_CPP_BIN="$HOME/llama-cpp-turboquant/build-cpu/bin/llama-server"
GGUF_DIR="$HOME/models/GGUF/rag"
OBSIDIAN_DIR="$HOME/obsidian"
VENV_PYTHON="$HOME/.venv/main/bin/python3"
RAG_SCRIPTS_DIR="$(pwd)/server"
LLAMA_SCRIPTS_DIR="$(pwd)/llama"
LOG_DIR="/tmp"
```

### Étape 2 : Dépendances Python

```bash
python3 -m venv ~/.venv/main
~/.venv/main/bin/pip install numpy requests rank_bm25
```

### Étape 3 : Télécharger les modèles

```bash
mkdir -p $GGUF_DIR

# Embedding (GGUF Qwen officiel)
huggingface-cli download Qwen/Qwen3-Embedding-0.6B-GGUF \
  Qwen3-Embedding-0.6B-Q8_0.gguf --local-dir $GGUF_DIR

# Reranker (Voodisss UNIQUEMENT — ne PAS utiliser d'autres sources)
huggingface-cli download Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp \
  Qwen3-Reranker-0.6B-Q4_K_M.gguf --local-dir $GGUF_DIR
```

### Étape 4 : Configurer les vaults

Éditer `server/rag_server_rerank.py` et mettre à jour `VAULTS_CONFIG` avec les chemins de vos vaults Obsidian, documentation ou transcriptions :

```python
VAULTS_CONFIG = {
    "void":     {"path": os.path.join(OBSIDIAN_DIR, "001 Void 000")},
    "linux":    {"path": os.path.join(OBSIDIAN_DIR, "000 linux 000")},
    # Ajoutez vos vaults ici
}
```

### Étape 5 : Premier lancement

```bash
# Démarrer la pile complète (embedding + reranker + serveur RAG)
bash llama/start-rag-llm_embed_reranker_server.sh

# Attendre l'indexation (~5-15 min au premier lancement, instantané aux suivants via le cache)
# Vérifier la santé :
curl -s http://127.0.0.1:8182/health | jq .
```

Sortie attendue (exemple) :

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

### Étape 6 : Alias shell (10 raccourcis)

L'installateur les ajoute automatiquement. Pour une configuration manuelle, ajouter à votre config shell
(`~/.config/fish/config.fish`, `~/.bashrc`, `~/.zshrc`) :

| Commande | Action |
|----------|--------|
| `llmers` | Démarrer la pile complète (embedding + reranker + serveur RAG) |
| `llmes` | Démarrer embedding + serveur RAG (sans reranker) |
| `llme` | Démarrer l'embedding seul (port 8181) |
| `llmr` | Démarrer le reranker seul (port 8184) |
| `rs` | Démarrer le serveur RAG Python seul (port 8182) |
| `rst` | Tail -f des logs du serveur RAG |
| `rag <vault> "<requête>"` | Recherche rapide (~20ms) |
| `ragr <vault> "<requête>"` | Recherche lente et précise avec reranker (~10-18s CPU, ~1s GPU) |
| `rc` | Vérification de santé des 3 services |
| `rsk` | Tuer le serveur RAG Python |

```bash
# Fish
alias llmers='bash /chemin/vers/rag-system/llama/start-rag-llm_embed_reranker_server.sh &'
alias llmes='bash /chemin/vers/rag-system/llama/start-rag-llm_embed_server.sh &'
alias llme='bash /chemin/vers/rag-system/llama/start-llm-embed-qwen3-06b.sh &'
alias llmr='bash /chemin/vers/rag-system/llama/start-llm-reranker-06b.sh &'
alias rs='bash /chemin/vers/rag-system/server/rag_server_rerank.py &'
alias rst='tail -f /tmp/rag_server_rerank.log'
alias rag='bash /chemin/vers/rag-system/server/search_vault.sh --no-rerank'
alias ragr='bash /chemin/vers/rag-system/server/search_vault.sh'
alias rc='curl -s http://127.0.0.1:8182/health | jq .'
alias rsk='pkill -f rag_server_rerank'
```

### Dépannage de l'installation

| Problème | Solution |
|----------|----------|
| Scores reranker ~`1e-28` | Mauvaise source GGUF. Re-télécharger depuis [Voodisss](https://huggingface.co/Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp) |
| `"This server does not support reranking"` | Flags manquants. S'assurer que `--reranking --pooling rank --embedding` sont tous présents |
| Port déjà utilisé | `pkill -f llama-server && pkill -f rag_server_rerank` puis redémarrer |
| OOM au démarrage | Ajouter `--cache-ram 0` aux deux scripts llama-server (désactive le cache de prompt hôte de 8 Gio) |
| Première indexation lente | Normal. Les démarrages suivants utilisent les embeddings en cache (instantané) |

---
---

## 1. Architecture

### Pipeline de Recherche

```
Requête Utilisateur
       │
       ├──→ [Embedding Qwen3-0.6B] ──→ Vecteur 1024d ──→ Similarité cosinus ──→ Classement vectoriel
       │                                                                              │
       └──→ [Tokenisation FR] ──→ BM25Okapi ──→ Classement BM25                      │
                                                          │                           │
                                                          └───── RRF (k=60) ──────────┘
                                                                      │
                                                              Top 18 candidats
                                                                      │
                                                          [Reranker Qwen3-0.6B]
                                                           (cross-encoder)
                                                                      │
                                                              Résultats finaux
```

### Ports

| Port | Service | Modèle | Flags Critiques |
|------|---------|--------|-----------------|
| 8181 | Embedding (bi-encoder) | Qwen3-Embedding-0.6B-Q8_0 | `--embedding --pooling last` |
| 8184 | Reranker (cross-encoder) | Qwen3-Reranker-0.6B-Q4_K_M | `--reranking --pooling rank --embedding` |
| 8182 | Serveur RAG (Python) | — | — |

### Matériel (exemple)

- À partir de 8 Go de RAM unifiée
- Linux CPU uniquement (`--n-gpu-layers 0`) => retirer ce flag pour utilisation GPU selon votre matériel
- Build personnalisé llama.cpp : `$LLAMA_CPP_BIN`

---

## 2. Chemins des Fichiers

### Scripts de Démarrage (llama.cpp)

```
$LLAMA_SCRIPTS_DIR/
├── start-llm-embed-qwen3-06b.sh      # Embedding 0.6B Q8_0 (port 8181) — ACTIF
├── start-llm-embed-qwen3-4b.sh       # Embedding 4B Q4_K_M (port 8181) — alternatif
├── start-llm-reranker-06b.sh         # Reranker 0.6B Q4_K_M (port 8184) — ACTIF
├── start-rag-llm_embed_reranker_server.sh  # Pile : embed + reranker + RAG
└── start-rag-llm_embed_server.sh     # Pile : embed + RAG (sans reranker)
```

### Serveur RAG et CLI

```
$RAG_SCRIPTS_DIR/
├── rag_server_rerank.py              # Serveur RAG principal (port 8182)
├── search_vault.sh                   # Client CLI (appelé par les alias fish)
└── test_tokens.sh                    # Script de mesure des tokens max par vault
```

### Wrappers Fish (exemple : ajouter l'alias rag='chemin vers rag.sh' dans votre fichier config.fish) pour utiliser le raccourci rag

```
$REPO_DIR/fish/
├── rag.sh                            # → search_vault.sh (alias `rag` et `ragr`)
├── rc.sh                             # → vérification de santé des 3 ports
├── rs.sh                             # → lance rag_server_rerank.py en arrière-plan
├── rsk.sh                            # → pkill -f rag_server_rerank.py
├── rst.sh                            # → tail -f des logs du serveur RAG
├── llmers.sh                         # → start-rag-llm_embed_reranker_server.sh &
├── llmes.sh                          # → start-rag-llm_embed_server.sh &
├── llmr.sh                           # → start-llm-reranker-06b.sh &
└── llme.sh                           # → start-llm-embed-qwen3-06b.sh &
```

### Modèles GGUF (exemple)

```
$GGUF_DIR/
├── Qwen3-Embedding-0.6B-Q8_0.gguf   # 610 Mo — embedding actif
├── Qwen3-Embedding-4B-Q4_K_M.gguf   # 2.4 Go — embedding alternatif
├── Qwen3-Reranker-0.6B-Q4_K_M.gguf  # 379 Mo — reranker actif
├── Qwen3-Reranker-0.6B.Q8_0.gguf    # 610 Mo — reranker alternatif
├── Qwen3-Reranker-4B-Q4_K_M.gguf    # 2.4 Go — inutilisé (trop lent sur CPU)
└── nomic-nofr/nomic-embed-text-v1.5.Q8_0.gguf  # legacy
```

### Vaults Obsidian (exemple)

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

### Cache d'Embedding (exemple)

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

### Binaire llama.cpp

```
$LLAMA_CPP_BIN
```

### Environnement Python

```
$VENV_PYTHON
```

---

## 3. Configuration Finale Validée

### Script d'Embedding (`start-llm-embed-qwen3-06b.sh`)

```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"

export LD_LIBRARY_PATH="$(dirname "$LLAMA_CPP_BIN"):$LD_LIBRARY_PATH"

if command -v ss &> /dev/null && ss -tln | grep -q :8181; then
    echo "⚠️  Port 8181 déjà utilisé. pkill -f Qwen3-Embedding"
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

### Script de Reranker (`start-llm-reranker-06b.sh`)

```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"

export LD_LIBRARY_PATH="$(dirname "$LLAMA_CPP_BIN"):$LD_LIBRARY_PATH"

if command -v ss &> /dev/null && ss -tln | grep -q :8184; then
    echo "⚠️  Port 8184 déjà utilisé. pkill -f Qwen3-Reranker"
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

### Paramètres du Serveur RAG (`rag_server_rerank.py`)

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `RERANK_CANDIDATES` | **18** | Maximum stable avant crash (30+ échoue). ~580ms/candidat. |
| `RRF_K` | 60 | Constante standard (Cormack et al. 2009) |
| `DEFAULT_TOP_K` | 5 | Nombre de résultats par défaut |
| `MIN_CONFIDENCE` | 50.0 | Seuil minimum en mode RRF seul |
| `MAX_CHARS` | 3000 | Troncation du texte avant embedding |
| `EMBEDDING_MODEL_ID` | `"qwen3-embed-06b"` | Identifiant de cache |
| `alpha_ratio` | **0.36** | Seuil pour le filtre `is_embeddable()` |
| `LLAMA_EMBED_URL` | `http://127.0.0.1:8181/embedding` | Endpoint d'embedding |
| `LLAMA_RERANK_URL` | `http://127.0.0.1:8184/v1/rerank` | Endpoint du reranker (**/v1/rerank**, PAS /reranking) |

---

## 4. Fonctionnement Détaillé

### 4.1 Indexation (au démarrage du serveur)

1. Parcours récursif de chaque vault (`os.walk`)
2. Lecture des fichiers `.md`
3. Découpage en chunks par en-têtes Markdown (`#`, `##`, `###`) via `chunk_by_markdown()`
4. Filtrage anti-bruit via `is_embeddable()` :
   - Rejette les chunks commençant par ` ``` `, `<`, `|`
   - Rejette si >25% des lignes sont des commandes/logs
   - Rejette si le ratio alphabétique est <36%
   - Rejette si le ratio de caractères imprimables est <95%
5. Pour chaque chunk :
   - Si le texte est en cache → vecteur chargé depuis le JSON
   - Sinon → appel à l'API d'embedding (port 8181) → vecteur calculé et mis en cache
6. Construction de l'index BM25 (`BM25Okapi`) sur les tokens français
7. Stockage en mémoire : `{id, source, path, text, vector, tokens}`

### 4.2 Recherche (par requête)

**Étape 1 — Vectoriel (bi-encoder) :**
- La requête est embeddée via `POST /embedding` (port 8181)
- Similarité cosinus contre tous les vecteurs du vault
- Classement par score décroissant

**Étape 2 — BM25 (sparse) :**
- Tokenisation française de la requête (regex avec accents)
- Score BM25 contre l'index du vault
- Classement par score décroissant

**Étape 3 — Reciprocal Rank Fusion :**
- Formule : `score(doc) = Σ 1/(k + rank)` avec k=60
- Fusionne les deux classements en un seul
- Les scores bruts (incomparables) sont ignorés ; seul le rang compte

**Étape 4 — Reranker (cross-encoder) :**
- Les top 18 candidats RRF sont envoyés au reranker
- Chaque document est préfixé par son nom de fichier : `[nom_fichier.md]\n{texte[:1400]}`
- Le reranker évalue chaque paire (requête, doc) conjointement via le chat template :
  ```
  <|im_start|>system
  Judge whether the Document meets the requirements based on the Query
  and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
  <|im_start|>user
  <Instruct>: Given a web search query, retrieve relevant passages that answer the query
  <Query>: {requête}
  <Document>: [{nom_fichier}]\n{texte}<|im_end|>
  <|im_start|>assistant
  <think>
   </think>
  ```
- Le classifieur `cls.output.weight` projette l'état caché final vers P(oui)/P(non)
- Score de pertinence : `relevance_score = P(oui)` (0.0 → 1.0)
- Si le reranker est injoignable → bascule en RRF pur (transparent)

### 4.3 Cache

- Clé : le **texte exact** du chunk
- Valeur : le vecteur d'embedding (liste de flottants)
- Nommage : `{vault}_cache__{model_id}.json`
- Le model_id (`qwen3-embed-06b`) permet de changer de modèle sans collision
- Fichier modifié → nouveaux chunks → cache miss → re-embedding automatique
- Fichier inchangé → cache hit → instantané

### 4.4 Endpoints API

**`GET /health`** (port 8182) :
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

**`POST /search`** (port 8182) :
```json
// Requête
{"vault": "void", "query": "ma question", "top_k": 5, "rerank": true}

// Réponse
{
  "query": "ma question",
  "vault": "void",
  "count": 5,
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

## 5. Choix Techniques et Justification

### 5.1 Qwen3-Embedding-0.6B vs Nomic v1.5 vs Qwen3-4B

| Critère | Nomic v1.5 | Qwen3-0.6B Q8_0 | Qwen3-4B Q4_K_M |
|---------|-----------|-----------------|-----------------|
| Français | Faible (centré anglais) | Natif (100+ langues) | Natif |
| Dimensions | 768 | 1024 | 2560 |
| Famille Reranker | Aucune | Qwen3-Reranker ✅ | Qwen3-Reranker ✅ |
| RAM | 150 Mo | 650 Mo | 2.8 Go |
| Vitesse requête | ~15 ms | ~60 ms | ~250 ms |
| MTEB multilingue | — | 64.33 | 69.45 |

**Choix : Qwen3-0.6B Q8_0.** Le gain en français et la cohérence avec le reranker sont prioritaires. Le 4B était trop lent pour un usage interactif aux côtés de l'agent LLM. Le reranker compense l'écart de qualité entre 0.6B et 4B.

### 5.2 Reranker 0.6B Q4_K_M vs Q8_0 vs 4B

Benchmark Voodisss (MTEB AskUbuntuDupQuestions, 0.6B) :

| Quant | Taille | Δ NDCG@10 |
|-------|--------|-----------|
| F16 | 1.12 Go | référence |
| Q8_0 | 610 Mo | -0.2% |
| **Q4_K_M** | **379 Mo** | **-0.3%** |
| Q4_0 | 360 Mo | -2.0% |
| Q2_K | 280 Mo | -28.7% |

**Choix : Q4_K_M.** Sweet spot officiel : 3× plus petit que F16, 0.3% de perte. Le 4B est inutilisable sur CPU avec l'agent LLM (trop lent, ~30-40s pour 18 candidats).

### 5.3 Pooling

| Modèle | Pooling | Justification |
|--------|---------|---------------|
| Embedding | `--pooling last` | Blog Qwen3 : "vecteur d'état caché correspondant au dernier token [EOS]" |
| Reranker | `--pooling rank` | Active le classifieur oui/non (`cls.output.weight`). Obligatoire. |

**Note :** Le guide Voodisss indique `pooling = mean` pour l'embedding. La documentation officielle Qwen3 (blog + README) dit explicitement `last` (token [EOS]). La documentation officielle prévaut.

### 5.4 Embedding ctx-size = 8192 — Reranker ctx-size = 1024

Mesure des tokens max par vault (tokenizeur Qwen3 exact) :

| Vault | Tokens max | Couvert par 8192 ? |
|-------|-----------|-------------------|
| void | 9 708 | ❌ (1 fichier aberrant : transcription vidéo) |
| llm | 4 606 | ✅ |
| telephone | 3 463 | ✅ |
| terminal | 3 141 | ✅ |
| browsing | 940 | ✅ |
| images | 522 | ✅ |
| linux | 422 | ✅ |

**Choix : ctx=8192 et ub=8192.** Couvre 99% des chunks sans troncation. Le fichier aberrant (9708 tokens) est proprement tronqué (début préservé). ctx/ub-size n'affecte pas la qualité tant que l'entrée rentre — les slots vides ne sont jamais utilisés.

**Reranker ctx-size = 1024 et ub=1024.** Couvre 100% des cas. Le reranker ne traite qu'**une seule paire** (requête + document) à la fois, soit ~515 tokens max.

### 5.5 Quantification du KV Cache : q8_0

```bash
-ctk q8_0    # quantification des clés (K)
-ctv q8_0    # quantification des valeurs (V)
```

| Quant KV | RAM KV Cache | Perte de Qualité |
|----------|-------------|------------------|
| f16 (défaut) | 56 Ko/token | référence |
| **q8_0** | **28 Ko/token** | **<0.1%** |
| q4_0 | 14 Ko/token | Notable (à éviter) |

**Choix : q8_0.** Moitié de la RAM, qualité quasi identique. Gain de performance gratuit.

### 5.6 RERANK_CANDIDATES = 18

Tests empiriques (CPU M1 Pro, ~580ms/candidat) :

| Candidats | Latence | Reranker | Qualité |
|-----------|---------|----------|---------|
| 5 | 2 655 ms | ✅ | Bonne (mais discrimination limitée) |
| 10 | 6 235 ms | ✅ | Acceptable |
| **18** | **~10 400 ms** | **✅** | **Optimale (max stable)** |
| 21 | ~12 200 ms | ✅ | Gain marginal sur 18 |
| 30 | 20 501 ms | ✅ (avec --timeout 120) | Identique à 21 |
| 50 | 20 497 ms | ❌ crash → bascule RRF | — |

**Choix : 18.** Bon équilibre entre latence et qualité. La littérature recommande 20-50 candidats (pour des rerankers GPU à 4ms/doc). Avec 18 candidats à 580ms/doc sur CPU, on couvre l'équivalent de 50 candidats GPU.

### 5.7 Inclusion du Nom de Fichier dans le Reranker

```python
rerank_docs = [f"[{c['source']}]\n{c['text'][:1400]}" for c in candidate_chunks]
```

Sans le nom de fichier, le reranker ne peut pas faire correspondre une requête qui est littéralement le nom du fichier. Avec le nom de fichier, le cross-encoder voit la correspondance exacte et score 0.9999 au lieu de 0.9895. Testé et validé.

### 5.8 GGUF Voodisss (Obligatoire pour le Reranker)

Les GGUF communautaires Qwen3-Reranker sont **cassés** (llama.cpp #16407). Il leur manque :
- Le tenseur `cls.output.weight` (classifieur oui/non)
- La métadonnée `pooling_type=RANK`
- Le chat template de reranking

Résultat : scores absurdes (`4.5e-23`). Seuls les GGUF **Voodisss** (convertis avec le `convert_hf_to_gguf.py` officiel) fonctionnent.

Le reranker Qwen3 est un **reranker génératif** : le modèle produit des logits, `cls.output.weight` (tenseur `[hidden_dim, 2]`) projette l'état caché final vers P(oui) et P(non), puis softmax → `relevance_score = P(oui)`.

### 5.9 Cache de Prompt Hôte Désactivé (`--cache-ram 0`)

La PR llama.cpp #16391 a introduit le cache de prompt en mémoire hôte avec un **défaut de 8 Gio**. Pour des serveurs d'embedding/reranking où les prompts ne sont **jamais réutilisés**, c'est du gaspillage pur. dvcdsys/code-index a documenté le problème en production : le RSS est passé de 365 Mo à 11.3 Go avant OOM kill. Avec `--cache-ram 0`, il plafonne à ~900 Mo sous la même charge.

### 5.10 Instruction-Aware (gain de 1-5%)

Les deux modèles supportent des instructions personnalisées. Qwen3 recommande :
- Écrire les instructions en **anglais** (même pour un usage multilingue)
- Instruction par défaut : `"Given a web search query, retrieve relevant passages that answer the query"`
- Gain mesuré : 1% à 5% selon les tâches

Actuellement, l'instruction par défaut est utilisée (injectée automatiquement par le chat template du reranker). Personnalisation possible ultérieurement.

---

## 6. Méthodologie (Inspirée de Dave Ebbelaar)

Le pipeline suit l'architecture présentée dans "Hybrid Retrieval from Scratch" (2026) :

1. **BM25** : capture les termes exacts, identifiants, mots rares. Manque les paraphrases.
2. **Embeddings denses** : capture le sens sémantique. Manque les termes exacts.
3. **RRF** : fusionne les deux classements par rang (pas par score). Les scores BM25 et cosinus sont incomparables ; le rang ne l'est pas.
4. **Reranker** : réordonne les candidats en lisant conjointement requête + document. C'est l'étape qui apporte le plus de gain qualitatif (NDCG +12 points dans le benchmark FinanceQA de la vidéo).

**Ce qui N'EST PAS implémenté (par rapport à la vidéo) :**
- Évaluation NDCG@10 avec vérité terrain
- Dataset d'évaluation généré par LLM
- Comparaison systématique de configurations

---

## 7. Installation depuis Zéro

### Prérequis

- llama.cpp compilé (build CPU) : `$LLAMA_CPP_BIN`
- Python 3 avec venv : `~/.venv/main/`
- Paquets Python : `numpy`, `requests`, `rank_bm25`
- Modèles GGUF téléchargés (Voodisss pour le reranker, Qwen officiel pour l'embedding)

### Installation

```bash
# 1. Créer le venv et installer les dépendances
python3 -m venv ~/.venv/main
~/.venv/main/bin/pip install numpy requests rank_bm25

# 2. Copier config.sh.example vers config.sh et adapter les chemins
cp config.sh.example config.sh
# Éditer config.sh avec vos chemins

# 3. Placer les scripts (voir section 2 pour les chemins)
chmod +x llama/*.sh
chmod +x server/*.py
chmod +x fish/*.sh

# 4. Configurer les alias fish (voir config.fish)

# 5. Premier lancement (indexation complète)
llmers
# Attendre la fin de l'indexation (~5-15 min selon la taille des vaults)
# Vérifier :
curl -s http://127.0.0.1:8182/health | jq .
```

### Ajouter un Nouveau Vault

1. Ajouter l'entrée dans `VAULTS_CONFIG` de `rag_server_rerank.py` :
```python
"docs": {
    "path": os.path.join(OBSIDIAN_DIR, "docs_techniques"),
},
```

2. Ajouter le nom du vault à la regex dans `search_vault.sh` :
```bash
if [[ "${1:-}" =~ ^(void|linux|browsing|terminal|llm|images|telephone|docs|obsidian|all)$ ]]; then
```

3. Redémarrer le serveur :
```bash
rsk
rs
```

---

## 8. Guide d'Utilisation Rapide

Voir `doc/english/00 Quick Start Guide.md` pour la version condensée.

### Exemples de recherche

```bash
# Recherche rapide (~20ms)
rag void "configuration nftables"

# Recherche lente et précise avec reranker (~10-18s CPU, ~1s GPU)
ragr void "configuration nftables"

# Rechercher dans un vault spécifique
rag linux "hooks dracut"

# Rechercher dans tous les vaults à la fois
ragr all "votre requête"

# Rechercher dans tous les vaults Obsidian
ragr obsidian "votre requête"
```

### Vérification de santé

```bash
rc
```

### Surveiller les logs

```bash
rst
```

### Arrêter le serveur RAG

```bash
rsk
```

---

## 9. Débogage

### Le Serveur RAG Ne Répond Pas (port 8182)

```bash
ps aux | grep rag_server_rerank
tail -50 $LOG_DIR/rag_server_rerank.log

# Erreur courante : IndentationError après édition
$VENV_PYTHON -c "import py_compile; py_compile.compile('$RAG_SCRIPTS_DIR/rag_server_rerank.py', doraise=True)"
```

### L'Embedding Ne Répond Pas (port 8181)

```bash
ps aux | grep llama-server | grep 8181
tail -20 $LOG_DIR/llm-embed-06b.log

# Erreur courante : le script pointe vers le mauvais modèle
cat $LLAMA_SCRIPTS_DIR/start-llm-embed-qwen3-06b.sh | grep "^\s*-m"
```

### Le Reranker Ne Répond Pas (port 8184)

```bash
ps aux | grep llama-server | grep 8184

# Test direct
curl -s http://127.0.0.1:8184/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"query":"test","documents":["document test"]}' | jq .

# Si scores ~1e-28 → GGUF cassé (mauvaise conversion)
# Si "This server does not support reranking" → flags manquants
# Vérifier les 3 flags obligatoires :
cat $LLAMA_SCRIPTS_DIR/start-llm-reranker-06b.sh | grep -E "reranking|pooling|embedding"
```

### "Connection refused" sur 8181 depuis le RAG

L'embedding ne tourne pas. Le serveur RAG a besoin que 8181 soit actif **avant** de recevoir des requêtes.

```bash
llme    # démarrer l'embedding
# attendre 3-5s
rag void "test"
```

### Scores Reranker Suspects (1e-28)

Le GGUF est mal converti. `cls.output.weight` manquant. Solution : re-télécharger depuis Voodisss.

### Latence Anormale (>20s)

```bash
# Vérifier quel modèle est chargé
ps aux | grep llama-server | grep -v grep

# Si le 4B est chargé au lieu du 0.6B → tuer et redémarrer avec le bon script
pkill -f llama-server
llme    # 0.6B
llmr    # reranker 0.6B
```

### Mesurer les Tokens Max par Vault

```bash
$RAG_SCRIPTS_DIR/test_tokens.sh
```

Nécessite `transformers` installé : `$VENV_PYTHON -m pip install transformers`

---

## 10. Nettoyage

### Désinstallation Complète

```bash
bash uninstall.sh
```

Le désinstallateur arrête les services en cours, supprime les alias shell, et optionnellement supprime le dépôt, les modèles, le cache et le venv. Il ne supprime PAS vos vaults Obsidian ni votre documentation.

Tester sans rien modifier : `bash uninstall.sh --dry-run`

### Supprimer le Cache (Ré-indexation Complète)

```bash
rm -f ~/.rag/*__qwen3-embed-06b*
```

### Supprimer l'Ancien Cache (Changement de Modèle)

```bash
rm -f ~/.rag/*__nomic*
rm -f ~/.rag/*__qwen3-embed-4b*
```

### Tuer Tous les Processus RAG

```bash
pkill -f llama-server
pkill -f rag_server_rerank
```

### Vérifier la Taille du Cache

```bash
du -sh ~/.rag/
```

---

## 11. RAM Estimée avec Agent IA

| Composant | Configuration | RAM |
|-----------|--------------|-----|
| Embedding 0.6B Q8_0 | ctx 8192, ub 8192, KV q8_0, parallel 1 | ~794 Mo |
| Reranker 0.6B Q4_K_M | ctx 1024, ub 1024, KV q8_0, parallel 1 | ~687 Mo |
| Serveur Python (numpy + BM25) | 3218 chunks | ~200 Mo |
| **Total Pile RAG** | | **~1.68 Go** |
| LLM Qwen3.6-35B-A3B Q4_K_XL | | ~20.4 Go |
| Système | | ~5 Go |
| **Total Système** | | **~27 Go / 32 Go** |

---

## 12. Limitations Connues

- **Pas de rechargement à chaud** : modifier un fichier Obsidian nécessite un redémarrage du serveur (`rs`)
- **Latence du reranker** : ~10s sur CPU pour 18 candidats (~580ms/candidat). Structurel (cross-encoder = 1 forward pass par paire)
- **RERANK_CANDIDATES > 21** : le reranker crash (timeout ou saturation mémoire). 21 est le maximum stable.
- **1 fichier aberrant** : "Is Void Linux Good - With Jake from @JakeLinux.md" (9708 tokens) est tronqué à 8192 tokens. Impact négligeable (1 chunk sur 3218).
- **Pas d'évaluation NDCG** : pas de vérité terrain pour mesurer objectivement la qualité
- **Filtre is_embeddable()** : peut rejeter des chunks techniques légitimes (seuil de 25% de lignes de code)
- **Cache orphelin** : les chunks supprimés/modifiés restent dans le JSON (pas de garbage collection)
- **HTTP mono-thread** : le serveur Python (`http.server`) ne gère qu'une requête à la fois
- **ctx-size ne dépend pas du nombre de chunks** : il est déterminé par la longueur max d'un chunk individuel, pas par la taille du corpus

---

## 13. Évolutions Possibles

- [ ] Passer à `http.server.ThreadingHTTPServer` pour les requêtes concurrentes
- [ ] Ajouter un endpoint `POST /reindex` pour re-scanner un vault sans redémarrer
- [ ] Construire un dataset d'évaluation (paires requête→chunk) pour mesurer le NDCG@10
- [ ] Ajouter des instructions personnalisées au reranker (gain de 1-5% selon Qwen)
      Instruction par défaut : "Given a web search query, retrieve relevant passages that answer the query"
      Pour un vault technique FR : "Retrieve relevant technical documentation passages that answer the query about Linux system administration"
      Note : Qwen3 recommande d'écrire les instructions en anglais même pour un usage multilingue
- [ ] Garbage collection du cache (supprimer les entrées orphelines)
- [ ] Support du modèle 4B en mode "batch hors ligne" pour l'indexation de gros vaults
- [ ] Investiguer pourquoi RERANK_CANDIDATES > 21 fait crasher le reranker (mémoire ? timeout interne ?)
- [ ] Tester `--parallel 1` vs `--parallel 2` pour le reranker
- [ ] Explorer le routage de modèles llama.cpp (`--models-preset models.ini`) pour servir embedding + reranker sur un seul port
- [ ] Améliorer le chunking pour les fichiers sans en-têtes Markdown (transcriptions, articles longs)
- [ ] Remplacer la similarité cosinus numpy par FAISS IndexHNSW au-delà de 50 000 chunks

---

## 14. Références

- **Pipeline hybride** : Dave Ebbelaar, "Hybrid Retrieval from Scratch" (2026)
  https://www.youtube.com/watch?v=XvKiTfd6Xvo
- **RRF** : Cormack, Clarke, Buettcher (2009) — "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
- **Qwen3-Embedding** : https://qwenlm.github.io/blog/qwen3-embedding/
  Article : arXiv:2506.05176
- **Qwen3-Reranker HuggingFace** : https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
- **Qwen3-Reranker GGUF (Voodisss)** :
  https://huggingface.co/Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp
- **Guide multi-modèles llama.cpp (Voodisss)** :
  https://gist.github.com/VooDisss/42bce4eb5c76d3c325633886c5e348ee
- **Issue llama.cpp #16407** (GGUF reranker cassés) :
  https://github.com/ggml-org/llama.cpp/issues/16407
- **PR llama.cpp #16391** (cache de prompt en mémoire hôte) :
  https://github.com/ggml-org/llama.cpp/pull/16391
- **llama.cpp** : https://github.com/ggml-org/llama.cpp

---

## 15. FAQ

**Q : Pourquoi pas une base de données vectorielle (Qdrant, ChromaDB, etc.) ?**
R : Avec ~3000 chunks, un tableau numpy en mémoire suffit. La recherche cosinus sur 3000 vecteurs de 1024d prend <1ms. Une base vectorielle ajoute de la complexité sans gain mesurable à cette échelle. Au-delà de ~100 000 chunks, reconsidérer (FAISS IndexHNSW ou Qdrant).

**Q : Pourquoi BM25 en plus de la recherche vectorielle ?**
R : La recherche vectorielle manque les termes exacts (noms de commandes, chemins, identifiants). BM25 les capture. Sur des vaults techniques avec beaucoup de commandes (`nftables`, `dracut`, `sfdisk`), BM25 est souvent plus fiable que la recherche vectorielle seule.

**Q : Pourquoi le reranker est-il si lent sur CPU ?**
R : Le cross-encoder fait un forward pass COMPLET du modèle pour CHAQUE paire (requête, document). Contrairement au bi-encoder qui encode une fois et compare les vecteurs, le cross-encoder retraite tout depuis le début. Sur CPU sans GPU, un forward pass 0.6B prend ~580ms. C'est structurel.

**Q : Peut-on utiliser le RAG sans reranker ?**
R : Oui. `rag` (alias) force `--no-rerank`. Le serveur fonctionne aussi si le reranker ne tourne pas (bascule RRF automatique). Latence : ~20ms.

**Q : Peut-on utiliser le RAG sans embedding ?**
R : Non. L'embedding est requis pour la recherche vectorielle. Sans le serveur d'embedding (port 8181), le serveur RAG retourne une erreur 500.

**Q : Comment ajouter un vault sans tout ré-indexer ?**
R : Ajouter l'entrée dans VAULTS_CONFIG, redémarrer le serveur. Les vaults existants sont chargés depuis le cache (instantané). Seul le nouveau vault est indexé depuis zéro.

**Q : Le cache est-il compatible entre modèles ?**
R : Non. Le nom de fichier inclut l'identifiant du modèle (`void_cache__qwen3-embed-06b.json`). Changer de modèle crée un nouveau fichier de cache.

**Q : Pourquoi tous les scores du reranker sont >0.99 ?**
R : Le reranker Qwen3 utilise un classifieur oui/non avec softmax. Sur des documents clairement pertinents, P(oui) → 1.0. La discrimination se fait sur les documents marginaux (scores 0.3-0.8).

**Q : Est-ce que ctx-size dépend du nombre de chunks dans le vault ?**
R : Non. ctx-size définit la longueur max d'une seule entrée. Que le vault contienne 3 000 ou 300 000 chunks, l'embedding reçoit toujours un chunk à la fois (~910 tokens max) et le reranker toujours une paire requête+document (~500 tokens max). ctx-size est déterminé par la longueur du plus grand chunk individuel, pas par la taille du corpus.

**Q : Pourquoi `--pooling last` et pas `--pooling mean` pour l'embedding ?**
R : Le blog officiel Qwen3 dit : "Le modèle d'Embedding traite un seul segment de texte en entrée, extrayant la représentation sémantique en utilisant le vecteur d'état caché correspondant au dernier token [EOS]." C'est `last`. Le guide Voodisss indique `mean`, mais la documentation officielle Qwen3 prévaut.

---

## 16. Glossaire

| Terme | Définition |
|-------|-----------|
| **Bi-encoder** | Encode requête et document séparément → vecteurs → similarité cosinus. Rapide mais perd les nuances. |
| **Cross-encoder** | Encode requête + document conjointement → score de pertinence. Lent mais précis. |
| **Reranker génératif** | Le reranker Qwen3 utilise un classifieur (cls.output.weight) qui projette l'état caché vers P(oui)/P(non). Pas un cross-encoder traditionnel. |
| **RRF** | Reciprocal Rank Fusion. Fusionne les classements par rang, pas par score. Formule : 1/(k+rang). |
| **BM25** | Algorithme de recherche par mots-clés avec pondération TF-IDF. Capture les termes exacts. |
| **Pooling last** | Le vecteur d'embedding est l'état caché du dernier token [EOS]. |
| **Pooling rank** | Mode classifieur pour le reranker. Extrait les logits oui/non via cls.output.weight. |
| **KV cache** | Mémoire d'attention (clés + valeurs) allouée par token de contexte. Quantifiable via -ctk/-ctv. |
| **Chunk** | Fragment de document issu du découpage Markdown. Unité d'indexation. |
| **Vault** | Répertoire Obsidian indexé comme collection distincte. |
| **GGUF** | Format de fichier pour modèles quantifiés utilisé par llama.cpp. |
| **Quantification** | Réduction de la précision des poids (F16→Q8→Q4) pour réduire la RAM. |
| **MTEB** | Massive Text Embedding Benchmark. Classement de référence pour les modèles d'embedding. |
| **NDCG@K** | Normalized Discounted Cumulative Gain. Métrique de qualité de récupération (0-1). |
| **MRL** | Matryoshka Representation Learning. Permet des dimensions d'embedding flexibles (32 à 2560 pour Qwen3-4B). |
| **Instruction-aware** | Capacité du modèle à adapter son comportement selon une instruction personnalisée (gain de 1-5%). |

---

## Système Recommandé

Void Linux avec niri desktop.
Faster boot, occupe seulement 1Go RAM, gestionnaire de packages rapide et complet.
Pas de system.d mais runit qui est bien plus léger, rapide et confidentiel. 
Excellent équilibre entre sécurité et fluidité.

## Licence                                                                                                                                                                          

[MIT](LICENSE) 
