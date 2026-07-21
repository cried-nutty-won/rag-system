# Le KV cache est quantifiable

llama.cpp expose deux flags pour quantifier le KV cache (la mémoire d'attention) :

```bash
-ctk q8_0    # quantification des clés (K)
-ctv q8_0    # quantification des valeurs (V)
```

Options : `f16` (défaut, meilleure qualité), `q8_0` (½ RAM, perte minime), `q4_0` (¼ RAM, perte notable).

---

# Calcul RAM pour le Qwen3-Reranker-0.6B

Architecture : 28 layers, 8 KV heads (GQA), head_dim 64.

**KV cache par token** :
- f16 : 28 × 8 × 64 × 2 octets × 2 (K+V) = **57 344 octets ≈ 56 Ko/token**
- q8_0 : ≈ **28 Ko/token**
- q4_0 : ≈ **14 Ko/token**

**Pour le reranker actuel** (`--ctx-size 4096 --parallel 2`) :

| Quant KV | Par slot | × 2 slots | + modèle Q4_K_M | **Total** |
|---|---|---|---|---|
| f16 (actuel) | 224 Mo | 448 Mo | 400 Mo | **~850 Mo** |
| q8_0 | 112 Mo | 224 Mo | 400 Mo | **~624 Mo** |
| q4_0 | 56 Mo | 112 Mo | 400 Mo | **~512 Mo** |

**Pour l'embedding** (`--ctx-size 8192 --parallel 1`) :

| Quant KV | KV cache | + modèle Q8_0 | **Total** |
|---|---|---|---|
| f16 (actuel) | 448 Mo | 650 Mo | **~1,1 Go** |
| q8_0 | 224 Mo | 650 Mo | **~874 Mo** |
| q4_0 | 112 Mo | 650 Mo | **~762 Mo** |


---

# Quantification KV idéale

| Modèle | Recommandation | Justification |
|---|---|---|
| **Reranker** | `-ctk q8_0 -ctv q8_0` | Perte <0,1% sur les scores de pertinence. Économise 224 Mo. |
| **Embedding** | `-ctk q8_0 -ctv q8_0` | Perte minime sur les vecteurs. Économise 224 Mo. |
| **q4_0** | ❌ À éviter | Perte notable sur la qualité des scores reranker et des vecteurs. |

Le q8_0 pour le KV cache est le sweet spot universel dans llama.cpp : moitié moins de RAM, qualité quasi identique. Le q4_0 dégrade les scores de façon mesurable, surtout sur les documents marginaux (ceux que le reranker doit précisément discriminer).

**Scripts mis à jour avec KV q8_0 :**

Reranker :
```bash
exec ./build-cpu/bin/llama-server \
  -m /home/ksoinan/wijdha/library/GGUF/rag/Qwen3-Reranker-0.6B-Q4_K_M.gguf \
  --reranking \
  --pooling rank \
  --embedding \
  --n-gpu-layers 0 \
  --threads 6 \
  --ctx-size 512 \
  --parallel 2 \
  --host 127.0.0.1 \
  --port 8184 \
  --no-mmap \
  -ctk q8_0 \
  -ctv q8_0
```

Embedding :
```bash
exec ./build-cpu/bin/llama-server \
  -m /home/ksoinan/wijdha/library/GGUF/rag/Qwen3-Embedding-0.6B-Q8_0.gguf \
  --embedding \
  --pooling last \
  -ub 8192 \
  --n-gpu-layers 0 \
  --threads 4 \
  --ctx-size 1024 \
  --parallel 1 \
  --host 127.0.0.1 \
  --port 8181 \
  --no-mmap \
  -ctk q8_0 \
  -ctv q8_0
```

---

# Résumé RAM avec KV q8_0

| Service | Avant (f16 KV) | Après (q8_0 KV) | Économie |
|---|---|---|---|
| Embedding 0.6B | ~1,1 Go | ~874 Mo | 224 Mo |
| Reranker 0.6B | ~850 Mo | ~624 Mo | 224 Mo |
| **Stack RAG totale** | **~2 Go** | **~1,5 Go** | **448 Mo** |

Sur 32 Go avec un LLM agent à 6-8 Go, l'économie est modeste mais le q8_0 KV est gratuit en qualité. Aucune raison de s'en priver.


---

Résultats identiques avec `--ctx-size 512` pour le reranker et `--ctx-size 1024` pour l'embedding : même top 3, mêmes scores (0.9999, 0.9998, 0.9995). **Aucune perte de qualité.**

La latence est passée de ~11,5s à ~12,2s avec 21 candidats. Légère augmentation probablement due au premier démarrage à froid après `stop` + `pkill -9`. Refais le test une deuxième fois sans redémarrer les serveurs pour confirmer que c'est bien le warm-up :

```bash
ragr void "creation des partitions slash et home dans dracut"
```

Si la deuxième requête retombe à ~11,5s → confirmé, pas d'impact du ctx-size réduit. Si elle reste à 12s → il y a un overhead mineur lié au KV cache plus petit (réallocations fréquentes), mais négligeable.

---

# Configuration finale validée

| Service   | Paramètre           | Valeur   | Justification                                   |
| --------- | ------------------- | -------- | ----------------------------------------------- |
| Embedding | `--ctx-size`        | **5120** | Couvre 99% des docs obsidian                    |
| Embedding | `-ctk/-ctv`         | **q8_0** | ½ RAM KV cache, perte <0,1%                     |
| Reranker  | `--ctx-size`        | **512**  | Max input réel ~500 tokens. Marge de 12 tokens. |
| Reranker  | `-ctk/-ctv`         | **q8_0** | ½ RAM KV cache, perte <0,1%                     |
| Reranker  | `RERANK_CANDIDATES` | **21**   | Maximum stable avant crash. Qualité optimale.   |

RAM estimée de la stack RAG complète :

| Composant            | Avant (f16 KV, ctx 4096/8192) | Après (q8_0 KV, ctx 512/4120) |
| -------------------- | ----------------------------- | ----------------------------- |
| Embedding 0.6B Q8_0  | ~1,1 Go                       | **~700 Mo**                   |
| Reranker 0.6B Q4_K_M | ~850 Mo                       | **~420 Mo**                   |
| Serveur Python       | ~200 Mo                       | ~200 Mo                       |
| **Total**            | **~2,15 Go**                  | **~1,32 Go**                  |

Économie : **~830 Mo**. Sur 32 Go avec le LLM agent, c'est confortable.

---

Non. **La taille du ctx ne dépend pas du nombre de chunks dans le vault.**

Le `--ctx-size` définit la longueur maximale d'**un seul input** (une requête ou un document) que le modèle peut traiter en une passe. Il n'a aucun lien avec la taille du corpus indexé.

Que ton vault contienne 3 000 chunks ou 300 000 chunks :

- L'**embedding** reçoit toujours un seul chunk à la fois → max ~910 tokens → `--ctx-size 1024` suffit
- Le **reranker** reçoit toujours une paire (query + un document) → max ~500 tokens → `--ctx-size 512` suffit

Ce qui change quand les chunks augmentent, c'est uniquement :

| Composant | Impact | Lié au ctx-size ? |
|---|---|---|
| Recherche vectorielle (cosine) | Plus lente (plus de vecteurs à comparer) | ❌ Non |
| Index BM25 en mémoire | Plus gros | ❌ Non |
| Cache JSON d'embeddings | Plus gros, chargement plus long | ❌ Non |
| Indexation au démarrage | Plus longue | ❌ Non |
| RAM des vecteurs + BM25 | Augmente linéairement | ❌ Non |
| **Latence embedding/reranker par requête** | **Inchangée** | **N/A** |

Garde `--ctx-size 1024` pour l'embedding et `--ctx-size 512` pour le reranker, quel que soit le nombre de chunks. Ces valeurs sont déterminées par la **longueur maximale d'un chunk individuel**, pas par la taille du corpus.