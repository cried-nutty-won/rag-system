`RAG — Guide d'utilisation rapide`

# Démarrage

| Commande | Action                                              |
| -------- | --------------------------------------------------- |
| llmers   | Stack complète (embedding + reranker + serveur RAG) |
| llmes    | Embedding + serveur RAG (sans reranker)             |
| llme     | Embedding seul (port 8181)                          |
| llmr     | Reranker seul (port 8184)                           |
| rs       | Serveur RAG Python seul (port 8182)                 |
| rst      | tail -f des logs serveur RAG                        |
| rc       | rag curl des 3 services                             |
| rsk      | Tue le serveur RAG Python uniquement                |
| stop     | Tue tous les llama-server                           |

# Recherche

| Commande                          | Mode                         | Latence  |
| --------------------------------- | ---------------------------- | -------- |
| `rag void "ma question"`          | RRF seul (rapide)            | ~20 ms   |
| `ragr void "ma question"`         | RRF + Reranker (précis)      | ~10-20 s |
| `rag all "ma question" 5`         | Tous vaults, top 5           | ~130 ms  |
| `rag obsidian "ma question"`      | Vaults Obsidian uniquement   | ~130 ms  |
| `ragr linux "config nftables" 10` | Vault linux, top 10, reranké | ~10-20 s |
|                                   |                              |          |

# Vaults disponibles

`void` · `linux` · `browsing` · `terminal` · `llm` · `images` · `telephone` · `obsidian` · `all`

- `obsidian` : tous les vaults Obsidian (sans les vaults externes)
- `all` : tous les vaults (Obsidian + externes)

# Vérification

| Commande                                       | Action                                                 |
| ---------------------------------------------- | ------------------------------------------------------ |
| `rc`                                           | Health check des 3 services (embedding, reranker, RAG) |
| `curl -s http://127.0.0.1:8182/health \| jq .` | Détails complets                                       |

# Test reranker isolé

```bash
curl -s http://127.0.0.1:8184/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"query":"nftables","documents":["nftables filtre réseau linux","il fait beau"]}' | jq .
```

# Arrêt

| Commande | Action                                    |
| -------- | ----------------------------------------- |
| `stop`   | Tue tous les llama-server + purge mémoire |
| `rsk`    | Tue le serveur RAG Python uniquement      |

# Après modification d'un fichier Obsidian

```bash
rsk
rs
```

Le cache détecte automatiquement les chunks modifiés. Seuls les nouveaux chunks sont ré-embeddés.

# Après changement de modèle d'embedding

```bash
rm -f ~/.rag/*__qwen3-embed-06b*
rsk
rs
```

# Logs

| Commande                         | Action                         |
| -------------------------------- | ------------------------------ |
| `rst`                            | `tail -f` des logs serveur RAG |
| `tail -f /tmp/llm-embed-06b.log` | Logs embedding                 |

# Paramètres actuels

| Paramètre | Valeur |
|-----------|--------|
| RERANK_CANDIDATES | 21 |
| top_k par défaut | 7 |
| ctx-size embedding | 8192 |
| ctx-size reranker | 1024 |
| KV cache | q8_0 (les deux) |
| batch size (-ub) | 8192 (embed) / 1024 (reranker) |

