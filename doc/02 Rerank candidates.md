# Tes données

| RERANK_CANDIDATES | Latence | Reranker | ~ms/candidat |
|---|---|---|---|
| 5 | 2 655 ms | ✅ | 531 |
| 10 | 6 235 ms | ✅ | 624 |
| 20 | 11 625 ms | ✅ | 581 |
| 30 | 16 258 ms | ❌ fallback RRF | — |
| 40 | 16 262 ms | ❌ fallback RRF | — |
| 50 | 16 293 ms | ❌ fallback RRF | — |

**~580 ms par candidat** en CPU. Et au-delà de 20, le reranker **échoue** (probablement le `--ctx-size 4096` du reranker qui sature avec 30×400 tokens, ou le `--parallel 2` qui ne peut pas absorber la charge). Les 16s sont le temps avant que la connexion lâche et que le fallback RRF prenne le relais.

---

# Ce que dit la littérature

Les recommandations standard sont calibrées pour des rerankers **GPU** (Cohere, Jina) à ~4 ms/document :

| Source | Recommandation |
|---|---|
| "Common values: **50-200**. More candidates = more compute but more chance to recover the true top-N" [[15]] | 50-200 candidats |
| "Retrieve **50-100**, rerank to 10. Best accuracy at Retrieve=100, Rerank=10 (MRR 0.695, 125ms)" [[11]] | 50-100 candidats |
| "K=20-50 is often optimal. Beyond K=100, quality gains are minimal" [[18]] | 20-50, max 100 |
| "For a 200ms budget with 4ms/doc → ~50 candidates" [[18]] | Budget ÷ coût/doc |

Tableau de référence (GPU) [[11]] :

| Retrieve | Rerank → top | MRR@10 | Latence | Verdict |
|---|---|---|---|---|
| 20 | 5 | 0.612 | 80ms | ❌ Trop peu |
| 50 | 10 | 0.683 | 105ms | ✅ Bon |
| 100 | 10 | 0.695 | 125ms | ✅ Meilleur |
| 200 | 10 | 0.698 | 180ms | ❌ Rendements décroissants |

---

# Adaptation à ton setup (CPU, 580ms/doc)

La formule est : **budget latence ÷ coût par document = nombre de candidats**.

| Budget latence | Candidats possibles | Équivalent GPU |
|---|---|---|
| 3s | 5 | — |
| 6s | 10 | — |
| 12s | **20** | ≈ 50 candidats GPU |
| 29s | 50 | ≈ 200 candidats GPU |

Avec 20 candidats à 580ms/doc, tu couvres l'équivalent de ce qu'un reranker GPU ferait avec 50 candidats en 200ms. Tu es dans la plage recommandée (20-50).

---

# Recommandation pour ton système

```python
RERANK_CANDIDATES = 20
```

C'est le **maximum fonctionnel** (au-delà, le reranker crash) et c'est dans la plage optimale de la littérature. Tes tests le confirment :

- **20 candidats** : le bon fichier en position 1, score 0.9999, le meilleur chunk sélectionné ("Procédure d'installation...")
- **10 candidats** : le bon fichier tombe en position 3, score 0.9994, un moins bon chunk sélectionné ("Au menu GRUB...")
- **5 candidats** : rapide (2,6s) mais le reranker a trop peu d'options pour discriminer finement

---

# Pourquoi 30+ échoue

Le reranker a `--ctx-size 4096`. Avec 30 documents de ~400 tokens + le chat template (~100 tokens par paire), la requête totale dépasse probablement ce que le serveur peut traiter en une seule passe. Le `--parallel 2` n'aide pas car il sert à gérer des requêtes HTTP concurrentes, pas à découper une grosse requête.

Si tu veux un jour pousser au-delà de 20, il faudrait augmenter le ctx-size du reranker :

```bash
--ctx-size 8192   # au lieu de 4096
```

Mais ça double la RAM du KV cache du reranker. Sur 32 Go avec le LLM agent, c'est probablement trop.

---

# Résumé des tests

| Candidats | Latence | Reranker | ms/candidat |
|---|---|---|---|
| 20 | 11 456 ms | ✅ | 573 |
| 30 | 20 501 ms | ✅ | 683 |
| 50 | 20 497 ms | ❌ fallback RRF | — |

Le `--timeout 120` a **débloqué 30 candidats** (qui échouait avant). Mais 50 échoue toujours — le reranker sature probablement en mémoire ou la connexion HTTP lâche avant la fin (~29s de traitement estimé).

**Le maximum fonctionnel est 30 candidats.**

---

# Qualité : 20 vs 30

Pour cette requête, les résultats sont **identiques** : même top 3, mêmes scores (0.9999, 0.9998, 0.9995). Les 10 candidats supplémentaires (rang 21-30 du RRF) n'ont rien apporté.

C'est cohérent avec la littérature : au-delà de 20-50 candidats, les gains sont marginaux. Les documents classés 21ᵉ-30ᵉ par le RRF ont une probabilité très faible d'être le meilleur résultat.

---

# Recommandation finale

```python
RERANK_CANDIDATES = 20
```

- **11,5s** au lieu de 20,5s (presque 2× plus rapide)
- Qualité identique sur tes requêtes
- Marge de sécurité (30 est à la limite, 50 crash)
- Cohérent avec la littérature (20-50 est la plage optimale, 20 est le minimum recommandé)

Si un jour tu ajoutes beaucoup de vaults et que le corpus grossit significativement, tu pourras re-tester 30. Mais pour 3146 chunks, 20 suffit.
