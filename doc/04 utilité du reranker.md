C'est exactement le rôle du reranker. Ce n'est pas un bug — c'est la différence fondamentale entre les deux modes.

# Pourquoi `rag` (RRF) est moins pertinent

La query `"ram estimée"` ne fait que **2 tokens**. Avec si peu d'information :

- **BM25** matche tous les documents contenant "ram" ou "estimée" → les chunks de `start-llm-minicpm-for-omp.sh.md` qui contiennent "RAM" 4 fois remontent en tête
- **Vectoriel** encode "ram estimée" en un vecteur peu discriminant (2 mots) → beaucoup de documents ont une similarité moyenne
- **RRF** fusionne les deux → les faux positifs BM25 polluent le classement

Le reranker, lui, **lit** chaque paire (query, document) et **comprend** que "ram estimée" signifie "estimation de la consommation RAM" → il classe les vrais tableaux de budget RAM en tête et les mentions incidentes de "RAM" en bas.

# Peut-on faire aussi bien sans reranker ?

**Non.** C'est la raison d'être du reranker. Sans lui, tu as un classement par mots-clés + similarité vectorielle brute. Avec lui, tu as un jugement sémantique profond. L'écart de qualité sur les queries courtes ou ambiguës est structurel.

# Ce qui améliore `rag` (RRF seul) sans reranker

La seule variable sur laquelle tu as prise : **la longueur de la query**.

```bash
# Mauvais (2 tokens, trop ambigu) :
rag all "ram estimée"

# Meilleur (plus de tokens, BM25 discrimine mieux) :
rag all "ram estimée stack RAG embedding reranker tableau consommation mémoire"
```

Plus la query est longue et spécifique, plus BM25 et le vectoriel sont discriminants, moins le reranker est nécessaire. Mais pour les queries courtes ou naturelles, le reranker est irremplaçable.

# Résumé

| | `rag` (RRF) | `ragr` (reranker) |
|---|---|---|
| Latence | 41 ms | 17 637 ms |
| Qualité sur query courte | ❌ Faux positifs BM25 | ✅ Jugement sémantique |
| Qualité sur query longue/spécifique | ✅ Correct | ✅ Excellent |
| Usage | Recherche rapide, mots-clés connus | Query naturelle, besoin de précision |

Les deux modes sont complémentaires. `rag` pour la vitesse, `ragr` pour la qualité. On ne peut pas avoir les deux simultanément en CPU.