#!/usr/bin/env fish

# arrêt des serveurs et llama.cpp
stop

# Nettoyage préliminaire de la mémoire RAM et du Swap
sync
echo "Purge des caches et réinitialisation du Swap..."
echo 3 | doas tee /proc/sys/vm/drop_caches > /dev/null
doas swapoff -a
doas swapon -a
pkill -f zen
sleep 1
rm -rf ~/.cache/zen/

echo "Lancement du Qwen & rag..."
/home/ksoinan/scripts/llm/rag/start-rag-llm_embed_reranker_server.sh &
/home/ksoinan/scripts/llm/start-llm-qwen-q4-xl-for-omp.sh > /tmp/qwen_server.log 2>&1 &

# Relancer zen avec une redirection compatible et isolée
sleep 7
/home/ksoinan/.local/bin/zen >/dev/null 2>&1 &
disown

echo "Le service s'exécute en arrière-plan."
