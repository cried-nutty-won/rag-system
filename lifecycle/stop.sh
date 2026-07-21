#!/bin/bash
# Arrêt complet de la stack RAG + purge mémoire

echo "Arrêt de llama-server..."

pkill -U "$USER" -f llama-server 2>/dev/null
pkill -U "$USER" -f rag_server_rerank.py 2>/dev/null
sleep 1

pkill -9 -U "$USER" -f llama-server 2>/dev/null
pkill -9 -U "$USER" -f rag_server_rerank.py 2>/dev/null

if command -v doas &>/dev/null; then
    PRIV=doas
elif command -v sudo &>/dev/null; then
    PRIV=sudo
else
    echo "⚠️  Ni doas ni sudo trouvé. Skip purge mémoire."
    echo "Tout est arrêté."
    exit 0
fi

sync
echo 3 | $PRIV tee /proc/sys/vm/drop_caches > /dev/null 2>&1
echo 1 | $PRIV tee /proc/sys/vm/compact_memory > /dev/null 2>&1
$PRIV swapoff -a 2>/dev/null
$PRIV swapon -a 2>/dev/null

echo "Tout est arrêté, mémoire purgée."
