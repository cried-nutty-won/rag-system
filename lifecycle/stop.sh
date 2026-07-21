#!/usr/bin/env fish

echo "Arrêt de llama-server"

# Arrêt des deux LLMs
pkill -U $USER -f llama-server
pkill -f rag_server.py
sleep 1
pkill -U $USER -f llama-server
pkill -f rag_server.py
sleep 1
sudo pkill -9 llama-server > /dev/null 2>&1
sudo pkill -9 rag_server.py > /dev/null 2>&1
sudo pkill -9 rag_server_rerank.py > /dev/null 2>&1

# Libération profonde de la mémoire
sync
echo 3 | doas tee /proc/sys/vm/drop_caches > /dev/null
echo 1 | doas tee /proc/sys/vm/compact_memory > /dev/null
doas swapoff -a
doas swapon -a

echo "Tout est arrêté, mémoire purgée."
