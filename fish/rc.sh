#!/usr/bin/env bash
curl -s http://127.0.0.1:8181/health | jq .   # embedding
curl -s http://127.0.0.1:8184/health | jq .   # reranker
curl -s http://127.0.0.1:8182/health | jq .   # rag
