#!/usr/bin/env fish
set REPO_DIR (status dirname | string replace '/fish' '')
exec "$REPO_DIR/llama/start-rag-llm_embed_server.sh" &
