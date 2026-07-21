#!/usr/bin/env fish
set REPO_DIR (status dirname | string replace '/fish' '')
exec "$REPO_DIR/llama/start-llm-embed-qwen3-06b.sh" &
