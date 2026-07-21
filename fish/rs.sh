#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
source "${REPO_DIR}/config.sh"
"${VENV_PYTHON}" -u "${RAG_SCRIPTS_DIR}/rag_server_rerank.py" > "${LOG_DIR}/rag_server_rerank.log" 2>&1 &
