#!/usr/bin/env bash
# rag-system — Uninstaller
# Removes aliases, stops services, and optionally removes all data
# Usage: bash uninstall.sh [--dry-run]

set -euo pipefail

# ── Colors ──────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }
header()  { echo -e "\n${BOLD}${CYAN}━━━ $* ━━━${NC}\n"; }

# ── Dry-run mode ────────────────────────────────────────────
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run|-n) DRY_RUN=true ;;
    esac
done

run() {
    if [[ "$DRY_RUN" == true ]]; then
        echo -e "  ${YELLOW}[DRY-RUN]${NC} $*"
    else
        "$@"
    fi
}

ask_yes_no() {
    local prompt="$1" default="$2" var_name="$3"
    echo -e "${BOLD}${prompt}${NC} ${CYAN}[${default}]${NC}"
    read -rp "  > " answer
    answer="${answer:-$default}"
    if [[ "$answer" =~ ^[Yy] ]]; then
        eval "$var_name=true"
    else
        eval "$var_name=false"
    fi
}

# ── Shell detection ─────────────────────────────────────────
detect_shell() {
    local shell_name
    shell_name="$(basename "${SHELL:-/bin/bash}")"
    case "$shell_name" in
        fish)
            SHELL_NAME="fish"
            SHELL_CONFIG="${HOME}/.config/fish/config.fish"
            ;;
        zsh)
            SHELL_NAME="zsh"
            SHELL_CONFIG="${HOME}/.zshrc"
            ;;
        bash)
            SHELL_NAME="bash"
            if [[ -f "${HOME}/.bashrc" ]]; then
                SHELL_CONFIG="${HOME}/.bashrc"
            elif [[ -f "${HOME}/.bash_profile" ]]; then
                SHELL_CONFIG="${HOME}/.bash_profile"
            else
                SHELL_CONFIG="${HOME}/.bashrc"
            fi
            ;;
        *)
            SHELL_NAME="$shell_name"
            SHELL_CONFIG="${HOME}/.${shell_name}rc"
            ;;
    esac
}

# ── Banner ──────────────────────────────────────────────────
clear
echo -e "${BOLD}${CYAN}"
echo "  ╔══════════════════════════════════════════════════╗"
echo "  ║          rag-system — Uninstaller               ║"
echo "  ╚══════════════════════════════════════════════════╝"
echo -e "${NC}"
echo -e "  This script will:"
echo -e "    • Stop running RAG services"
echo -e "    • Remove shell aliases (10 shortcuts)"
echo -e "    • Optionally remove: repo, models, cache, venv"
echo ""
echo -e "  ${YELLOW}It will NOT remove:${NC}"
echo -e "    • Your Obsidian vaults or documentation"
echo -e "    • openfox-rag (separate repo)"
echo ""

detect_shell
info "Shell: ${SHELL_NAME} → ${SHELL_CONFIG}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load config if present
RAG_DIR="$SCRIPT_DIR"
GGUF_DIR=""
VENV_DIR=""
if [[ -f "${RAG_DIR}/config.sh" ]]; then
    source "${RAG_DIR}/config.sh" 2>/dev/null || true
fi
GGUF_DIR="${GGUF_DIR:-$HOME/models/GGUF/rag}"
VENV_DIR="${VENV_DIR:-$HOME/.venv/main}"
CACHE_DIR="${HOME}/.rag"

# ── Step 1: Stop running services ──────────────────────────
header "Step 1/5: Stop running services"

# Check for running processes
RAG_RUNNING=false
EMBED_RUNNING=false
RERANK_RUNNING=false

if pgrep -f "rag_server_rerank" &>/dev/null; then
    RAG_RUNNING=true
fi
if pgrep -f "Qwen3-Embedding" &>/dev/null; then
    EMBED_RUNNING=true
fi
if pgrep -f "Qwen3-Reranker" &>/dev/null; then
    RERANK_RUNNING=true
fi

if [[ "$RAG_RUNNING" == true || "$EMBED_RUNNING" == true || "$RERANK_RUNNING" == true ]]; then
    info "Running services detected:"
    [[ "$RAG_RUNNING" == true ]] && echo -e "  ${YELLOW}•${NC} RAG server (port 8182)"
    [[ "$EMBED_RUNNING" == true ]] && echo -e "  ${YELLOW}•${NC} Embedding server (port 8181)"
    [[ "$RERANK_RUNNING" == true ]] && echo -e "  ${YELLOW}•${NC} Reranker server (port 8184)"
    echo ""
    ask_yes_no "Stop all RAG services?" "y" STOP_SERVICES
    if [[ "$STOP_SERVICES" == true ]]; then
        run pkill -f "rag_server_rerank" 2>/dev/null || true
        run pkill -f "Qwen3-Embedding" 2>/dev/null || true
        run pkill -f "Qwen3-Reranker" 2>/dev/null || true
        # Also kill by port if pkill didn't work
        run pkill -f "llama-server.*8181" 2>/dev/null || true
        run pkill -f "llama-server.*8184" 2>/dev/null || true
        success "Services stopped"
    else
        warn "Services still running — stop them manually before removing files"
    fi
else
    success "No RAG services running"
fi

# ── Step 2: Remove shell aliases ───────────────────────────
header "Step 2/5: Remove shell aliases"

if grep -q "rag-system aliases" "$SHELL_CONFIG" 2>/dev/null; then
    ask_yes_no "Remove rag-system aliases from ${SHELL_CONFIG}?" "y" REMOVE_ALIASES
    if [[ "$REMOVE_ALIASES" == true ]]; then
        if [[ "$DRY_RUN" == true ]]; then
            echo -e "  ${YELLOW}[DRY-RUN]${NC} Would remove alias block from ${SHELL_CONFIG}"
        else
            sed -i '/# ── rag-system aliases ──/,/# ── end rag-system aliases ──/d' "$SHELL_CONFIG"
            sed -i '/^$/N;/^\n$/d' "$SHELL_CONFIG"
        fi
        success "Aliases removed from ${SHELL_CONFIG}"
    else
        info "Aliases kept"
    fi
else
    info "No rag-system aliases found in ${SHELL_CONFIG}"
fi

# ── Step 3: Remove embedding cache ─────────────────────────
header "Step 3/5: Remove embedding cache"

if [[ -d "$CACHE_DIR" ]]; then
    CACHE_SIZE=$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)
    info "Cache directory: ${CACHE_DIR} (${CACHE_SIZE})"
    ask_yes_no "Remove embedding cache?" "y" REMOVE_CACHE
    if [[ "$REMOVE_CACHE" == true ]]; then
        run rm -rf "$CACHE_DIR"
        success "Cache removed"
    else
        info "Cache kept"
    fi
else
    info "No cache directory found at ${CACHE_DIR}"
fi

# ── Step 4: Remove models ──────────────────────────────────
header "Step 4/5: Remove GGUF models"

if [[ -d "$GGUF_DIR" ]]; then
    MODEL_SIZE=$(du -sh "$GGUF_DIR" 2>/dev/null | cut -f1)
    info "Models directory: ${GGUF_DIR} (${MODEL_SIZE})"
    ls -lh "$GGUF_DIR"/*.gguf 2>/dev/null | while read -r line; do
        echo -e "  ${CYAN}•${NC} $(echo "$line" | awk '{print $NF, $5}')"
    done
    echo ""
    ask_yes_no "Remove GGUF models?" "n" REMOVE_MODELS
    if [[ "$REMOVE_MODELS" == true ]]; then
        run rm -rf "$GGUF_DIR"
        success "Models removed"
    else
        info "Models kept"
    fi
else
    info "No models directory found at ${GGUF_DIR}"
fi

# ── Step 5: Remove repo and venv ───────────────────────────
header "Step 5/5: Remove repo and venv (optional)"

ask_yes_no "Remove the rag-system repo (${RAG_DIR})?" "n" REMOVE_REPO
if [[ "$REMOVE_REPO" == true ]]; then
    run rm -rf "$RAG_DIR"
    success "Repo removed"
else
    info "Repo kept"
fi

if [[ -d "$VENV_DIR" ]]; then
    ask_yes_no "Remove Python venv (${VENV_DIR})?" "n" REMOVE_VENV
    if [[ "$REMOVE_VENV" == true ]]; then
        run rm -rf "$VENV_DIR"
        success "Venv removed"
    else
        info "Venv kept"
    fi
else
    info "No venv found at ${VENV_DIR}"
fi

# ── Summary ─────────────────────────────────────────────────
header "Uninstall complete"

echo -e "${BOLD}What was removed:${NC}"
if [[ "${STOP_SERVICES:-false}" == true ]]; then
    echo -e "  ${GREEN}✓${NC} RAG services (stopped)"
fi
if [[ "${REMOVE_ALIASES:-false}" == true ]]; then
    echo -e "  ${GREEN}✓${NC} Shell aliases (${SHELL_CONFIG})"
else
    echo -e "  ${YELLOW}–${NC} Shell aliases (kept)"
fi
if [[ "${REMOVE_CACHE:-false}" == true ]]; then
    echo -e "  ${GREEN}✓${NC} Embedding cache (${CACHE_DIR})"
else
    echo -e "  ${YELLOW}–${NC} Embedding cache (kept)"
fi
if [[ "${REMOVE_MODELS:-false}" == true ]]; then
    echo -e "  ${GREEN}✓${NC} GGUF models (${GGUF_DIR})"
else
    echo -e "  ${YELLOW}–${NC} GGUF models (kept)"
fi
if [[ "${REMOVE_REPO:-false}" == true ]]; then
    echo -e "  ${GREEN}✓${NC} rag-system repo"
else
    echo -e "  ${YELLOW}–${NC} rag-system repo (kept)"
fi
if [[ "${REMOVE_VENV:-false}" == true ]]; then
    echo -e "  ${GREEN}✓${NC} Python venv"
else
    echo -e "  ${YELLOW}–${NC} Python venv (kept)"
fi
echo ""
echo -e "${BOLD}What was NOT removed:${NC}"
echo -e "  ${CYAN}•${NC} Your Obsidian vaults and documentation"
echo -e "  ${CYAN}•${NC} openfox-rag (separate repo)"
echo ""
echo -e "${BOLD}To reload your shell:${NC}"
if [[ "$SHELL_NAME" == "fish" ]]; then
    echo -e "  ${CYAN}source ~/.config/fish/config.fish${NC}"
elif [[ "$SHELL_NAME" == "zsh" ]]; then
    echo -e "  ${CYAN}source ~/.zshrc${NC}"
elif [[ "$SHELL_NAME" == "bash" ]]; then
    echo -e "  ${CYAN}source ~/.bashrc${NC}"
else
    echo -e "  ${CYAN}source ${SHELL_CONFIG}${NC}"
fi
echo ""
