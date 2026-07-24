#!/usr/bin/env bash
# rag-system — Standalone uninstaller
# Usage: bash uninstall.sh [--dry-run]

set -euo pipefail

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
echo -e "  Removes: services, aliases, cache, models, venv, repo"
echo -e "  ${YELLOW}Does NOT remove:${NC} Obsidian vaults, documentation, openfox-rag"
echo ""

detect_shell
info "Shell: ${SHELL_NAME} → ${SHELL_CONFIG}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RAG_DIR="$SCRIPT_DIR"

# Load config
GGUF_DIR="${HOME}/models/GGUF/rag"
VENV_DIR="${HOME}/.venv/main"
CACHE_DIR="${HOME}/.rag"
if [[ -f "${RAG_DIR}/config.sh" ]]; then
    source "${RAG_DIR}/config.sh" 2>/dev/null || true
fi

# ── Step 1: Stop services ──────────────────────────────────
header "Step 1/5: Stop services"

if pgrep -f "rag_server_rerank" &>/dev/null || \
   pgrep -f "llama-server.*8181" &>/dev/null || \
   pgrep -f "llama-server.*8184" &>/dev/null; then
    ask_yes_no "Stop all RAG services?" "y" STOP_SERVICES
    if [[ "$STOP_SERVICES" == true ]]; then
        run pkill -f "rag_server_rerank" 2>/dev/null || true
        run pkill -f "llama-server.*8181" 2>/dev/null || true
        run pkill -f "llama-server.*8184" 2>/dev/null || true
        run pkill -f "Qwen3-Embedding" 2>/dev/null || true
        run pkill -f "Qwen3-Reranker" 2>/dev/null || true
        success "Services stopped"
    fi
else
    success "No RAG services running"
fi

# ── Step 2: Remove aliases ─────────────────────────────────
header "Step 2/5: Remove shell aliases"

if grep -q "rag-system aliases\|openfox-rag aliases" "$SHELL_CONFIG" 2>/dev/null; then
    ask_yes_no "Remove RAG aliases from ${SHELL_CONFIG}?" "y" REMOVE_ALIASES
    if [[ "$REMOVE_ALIASES" == true ]]; then
        run sed -i '/# ── rag-system aliases ──/,/# ── end rag-system aliases ──/d' "$SHELL_CONFIG"
        run sed -i '/# ── openfox-rag aliases ──/,/# ── end openfox-rag aliases ──/d' "$SHELL_CONFIG"
        run sed -i '/^$/N;/^\n$/d' "$SHELL_CONFIG"
        success "Aliases removed"
    fi
else
    info "No RAG aliases found"
fi

# ── Step 3: Remove cache ───────────────────────────────────
header "Step 3/5: Remove embedding cache"

if [[ -d "$CACHE_DIR" ]]; then
    CACHE_SIZE=$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)
    info "Cache: ${CACHE_DIR} (${CACHE_SIZE})"
    ask_yes_no "Remove embedding cache?" "y" REMOVE_CACHE
    if [[ "$REMOVE_CACHE" == true ]]; then
        run rm -rf "$CACHE_DIR"
        success "Cache removed"
    fi
else
    info "No cache found"
fi

# ── Step 4: Remove models ──────────────────────────────────
header "Step 4/5: Remove GGUF models"

if [[ -d "$GGUF_DIR" ]]; then
    MODEL_SIZE=$(du -sh "$GGUF_DIR" 2>/dev/null | cut -f1)
    info "Models: ${GGUF_DIR} (${MODEL_SIZE})"
    ask_yes_no "Remove GGUF models?" "y" REMOVE_MODELS
    if [[ "$REMOVE_MODELS" == true ]]; then
        run rm -rf "$GGUF_DIR"
        success "Models removed"
    fi
else
    info "No models found"
fi

# ── Step 5: Remove repo + venv ─────────────────────────────
header "Step 5/5: Remove repo and venv"

ask_yes_no "Remove rag-system repo (${RAG_DIR})?" "y" REMOVE_REPO
if [[ "$REMOVE_REPO" == true ]]; then
    run rm -rf "$RAG_DIR"
    success "Repo removed"
fi

if [[ -d "$VENV_DIR" ]]; then
    VENV_SIZE=$(du -sh "$VENV_DIR" 2>/dev/null | cut -f1)
    info "Venv: ${VENV_DIR} (${VENV_SIZE})"
    ask_yes_no "Remove Python venv?" "y" REMOVE_VENV
    if [[ "$REMOVE_VENV" == true ]]; then
        run rm -rf "$VENV_DIR"
        # Clean activate line from shell config
        if [[ -f "$SHELL_CONFIG" ]]; then
            run sed -i "\|${VENV_DIR}/bin/activate|d" "$SHELL_CONFIG"
        fi
        success "Venv removed (+ activate line cleaned)"
    fi
else
    info "No venv found"
fi

# ── Summary ─────────────────────────────────────────────────
header "Done"

echo -e "${BOLD}Removed:${NC}"
[[ "${STOP_SERVICES:-false}" == true ]] && echo -e "  ${GREEN}✓${NC} Services"
[[ "${REMOVE_ALIASES:-false}" == true ]] && echo -e "  ${GREEN}✓${NC} Aliases"
[[ "${REMOVE_CACHE:-false}" == true ]] && echo -e "  ${GREEN}✓${NC} Cache"
[[ "${REMOVE_MODELS:-false}" == true ]] && echo -e "  ${GREEN}✓${NC} Models"
[[ "${REMOVE_REPO:-false}" == true ]] && echo -e "  ${GREEN}✓${NC} Repo"
[[ "${REMOVE_VENV:-false}" == true ]] && echo -e "  ${GREEN}✓${NC} Venv"
echo ""
echo -e "${BOLD}Not removed:${NC} Obsidian vaults, documentation, openfox-rag"
echo ""
echo -e "${BOLD}Reload:${NC} ${CYAN}source ${SHELL_CONFIG}${NC}"
echo ""
