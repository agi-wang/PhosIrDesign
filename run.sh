#!/usr/bin/env bash
set -e
REPO_URL="${REPO_URL:-https://github.com/agi-wang/PhosIrDesign.git}"
REPO_DIR="${REPO_DIR:-PhosIrDesign}"
BRANCH="${BRANCH:-main}"
WORKFLOW_SCRIPT="scripts/workflows/workflow.sh"
SETUP_SCRIPT="environment/uv.sh"
REQUIREMENTS_FILE="environment/requirements.txt"
LOG_STEP=0
log_prefix() { LOG_STEP=$((LOG_STEP + 1)); printf '%04d' "$LOG_STEP"; }
info() { echo "[step $(log_prefix)] INFO: $*"; }
warn() { echo "[step $(log_prefix)] WARNING: $*"; }
err() { echo "[step $(log_prefix)] ERROR: $*" >&2; }
if ! command -v bash >/dev/null; then err "bash not found"; exit 1; fi
OS="$(uname)"
install_with_brew() {
  if ! command -v brew >/dev/null; then /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"; fi
  brew update || true
  for pkg in "$@"; do brew install "$pkg" || true; done
}
install_with_apt() { sudo apt-get update -y || true; sudo apt-get install -y "$@" || true; }
install_with_yum() { sudo yum install -y "$@" || true; }
install_with_pacman() { sudo pacman -Sy --noconfirm "$@" || true; }
ensure_tool() {
  local t="$1"
  if command -v "$t" >/dev/null; then return 0; fi
  info "Installing $t"
  if [ "$OS" = "Darwin" ]; then
    install_with_brew "$t"
  elif command -v apt-get >/dev/null; then
    install_with_apt "$t"
  elif command -v yum >/dev/null; then
    install_with_yum "$t"
  elif command -v pacman >/dev/null; then
    install_with_pacman "$t"
  else
    warn "Unknown package manager; please install $t manually"
  fi
}
ensure_tool curl
ensure_tool git

if [ -f "$WORKFLOW_SCRIPT" ] && [ -f "$SETUP_SCRIPT" ]; then
  info "Detected execution inside project root. Skipping git clone."
  REPO_DIR="."
else
  if [ -d "$REPO_DIR/.git" ]; then
    info "Updating repository"
    cd "$REPO_DIR"
    git fetch --depth 1 origin "$BRANCH"
    git checkout "$BRANCH"
    git pull --ff-only --depth 1
  else
    info "Cloning repository"
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
  fi
fi

if [ -f "$SETUP_SCRIPT" ]; then
  info "Setting up environment via $SETUP_SCRIPT"
  bash "$SETUP_SCRIPT"
  if [ -f ".venv/bin/activate" ]; then
    . ".venv/bin/activate"
    export PATH="$(pwd)/.venv/bin:$PATH"
    hash -r 2>/dev/null || true
  elif [ -f ".venv/Scripts/activate" ]; then
    . ".venv/Scripts/activate"
  fi
else
  if command -v python3 >/dev/null; then PY=python3; elif command -v python >/dev/null; then PY=python; else PY=""; fi
  if [ -n "$PY" ] && [ -f "$REQUIREMENTS_FILE" ]; then
    info "Creating venv and installing requirements"
    "$PY" -m venv .venv || true
    if [ -f ".venv/bin/activate" ]; then
      . ".venv/bin/activate"
      export PATH="$(pwd)/.venv/bin:$PATH"
      hash -r 2>/dev/null || true
    elif [ -f ".venv/Scripts/activate" ]; then
      . ".venv/Scripts/activate"
    fi
    pip install -r "$REQUIREMENTS_FILE" || true
  fi
fi
if [ -f "$WORKFLOW_SCRIPT" ]; then
  info "Running workflow"
  bash "$WORKFLOW_SCRIPT" "$@"
else
  err "No workflow script found"; exit 1
fi
