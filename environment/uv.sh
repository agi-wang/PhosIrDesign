#!/bin/bash
# Clean uv-based environment bootstrap for GitHub release
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PATH="$ROOT_DIR/.venv"
REQUIREMENTS_FILE="$ROOT_DIR/environment/requirements.txt"
PY_VER="${UV_PYTHON:-3.9}"

if [ "${UV_LOCAL_TEST:-0}" = "1" ]; then
  mkdir -p "$ROOT_DIR/Project_Output"
  exit 0
fi

echo "==========================================="
echo "Setting up Python environment with uv"
echo "==========================================="

ensure_brew_and_libomp() {
  OS_NAME="$(uname -s)"
  if [ "$OS_NAME" != "Darwin" ]; then
    echo "Non-macOS detected ($OS_NAME); skipping Homebrew/libomp check."
    return
  fi

  if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Add brew to PATH for current shell
    if [ -d "/opt/homebrew/bin" ]; then
      export PATH="/opt/homebrew/bin:$PATH"
    elif [ -d "/usr/local/bin" ]; then
      export PATH="/usr/local/bin:$PATH"
    fi
  fi

  if ! command -v brew >/dev/null 2>&1; then
    echo "Failed to install or locate Homebrew. Please install it manually."
    exit 1
  fi

  if ! brew list --versions libomp >/dev/null 2>&1; then
    echo "libomp not found. Installing libomp via Homebrew..."
    brew install libomp
  else
    echo "libomp already installed."
  fi

  LIBOMP_PREFIX="$(brew --prefix libomp 2>/dev/null || true)"
  if [ -n "$LIBOMP_PREFIX" ] && [ -d "$LIBOMP_PREFIX/lib" ]; then
    LIBOMP_LIB_PATH="$LIBOMP_PREFIX/lib"
    case ":${DYLD_LIBRARY_PATH:-}:" in
      *":$LIBOMP_LIB_PATH:"*) ;;
      *) export DYLD_LIBRARY_PATH="$LIBOMP_LIB_PATH${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" ;;
    esac
    echo "Ensured DYLD_LIBRARY_PATH includes libomp at $LIBOMP_LIB_PATH"
  else
    echo "WARNING: libomp prefix not found; xgboost may still fail to load."
  fi
}

ensure_brew_and_libomp

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi

  # Try common install locations
  if [ -f "$HOME/.local/bin/uv" ]; then
    export PATH="$HOME/.local/bin:$PATH"
  elif [ -f "$HOME/.cargo/env" ]; then
    # shellcheck disable=SC1090
    source "$HOME/.cargo/env"
  fi

  if command -v uv >/dev/null 2>&1; then
    return
  fi

  echo "uv not found. Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"

  if ! command -v uv >/dev/null 2>&1; then
    echo "Failed to install or locate uv. Please install it manually."
    exit 1
  fi
}

ensure_uv
echo "uv version: $(uv --version)"

# Auto-detect China mainland and set USTC mirror
configure_china_mirror() {
  if [ -n "${UV_INDEX_URL:-}" ]; then
    echo "UV_INDEX_URL already set to $UV_INDEX_URL; skipping mirror detection."
    return
  fi
  echo "Detecting network region (testing actual download speed)..."
  # Test real download speed from PyPI CDN, not just DNS/homepage
  local speed
  speed=$(curl -s --connect-timeout 5 --max-time 8 -o /dev/null -w '%{speed_download}' \
    "https://files.pythonhosted.org/packages/py3/p/pip/pip-24.0-py3-none-any.whl" 2>/dev/null || echo "0")
  # speed is in bytes/sec; 100KB/s = 102400
  local threshold=102400
  local speed_int=${speed%%.*}
  speed_int=${speed_int:-0}
  echo "PyPI download speed: ${speed_int} B/s (threshold: ${threshold} B/s)"
  if [ "$speed_int" -lt "$threshold" ]; then
    echo "PyPI slow/unreachable; switching to USTC mirror for faster downloads."
    export UV_INDEX_URL="https://pypi.mirrors.ustc.edu.cn/simple/"
    echo "UV_INDEX_URL=$UV_INDEX_URL"
  else
    echo "PyPI fast enough; using default index."
  fi
}
configure_china_mirror

if [ -f "$VENV_PATH/bin/activate" ] && ! grep -Fq "VIRTUAL_ENV='$VENV_PATH'" "$VENV_PATH/bin/activate"; then
  echo "Existing virtual environment was created for a different path; recreating..."
  rm -rf "$VENV_PATH"
fi

if [ -x "$VENV_PATH/bin/python" ]; then
  CURRENT_PY_MM="$("$VENV_PATH/bin/python" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
  DESIRED_PY_MM="$(printf '%s' "$PY_VER" | sed -E 's/[^0-9]*([0-9]+)\.([0-9]+).*/\1.\2/')"
  if [ -n "$DESIRED_PY_MM" ] && [ "$CURRENT_PY_MM" != "$DESIRED_PY_MM" ]; then
    echo "Existing virtual environment uses Python $CURRENT_PY_MM; recreating for Python $DESIRED_PY_MM..."
    rm -rf "$VENV_PATH"
  fi
fi

# Recreate venv only when explicitly requested
if [ "${UV_RECREATE:-0}" = "1" ] && [ -d "$VENV_PATH" ]; then
  echo "UV_RECREATE=1 detected, removing existing venv..."
  rm -rf "$VENV_PATH"
fi

if [ ! -d "$VENV_PATH" ]; then
  echo "Creating virtual environment at $VENV_PATH (Python $PY_VER)..."
  uv venv "$VENV_PATH" --python "$PY_VER" || uv venv "$VENV_PATH"
else
  echo "Using existing virtual environment at $VENV_PATH"
fi

echo "Activating virtual environment..."
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"
export PATH="$VENV_PATH/bin:$PATH"
hash -r 2>/dev/null || true

echo "Installing core requirements..."
uv pip install -r "$REQUIREMENTS_FILE"

python - <<'PY'
from rdkit import rdBase
expected = "2024.09.6"
actual = rdBase.rdkitVersion
if actual != expected:
    raise SystemExit(f"RDKit version mismatch: expected {expected}, got {actual}")
print(f"RDKit version: {actual}")
PY

echo ""
echo "Installing PyTorch CPU wheel..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "Installing extra ML utilities..."
EXTRA_PKGS=(
  imbalanced-learn
  statsmodels
)
uv pip install "${EXTRA_PKGS[@]}"

echo ""
echo "==========================================="
echo "Environment setup complete!"
echo "==========================================="
echo "Python version: $(python --version)"
echo ""
echo "Activate with: source \"$VENV_PATH/bin/activate\""
echo "To recreate: UV_RECREATE=1 UV_PYTHON=$PY_VER bash environment/uv.sh"
