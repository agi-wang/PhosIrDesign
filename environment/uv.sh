#!/bin/bash
# Clean uv-based environment bootstrap for GitHub release
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PATH="$ROOT_DIR/.venv"
REQUIREMENTS_FILE="$ROOT_DIR/environment/requirements.txt"
PY_VER="${UV_PYTHON:-3.9}"
XGBOOST_VERSION="${XGBOOST_VERSION:-2.1.4}"
PATCHED_XGBOOST_WHEEL="${PATCHED_XGBOOST_WHEEL:-}"
AUTO_BUILD_PATCHED_XGBOOST="${AUTO_BUILD_PATCHED_XGBOOST:-0}"
INSTALL_TORCH="${INSTALL_TORCH:-0}"

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

run_privileged() {
  if [ "$(id -u)" = "0" ]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "WARNING: need root privileges or sudo to install: $*" >&2
    return 1
  fi
}

ensure_patched_xgboost_build_deps() {
  OS_NAME="$(uname -s)"
  if [ "$OS_NAME" != "Linux" ]; then
    return
  fi

  missing=()
  for tool in gcc g++ cmake patch; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      missing+=("$tool")
    fi
  done

  if [ "${#missing[@]}" -gt 0 ] && command -v apt-get >/dev/null 2>&1; then
    echo "Installing XGBoost source-build dependencies: build-essential cmake ninja-build patch"
    run_privileged apt-get update -y
    run_privileged env DEBIAN_FRONTEND=noninteractive apt-get install -y \
      build-essential cmake ninja-build patch
  fi

  still_missing=()
  for tool in gcc g++ cmake patch; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      still_missing+=("$tool")
    fi
  done

  if [ "${#still_missing[@]}" -gt 0 ]; then
    echo "Missing XGBoost build tools: ${still_missing[*]}" >&2
    exit 1
  fi
}

ensure_linux_openmp_runtime() {
  OS_NAME="$(uname -s)"
  if [ "$OS_NAME" != "Linux" ]; then
    return
  fi

  if command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q 'libgomp\.so\.1'; then
    return
  fi

  if [ -e /usr/lib/x86_64-linux-gnu/libgomp.so.1 ] || [ -e /lib/x86_64-linux-gnu/libgomp.so.1 ]; then
    return
  fi

  if command -v apt-get >/dev/null 2>&1; then
    echo "Installing Linux OpenMP runtime for XGBoost: libgomp1"
    run_privileged apt-get update -y
    run_privileged env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libgomp1
  fi
}

find_cached_patched_xgboost_wheel() {
  python - "$ROOT_DIR/wheelhouse" "$XGBOOST_VERSION" <<'PY'
from pathlib import Path
import sys

from packaging.tags import sys_tags
from packaging.utils import InvalidWheelFilename, parse_wheel_filename

wheelhouse = Path(sys.argv[1])
version = sys.argv[2]
supported_tags = set(sys_tags())
compatible = []

for wheel in sorted(wheelhouse.glob(f"xgboost-{version}-*.whl")):
    try:
        name, wheel_version, _build, wheel_tags = parse_wheel_filename(wheel.name)
    except InvalidWheelFilename as exc:
        print(f"Skipping invalid patched XGBoost wheel: {wheel.name} ({exc})", file=sys.stderr)
        continue

    if name != "xgboost" or str(wheel_version) != version:
        continue

    if wheel_tags & supported_tags:
        compatible.append(wheel)
    else:
        print(f"Skipping incompatible patched XGBoost wheel: {wheel.name}", file=sys.stderr)

if compatible:
    print(compatible[-1])
PY
}

require_compatible_patched_xgboost_wheel() {
  local wheel_path="$1"
  python - "$wheel_path" "$XGBOOST_VERSION" <<'PY'
from pathlib import Path
import sys

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

wheel = Path(sys.argv[1])
expected_version = sys.argv[2]
name, version, _build, wheel_tags = parse_wheel_filename(wheel.name)

if name != "xgboost" or str(version) != expected_version:
    raise SystemExit(f"Unexpected patched XGBoost wheel name/version: {wheel.name}")

if not (wheel_tags & set(sys_tags())):
    raise SystemExit(f"Incompatible patched XGBoost wheel for this Python/platform: {wheel.name}")
PY
}

install_patched_xgboost() {
  local patched_wheel="$PATCHED_XGBOOST_WHEEL"
  local patch_file="$ROOT_DIR/patches/xgboost-$XGBOOST_VERSION-mac-libcxx-column-shuffle.patch"

  echo "Note: original experiments were run on macOS; this XGBoost patch preserves data reproducibility."

  if [ -n "$patched_wheel" ]; then
    if [ ! -f "$patched_wheel" ]; then
      echo "PATCHED_XGBOOST_WHEEL not found: $patched_wheel" >&2
      exit 1
    fi
    require_compatible_patched_xgboost_wheel "$patched_wheel"
  else
    patched_wheel="$(find_cached_patched_xgboost_wheel)"
  fi

  if [ -n "$patched_wheel" ] && [ "$patch_file" -nt "$patched_wheel" ]; then
    if [ "$AUTO_BUILD_PATCHED_XGBOOST" = "1" ]; then
      echo "Cached patched XGBoost wheel is older than patch; rebuilding."
      patched_wheel=""
    else
      echo "Cached patched XGBoost wheel is older than patch: $patched_wheel" >&2
      echo "Rebuild it before running, or set AUTO_BUILD_PATCHED_XGBOOST=1 to allow source compilation." >&2
      exit 1
    fi
  fi

  if [ -z "$patched_wheel" ] && [ "$AUTO_BUILD_PATCHED_XGBOOST" = "1" ]; then
    if [ ! -f "$patch_file" ]; then
      echo "XGBoost patch file not found: $patch_file" >&2
      exit 1
    fi

    ensure_patched_xgboost_build_deps
    echo "Building patched XGBoost wheel..."
    uv pip install pip wheel setuptools
    PYTHON_BIN=python XGBOOST_VERSION="$XGBOOST_VERSION" \
      "$ROOT_DIR/scripts/build_patched_xgboost_wheel.sh"
    patched_wheel="$(find_cached_patched_xgboost_wheel)"
  elif [ -n "$patched_wheel" ]; then
    echo "Using prebuilt patched XGBoost wheel: $patched_wheel"
  fi

  if [ -z "$patched_wheel" ] || [ ! -f "$patched_wheel" ]; then
    echo "Prebuilt patched XGBoost wheel not found in $ROOT_DIR/wheelhouse." >&2
    echo "Add a Linux-compatible wheel, or set AUTO_BUILD_PATCHED_XGBOOST=1 to allow source compilation." >&2
    exit 1
  fi

  ensure_linux_openmp_runtime

  echo "Installing patched XGBoost wheel: $patched_wheel"
  uv pip install --force-reinstall --no-deps "$patched_wheel"
  python - <<'PY'
import json
import xgboost as xgb
print("XGBoost version:", xgb.__version__)
print("XGBoost build_info:", json.dumps(xgb.build_info(), sort_keys=True))
PY
}

install_macos_xgboost() {
  echo "macOS detected; installing unpatched XGBoost $XGBOOST_VERSION from PyPI."
  uv pip install --force-reinstall --no-deps "xgboost==$XGBOOST_VERSION"
  python - <<'PY'
import json
import xgboost as xgb
print("XGBoost version:", xgb.__version__)
print("XGBoost build_info:", json.dumps(xgb.build_info(), sort_keys=True))
PY
}

install_xgboost() {
  OS_NAME="$(uname -s)"
  if [ "$OS_NAME" = "Darwin" ]; then
    install_macos_xgboost
  else
    install_patched_xgboost
  fi
}

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

install_xgboost

python - <<'PY'
from rdkit import rdBase
expected = "2024.09.6"
actual = rdBase.rdkitVersion
if actual != expected:
    raise SystemExit(f"RDKit version mismatch: expected {expected}, got {actual}")
print(f"RDKit version: {actual}")
PY

if [ "$INSTALL_TORCH" = "1" ]; then
  echo ""
  echo "Installing optional PyTorch CPU wheel..."
  uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
  echo ""
  echo "Skipping optional PyTorch install. Set INSTALL_TORCH=1 to install CPU-only PyTorch."
fi

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
echo "Optional PyTorch CPU install: INSTALL_TORCH=1 bash environment/uv.sh"
