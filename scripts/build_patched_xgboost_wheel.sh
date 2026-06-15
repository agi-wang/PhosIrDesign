#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
XGBOOST_VERSION="${XGBOOST_VERSION:-2.1.4}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build/xgboost-patched-$XGBOOST_VERSION}"
WHEEL_DIR="${WHEEL_DIR:-$ROOT_DIR/wheelhouse}"
PATCH_FILE="${PATCH_FILE:-$ROOT_DIR/patches/xgboost-$XGBOOST_VERSION-mac-libcxx-column-shuffle.patch}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
  fi
fi

if [ ! -f "$PATCH_FILE" ]; then
  echo "Patch file not found: $PATCH_FILE" >&2
  exit 1
fi

if [ -n "${UV_INDEX_URL:-}" ] && [ -z "${PIP_INDEX_URL:-}" ]; then
  export PIP_INDEX_URL="$UV_INDEX_URL"
fi

if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
  "$PYTHON_BIN" -m ensurepip --upgrade
fi

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/download" "$WHEEL_DIR"

"$PYTHON_BIN" -m pip download --no-binary=xgboost --no-deps "xgboost==$XGBOOST_VERSION" -d "$BUILD_DIR/download"

ARCHIVE="$(find "$BUILD_DIR/download" -maxdepth 1 -name "xgboost-$XGBOOST_VERSION*.tar.gz" -print -quit)"
if [ -z "$ARCHIVE" ]; then
  echo "Could not find xgboost source archive in $BUILD_DIR/download" >&2
  exit 1
fi

tar -xzf "$ARCHIVE" -C "$BUILD_DIR"
SRC_DIR="$(find "$BUILD_DIR" -maxdepth 1 -type d -name "xgboost-$XGBOOST_VERSION*" -print -quit)"
if [ -z "$SRC_DIR" ]; then
  echo "Could not find extracted xgboost source directory in $BUILD_DIR" >&2
  exit 1
fi

(
  cd "$SRC_DIR"
  if [ -f "cpp_src/src/common/random.cc" ]; then
    (cd cpp_src && patch -p1 < "$PATCH_FILE")
  else
    patch -p1 < "$PATCH_FILE"
  fi
  "$PYTHON_BIN" -m pip wheel . --no-deps -w "$WHEEL_DIR"
)

echo "Patched XGBoost wheel written to: $WHEEL_DIR"
