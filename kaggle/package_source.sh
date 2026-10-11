#!/usr/bin/env bash
# Assemble the Kaggle Dataset that carries the benchmark source (poc + kagglebench)
# alongside the pushed task file. Produces a staging directory ready for
# `kaggle datasets create -p <staging>`.
#
# Usage:
#   bash kaggle/package_source.sh [STAGING_DIR]
#
# Default STAGING_DIR: ./build/kaggle-dataset
#
# This does NOT upload anything. It only stages files locally.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGING="${1:-$REPO_ROOT/build/kaggle-dataset}"
SLUG="where-did-my-data-go-src"

# Only ever wipe an empty dir or a previous staging of this script.
if [ -d "$STAGING" ] && [ -n "$(ls -A "$STAGING")" ] && [ ! -f "$STAGING/dataset-metadata.json" ]; then
  echo "Refusing to wipe $STAGING: it is not empty and is not a staged dataset." >&2
  exit 1
fi
rm -rf "$STAGING"
mkdir -p "$STAGING"
STAGING="$(cd "$STAGING" && pwd)"

# The two importable packages, side by side, shipped as ONE zip file. `kaggle datasets
# create` skips sub-directories by default (--dir-mode skip), so a plain folder layout would
# upload nothing. A single file uploads as-is; task.py's bootstrap imports from the zip
# directly (zipimport) or from wherever Kaggle extracts it.
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cp -R "$REPO_ROOT/src/poc" "$WORK/poc"
cp -R "$REPO_ROOT/kagglebench" "$WORK/kagglebench"

# Strip caches.
find "$WORK" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
find "$WORK" -type f -name '*.pyc' -delete 2>/dev/null || true

(cd "$WORK" && python3 -W ignore::RuntimeWarning -m zipfile -c "$STAGING/wdmdg_src.zip" poc kagglebench)

# Minimal dataset metadata (fill in <USERNAME> before `kaggle datasets create`, or let
# the CLI infer the owner from your authenticated config).
cat > "$STAGING/dataset-metadata.json" <<JSON
{
  "title": "where-did-my-data-go-src",
  "id": "<USERNAME>/${SLUG}",
  "licenses": [{ "name": "CC0-1.0" }]
}
JSON

echo "Staged Kaggle dataset source at: $STAGING"
echo "Contents:"
ls -1 "$STAGING" | sed "s#^#  #"
echo
echo "Next: set the owner in $STAGING/dataset-metadata.json, then"
echo "  kaggle datasets create -p \"$STAGING\" -u"
