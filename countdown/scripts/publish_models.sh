#!/usr/bin/env bash
# Publish models_store/ as a GitHub release and record its hashes in the tracked manifest.
#   scripts/publish_models.sh [tag]        (default models-v1)
# Commit models_store.sha256 afterwards: the manifest in git is what makes a download trusted.
set -euo pipefail
cd "$(dirname "$0")/.."
TAG="${1:-models-v1}"
.venv/bin/countdown models manifest --tag "$TAG"
NOTES="Deployed experts (E1, E2, E3) for \`countdown analyze\`. Verify with \`countdown models verify\`; hashes are in models_store.sha256 in the repo."
if gh release view "$TAG" >/dev/null 2>&1; then
  gh release upload "$TAG" models_store/E*.joblib --clobber
else
  gh release create "$TAG" models_store/E*.joblib --title "Model zoo $TAG" --notes "$NOTES"
fi
echo "published $TAG; now commit models_store.sha256"
