#!/usr/bin/env bash
# Phase 2, task 2.8 -- self-generated PQC / classical handshake captures.
#
# Wrapper around scripts/gen_pqc_pcaps.py.  Kept as a shell entry point because PHASE-2.md
# names it, and because it is the natural place to prefer a PQC-capable TLS client if one
# ever appears on the box (OpenSSL >= 3.5 or oqs-provider): until then the Python path
# hand-builds the ClientHello, which works with any OpenSSL because it never uses one.
#
# Usage: scripts/gen_pqc_pcaps.sh [output-dir]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
OUT="${1:-$REPO/data/_pqc_synth}"

PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"

if openssl list -kem-algorithms 2>/dev/null | grep -qi 'ML-KEM\|kyber'; then
  echo "note: local OpenSSL advertises a PQ KEM; the hand-built ClientHello is still used"
  echo "      (it is version-independent and exercises the same server behaviour)."
fi

exec "$PY" "$REPO/scripts/gen_pqc_pcaps.py" --out "$OUT"
