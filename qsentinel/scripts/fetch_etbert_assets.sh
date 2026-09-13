#!/usr/bin/env bash
# Fetch the ET-BERT artifacts the byte baseline replicates against.
#
#   reference/pretrained/et-bert/pretrained_model.bin   716 MB, Google Drive (needs gdown)
#   reference/pretrained/et-bert/encryptd_vocab.txt     60,005 tokens, from the repo
#   reference/ET-BERT/                                  the UER-py source we transcribed
#
# The CSTNET-TLS 1.3 corpus itself is a separate manual download -- see
# data/CSTNET-TLS1.3/README.md.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/reference/pretrained/et-bert"
mkdir -p "$DEST"

if [ ! -s "$DEST/pretrained_model.bin" ]; then
  echo "==> pre-trained checkpoint (716 MB)"
  python -m gdown "https://drive.google.com/uc?id=1r1yE34dU2W8zSqx1FkB8gCWri4DQWVtE" \
    -O "$DEST/pretrained_model.bin"
fi

if [ ! -s "$DEST/encryptd_vocab.txt" ]; then
  echo "==> vocabulary"
  curl -fsSL -o "$DEST/encryptd_vocab.txt" \
    https://raw.githubusercontent.com/linwhitehat/ET-BERT/main/models/encryptd_vocab.txt
fi

if [ ! -d "$ROOT/reference/ET-BERT" ]; then
  echo "==> reference implementation"
  git clone --depth 1 https://github.com/linwhitehat/ET-BERT.git "$ROOT/reference/ET-BERT"
fi
echo "done."
