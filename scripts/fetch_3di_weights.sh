#!/usr/bin/env bash
# Fetch the weights needed to annotate sequences with 3Di structural states.
#
# Two separate things, from two different places:
#   data/prostt5/                  ProstT5 encoder, pulled from HuggingFace (Rostlab/ProstT5_fp16)
#   data/prostt5/cnn_chkpnt/model.pt   the small CNN head mapping embeddings -> 3Di states,
#                                      a file in the ProstT5 GitHub repo (~a few MB)
#
# paper/threedi.py will download the CNN head on demand if it is missing, but it needs the
# encoder cache to exist; running this up front keeps the first benchmark run from stalling
# on a multi-GB download. Override the locations with PEINT_PAPER_PROSTT5_CACHE_DIR /
# PEINT_PAPER_PROSTT5_CNN_CHECKPOINT.
#
# NOTE: this does NOT fetch the ground-truth structures or the a3m alignments. Those come
# from the trRosetta training set (~30 GB), which you should download separately:
#   https://files.ipd.uw.edu/pub/trRosetta/training_set.tar.gz
# then point PEINT_PAPER_GROUND_TRUTH_STRUCTURE_DIR / PEINT_PAPER_INPUT_A3M_DIR at it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${PEINT_PAPER_PROSTT5_CACHE_DIR:-$ROOT/data/prostt5}"
CNN_CKPT="${PEINT_PAPER_PROSTT5_CNN_CHECKPOINT:-$CACHE_DIR/cnn_chkpnt/model.pt}"

CNN_URL="https://github.com/mheinzinger/ProstT5/raw/main/cnn_chkpnt/model.pt"

mkdir -p "$CACHE_DIR" "$(dirname "$CNN_CKPT")"

# --- CNN head ---
if [ -s "$CNN_CKPT" ]; then
    echo "CNN head already present: $CNN_CKPT"
else
    echo "Downloading ProstT5 CNN head -> $CNN_CKPT"
    curl -fsSL -o "$CNN_CKPT" "$CNN_URL"
    echo "Done."
fi

# --- ProstT5 encoder ---
# from_pretrained(cache_dir=...) is the same call paper/threedi.py makes, so warming it here
# populates exactly the cache layout the benchmark expects.
echo "Fetching ProstT5 encoder into $CACHE_DIR (skipped if already cached) ..."
python - "$CACHE_DIR" <<'PY'
import sys
from transformers import T5EncoderModel, T5Tokenizer

cache_dir = sys.argv[1]
T5Tokenizer.from_pretrained("Rostlab/ProstT5_fp16", do_lower_case=False, cache_dir=cache_dir)
T5EncoderModel.from_pretrained("Rostlab/ProstT5_fp16", cache_dir=cache_dir)
print("ProstT5 encoder cached.")
PY

echo
echo "3Di weights ready:"
echo "  encoder cache: $CACHE_DIR"
echo "  CNN head     : $CNN_CKPT"
