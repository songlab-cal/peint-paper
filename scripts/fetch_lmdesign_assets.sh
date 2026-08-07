#!/usr/bin/env bash
# Fetch the assets the lm-design energy needs for the ESM-MCMC proposer experiment
# (benchmarks/esm_mcmc_proposer.py):
#   data/lm_design/linear_projection_model.pt   the ~12k-param linear distogram projection
#                                               (rides on frozen ESM2 attention maps)
#   data/lm_design/ngram_stats/*.p              background n-gram frequencies (orders 1-4)
#
# Both come from facebookresearch/esm examples/lm-design (MIT). The projection weights are a
# single small file; the n-gram stats are small pickles. Everything lands under data/lm_design/
# and is gitignored. Point PEINT_PAPER_LMDESIGN_DIR elsewhere if you already have copies.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="${PEINT_PAPER_LMDESIGN_DIR:-$ROOT/data/lm_design}"
NGRAM_DIR="$DIR/ngram_stats"
mkdir -p "$NGRAM_DIR"

WEIGHTS_URL="https://dl.fbaipublicfiles.com/fair-esm/examples/lm_design/linear_projection_model.pt"
NGRAM_BASE="https://raw.githubusercontent.com/facebookresearch/esm/main/examples/lm-design/utils/ngram_stats"

if [ -f "$DIR/linear_projection_model.pt" ]; then
    echo "linear projection weights already present: $DIR/linear_projection_model.pt"
else
    echo "Downloading linear projection weights -> $DIR ..."
    curl -fSL "$WEIGHTS_URL" -o "$DIR/linear_projection_model.pt"
fi

for f in monogram_seg.p bigram_seg.p trigram_seg.p quadgram_seg.p; do
    if [ -f "$NGRAM_DIR/$f" ]; then
        echo "ngram stat already present: $NGRAM_DIR/$f"
    else
        echo "Downloading $f -> $NGRAM_DIR ..."
        curl -fSL "$NGRAM_BASE/$f" -o "$NGRAM_DIR/$f"
    fi
done

echo "Done. lm-design assets in $DIR"
