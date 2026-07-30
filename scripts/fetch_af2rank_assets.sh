#!/usr/bin/env bash
# Fetch the assets the AF2Rank structure-scoring path needs:
#   data/af2_params/  AlphaFold 2022-12-06 parameters (~4 GB download)
#   bin/TMalign       TM-align binary, built from the Zhang group source
#   bin/TMscore       TM-score binary (same source family; not used by the code paths
#                     here, but built alongside since the two usually travel together)
#
# Follows the standard ColabDesign/AF2Rank setup. ColabDesign itself is a pip install:
#   pip install git+https://github.com/sokrypton/ColabDesign.git@v1.1.3 --no-deps
#
# Everything lands inside this repo and is gitignored. Point
# PEINT_PAPER_AF2_WEIGHTS_DIR / PEINT_PAPER_TMALIGN_PATH elsewhere if you already
# have copies.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARAMS_DIR="$ROOT/data/af2_params"
BIN_DIR="$ROOT/bin"

AF2_PARAMS_URL="https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"

mkdir -p "$PARAMS_DIR" "$BIN_DIR"

# --- AlphaFold parameters ---
# mk_af_model(data_dir=...) expects the params/ subdirectory underneath.
if [ -d "$PARAMS_DIR/params" ] && [ -n "$(ls -A "$PARAMS_DIR/params" 2>/dev/null)" ]; then
    echo "AlphaFold params already present: $PARAMS_DIR/params"
else
    echo "Downloading AlphaFold params (~4 GB) into $PARAMS_DIR/params ..."
    mkdir -p "$PARAMS_DIR/params"
    curl -fsSL "$AF2_PARAMS_URL" | tar x -C "$PARAMS_DIR/params"
    echo "Done."
fi

# --- TM-align / TM-score ---
build_tm() {
    local name="$1" url="$2"
    if [ -x "$BIN_DIR/$name" ]; then
        echo "$name already built: $BIN_DIR/$name"
        return
    fi
    echo "Building $name ..."
    local src
    src="$(mktemp -d)"
    trap 'rm -rf "$src"' RETURN
    curl -fsSL -o "$src/$name.cpp" "$url"
    # -static keeps the binary usable on nodes without a matching libstdc++; drop it
    # if your toolchain has no static libs.
    g++ -static -O3 -ffast-math -lm -o "$BIN_DIR/$name" "$src/$name.cpp" \
        || g++ -O3 -ffast-math -lm -o "$BIN_DIR/$name" "$src/$name.cpp"
    echo "Built: $BIN_DIR/$name"
}

build_tm TMalign "https://zhanggroup.org/TM-align/TMalign.cpp"
build_tm TMscore "https://zhanggroup.org/TM-score/TMscore.cpp"

echo
echo "AF2Rank assets ready:"
echo "  params : $PARAMS_DIR/params"
echo "  TMalign: $BIN_DIR/TMalign"
echo "  TMscore: $BIN_DIR/TMscore"
