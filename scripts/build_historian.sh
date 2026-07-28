#!/usr/bin/env bash
# Build Historian (https://github.com/evoldoers/historian) with a Linux-compatible
# Makefile. The upstream Makefile does not build cleanly on recent Linux, so this
# repo ships a patched Makefile (historian_makefile/Makefile) and copies it into
# the submodule before building. Produces historian/bin/historian.
#
# Prereqs: g++/clang++, make, GSL (libgsl-dev), Boost.regex, zlib.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB="$ROOT/historian"
OVERLAY="$ROOT/historian_makefile/Makefile"

if [ ! -d "$SUB/src" ]; then
    echo "Historian submodule not populated. Run:" >&2
    echo "  git submodule update --init --recursive historian" >&2
    exit 1
fi

echo "Copying patched Makefile into the submodule..."
cp "$OVERLAY" "$SUB/Makefile"

# Pick a C++ compiler that can actually find the standard library. The Makefile
# defaults to clang++ and only falls back to g++ when clang++ is *absent*; some
# systems (e.g. this cluster) have a clang++ that can't locate libstdc++ headers,
# so we probe with a trivial <string> compile and fall back to g++ (which needs
# Boost.regex instead of the clang std::regex path).
detect_cpp() {
    for cxx in clang++ g++; do
        command -v "$cxx" >/dev/null 2>&1 || continue
        if printf '#include <string>\nint main(){return 0;}\n' \
            | "$cxx" -std=c++11 -x c++ -c -o /dev/null - >/dev/null 2>&1; then
            echo "$cxx"; return 0
        fi
    done
    return 1
}
CPP="$(detect_cpp)" || { echo "No working C++ compiler (need clang++ or g++)" >&2; exit 1; }
echo "Using C++ compiler: $CPP"

MAKE_ARGS=("CPP=$CPP")
[ "$CPP" = "g++" ] && MAKE_ARGS+=("USING_BOOST=1")

echo "Building historian..."
make -C "$SUB" clean >/dev/null 2>&1 || true
make -C "$SUB" "${MAKE_ARGS[@]}" -j"$(nproc)"

BIN="$SUB/bin/historian"
if [ -x "$BIN" ]; then
    echo "Built: $BIN"
    "$BIN" help >/dev/null 2>&1 && echo "historian runs OK" || echo "WARNING: built but 'historian help' failed"
else
    echo "ERROR: expected binary $BIN was not produced" >&2
    exit 1
fi
