# peint-paper

Paper-specific figures and benchmarks for PEINT. Depends on the model library
[`peint`](../peint) (imported as the `protevo` package) via an editable install.

## Setup

```bash
# 1. Install the model library (editable) + benchmarking deps
pip install -e ../peint
pip install -e .            # this repo's benchmarking extras

# 2. Build Historian (ancestral reconstruction; used by the indel / PCP analyses)
git submodule update --init --recursive
bash scripts/build_historian.sh     # -> historian/bin/historian
```

Add `historian/bin` to your `PATH` (the analysis code invokes `historian`).

## External tools

| Tool | Purpose | How it's provided |
|------|---------|-------------------|
| IQ-TREE 2 / AliSim | classical-model simulation | provided by `peint` (`iqtree2` submodule) |
| Historian | ancestral sequence reconstruction | submodule + patched `historian_makefile/Makefile` |
| MAFFT | alignment | expected on `PATH` (benchmarking dep) |

Data (inputs and large result dirs) live outside the repo and are referenced via
a single configurable `DATA_ROOT` (see `paper_config.py`).
