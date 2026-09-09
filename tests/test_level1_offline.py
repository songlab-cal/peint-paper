"""Level-1 replotting must be deterministic and offline.

The generalization panels label families with Pfam/SCOPe/ECOD. ``paper.generalization`` caches
those labels as ``family_labels.json`` and the figure_data tier ships a versioned copy. If the
cache is not found, the module rebuilds it from live ECOD/SCOPe/SIFTS/Pfam downloads, which
drift against the published figure. These tests pin the offline path.
"""

import json
import pathlib
import socket

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
FIGURE_DATA = REPO / "local_data" / "figure_data"

pytestmark = pytest.mark.skipif(
    not (FIGURE_DATA / "annotations" / "family_labels.json").exists(),
    reason="needs the figure_data tier (annotations/family_labels.json)",
)

# The published Extended Data Fig. 9 partition, under the powered Pfam-family scheme.
PUBLISHED_NOVEL = 81


@pytest.fixture
def no_network(monkeypatch):
    """Any socket use becomes a hard failure, so a download cannot pass silently."""
    def _blocked(*a, **k):
        raise AssertionError("Level-1 must not touch the network")

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)


@pytest.fixture
def level1_env(monkeypatch, tmp_path):
    """A local_data tree holding only figure_data, i.e. the replot tier."""
    ld = tmp_path / "local_data"
    (ld).mkdir()
    (ld / "figure_data").symlink_to(FIGURE_DATA)
    monkeypatch.setenv("PEINT_PAPER_LOCAL_DATA", str(ld))
    monkeypatch.setenv("PEINT_PAPER_LOCAL_DATA_ONLY", "1")
    monkeypatch.setenv("PEINT_PAPER_FIGURES_DIR", str(tmp_path / "out"))
    return ld


def test_annotation_dir_falls_back_to_shipped_labels(level1_env):
    import importlib
    import paper_config as cfg
    importlib.reload(cfg)
    p = pathlib.Path(cfg.ANNOTATION_DIR)
    assert p.is_dir(), p
    assert (p / "family_labels.json").exists()
    assert "figure_data" in str(p), f"expected the shipped labels, got {p}"


def test_partition_offline_matches_published(level1_env, no_network):
    """The shipped labels must reproduce the published novel-set size, with no network."""
    import importlib
    import paper_config as cfg
    importlib.reload(cfg)
    from paper import generalization as gen
    importlib.reload(gen)

    labels = gen.build_family_labels()          # must hit the shipped cache, not download
    assert len(labels) > 10000, len(labels)

    part = gen.partition("pfam_family", labels=labels)
    assert len(part["novel"]) == PUBLISHED_NOVEL, (
        f"novel set is {len(part['novel'])}, published is {PUBLISHED_NOVEL}"
    )


def test_shipped_labels_are_valid_json_and_cover_the_eval_set(level1_env):
    import importlib
    import paper_config as cfg
    importlib.reload(cfg)
    labels = json.loads((pathlib.Path(cfg.ANNOTATION_DIR) / "family_labels.json").read_text())
    from paper import generalization as gen
    importlib.reload(gen)
    missing = [f for f in gen.eval_families() if f not in labels]
    assert not missing, f"{len(missing)} eval families unlabelled, e.g. {missing[:5]}"


# ======================================================================================
# Fig 5a / 5d / ED 7a must replot without a ProteinGym checkout or CherryML cache
# ======================================================================================
VEP_RUNS = REPO / "local_data" / "vep" / "test_lls" / "production"

# The assay set the VEP panels compare models over: the intersection of the per-run tables.
PUBLISHED_VEP_FAMILIES = 201


def test_figure5_does_not_read_a_proteingym_cache():
    """The family list must come from the shipped tables, not a CherryML cache directory."""
    src = (REPO / "figures" / "figure5_vep.py").read_text()
    assert "_cache_cherryml" not in src, "figure5_vep must not reach into a ProteinGym cache"
    assert "PROTEINGYM_DIR" not in src, "figure5_vep must not need a ProteinGym checkout"


@pytest.mark.skipif(not VEP_RUNS.is_dir(), reason="needs the shipped local_data/vep tables")
def test_vep_family_overlap_comes_out_of_the_shipped_tables():
    """Intersecting the per-run tables reproduces the published assay set on its own."""
    import csv

    runs = sorted(d for d in VEP_RUNS.iterdir() if (d / "spearman_results.csv").exists())
    assert runs, "no per-run spearman_results.csv found"
    overlap = None
    for d in runs:
        fams = {r["family"] for r in csv.DictReader(open(d / "spearman_results.csv"))}
        overlap = fams if overlap is None else overlap & fams
    assert len(overlap) == PUBLISHED_VEP_FAMILIES, len(overlap)


def test_figure5_imports_without_proteingym(monkeypatch):
    monkeypatch.setenv("PEINT_PROTEINGYM_DIR", "/nonexistent/proteingym")
    monkeypatch.setenv("PEINT_PAPER_LOCAL_DATA_ONLY", "1")
    import importlib
    mod = importlib.import_module("figures.figure5_vep")
    importlib.reload(mod)
    assert hasattr(mod, "VEP_RESULTS_DIR")
