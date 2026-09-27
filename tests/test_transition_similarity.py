"""Invariants the held-out/training DIAMOND comparison silently depends on.

Each test here pins something that, if it broke, would not raise -- it would return a
plausible-looking number computed from the wrong data. All of them run on synthetic input, so
none of the cluster data trees are needed.
"""

import gzip
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]

ts = pytest.importorskip(
    "paper.transition_similarity",
    reason="needs paper_config, which resolves the peint checkout",
)


def _write_transitions(path, rows, n_cols):
    """A transitions file in the real format: '<n> transitions' then 'x y t' rows."""
    with open(path, "w") as fh:
        fh.write(f"{len(rows)} transitions\n")
        for x, y, t in rows:
            assert len(x) == len(y) == n_cols
            fh.write(f"{x} {y} {t}\n")


def test_dedupe_collapses_both_transition_directions(tmp_path):
    """Every cherry is listed as (x,y,t) and again as (y,x,t); the union of the columns is the
    distinct sequence set, not twice it."""
    a, b, c = "MKV-AA", "MRV-AA", "MKVWAA"
    p = tmp_path / "fam.txt"
    _write_transitions(p, [(a, b, "0.1"), (b, a, "0.1"), (a, c, "0.2"), (c, a, "0.2")], 6)
    assert ts.unique_aligned_sequences(p) == sorted({a, b, c})


def test_domain_slice_indexes_seq1_columns(tmp_path):
    """A Pfam range is in seq1 residue coordinates, which are the aligned columns 1:1. Slicing
    must honour the 1-based inclusive convention and drop gaps only after slicing."""
    row = "AAAACCCCDDDD"                       # 12 columns, no gaps: columns == residues
    p = tmp_path / "fam.txt"
    _write_transitions(p, [(row, row, "0.1")], 12)
    dom = {"acc": "PF00001", "start": 5, "end": 8}
    seqs = ts.unique_aligned_sequences(p)
    sliced = seqs[0][dom["start"] - 1:dom["end"]].replace(ts.GAP, "")
    assert sliced == "CCCC"


def test_domain_slice_skips_gap_columns(tmp_path):
    """Gaps inside the domain range are removed, and the residues kept are exactly those whose
    columns fall in the range -- not a fixed-length window of the ungapped sequence."""
    row = "--AA--CCCC--"                       # ungapped = "AACCCC"
    p = tmp_path / "fam.txt"
    _write_transitions(p, [(row, row, "0.1")], 12)
    seqs = ts.unique_aligned_sequences(p)
    assert seqs[0][6:10].replace(ts.GAP, "") == "CCCC"     # columns 7-10
    assert seqs[0].replace(ts.GAP, "") == "AACCCC"         # and NOT equal to the slice


def test_check_frame_rejects_a_domain_past_the_last_column():
    """If the transitions and the Pfam ranges ever stop sharing a frame, this must raise rather
    than slice something plausible out of the wrong columns."""
    with pytest.raises(ValueError, match="not in the same frame"):
        ts.check_frame("fam", ncols=50, domains=[{"acc": "PF1", "start": 40, "end": 80}],
                       strict=False)
    ts.check_frame("fam", ncols=50, domains=[{"acc": "PF1", "start": 40, "end": 50}], strict=False)


def test_concatenated_gzip_members_read_back_whole(tmp_path):
    """Query chunking concatenates one gzip member per part. If any reader stopped at the first
    member, most of the hits would vanish without an error -- the highest-risk silent failure in
    the pipeline, since it would just look like fewer queries had matches."""
    parts = []
    for i in range(3):
        p = tmp_path / f"part{i}.tsv.gz"
        with gzip.open(p, "wt") as fh:
            for j in range(10):
                fh.write(f"q{i}_{j}\tsfam_1_A|{j}\t50.0\t100\t50\t100\t100\t1e-5\t80.0\n")
        parts.append(p)
    out = ts.concat_gzip(parts, tmp_path / "joined.tsv.gz")

    assert sum(1 for _ in gzip.open(out, "rt")) == 30
    assert sum(1 for _ in ts._open_text(out)) == 30
    assert len(ts.density_profile(out)) == 30


def test_split_fasta_partitions_every_record_exactly_once(tmp_path):
    """Parts must be disjoint and complete: a dropped record is a query silently never searched,
    and a duplicated one is a query counted twice in the distribution."""
    src = tmp_path / "q.fasta"
    ids = [f"fam{i%3}_1_A|{i}" for i in range(17)]
    with open(src, "w") as fh:
        for i, sid in enumerate(ids):
            fh.write(f">{sid}\nMKV{'A' * (i + 1)}\n")
    parts = ts.split_fasta(src, 4, tmp_path / "parts")

    seen = []
    for p in parts:
        seen += [ln[1:].strip() for ln in open(p) if ln.startswith(">")]
    assert sorted(seen) == sorted(ids)
    assert len(seen) == len(set(seen)) == 17


def test_density_profile_ranks_by_bitscore_and_collapses_families(tmp_path):
    """Raw rank counts sequences; family rank counts distinct training families. A neighbourhood
    of near-duplicates from one family must not read as a dense one."""
    p = tmp_path / "hits.tsv.gz"
    rows = [
        # qseqid, sseqid, pident, length, nident, qlen, slen, evalue, bitscore
        ("q1", "famA_1_A|1", 90.0, 100, 90, 100, 100, "1e-9", 200.0),
        ("q1", "famA_1_A|2", 85.0, 100, 85, 100, 100, "1e-8", 180.0),
        ("q1", "famA_1_A|3", 80.0, 100, 80, 100, 100, "1e-7", 170.0),
        ("q1", "famB_1_A|9", 40.0, 100, 40, 100, 100, "1e-3", 90.0),
    ]
    with gzip.open(p, "wt") as fh:
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    prof = ts.density_profile(p).set_index("qseqid").loc["q1"]

    assert prof["n_hits"] == 4 and prof["n_families"] == 2
    assert prof["gident_rank1"] == pytest.approx(90.0)
    assert prof["gident_rank2"] == pytest.approx(85.0)   # same family, still rank 2 raw
    assert prof["gident_fam1"] == pytest.approx(90.0)
    assert prof["gident_fam2"] == pytest.approx(40.0)    # the next FAMILY is far worse
    assert prof["n_fam_ge30"] == 2 and prof["n_fam_ge50"] == 1
