"""Domain/fold novelty of the held-out set — the machinery behind the generalization analysis.

A reviewer asked whether the simulation quality we report reflects genuine generalization
or memorization of protein classes seen in training. The eval families are already held out
at the PDB level (no eval PDB appears in the 14,498 training families), but a held-out chain
can still belong to a Pfam family or structural superfamily (SCOPe/ECOD) that *was* in training. This
module labels every one of the 15,051 families with its Pfam and structural classifications and splits
553 held-out (eval) families into:

* **novel** — none of the family's labels (under a given scheme) appears in ANY training family;
* **seen**  — at least one label appears in training;
* **unlabeled** — the chain carries no label under that scheme (excluded from that scheme's test).

The per-metric scripts (``benchmarks/generalization_*.py``) then re-stratify the already-cached
metrics (AF2Rank pLDDT, OmegaFold pLDDT, JSD) by novel vs seen. If the metrics are
indistinguishable across strata — and track the empirical "Real" baseline the same way — the
model generalizes to protein classes absent from training.

Two granularities are supported:

* **family level** — one label set per chain, from SIFTS (structure-mapped, curated). Drives the
  pLDDT and family-mean-JSD stratifications.
* **domain level** — Pfam domains located directly on ``seq1`` with pyhmmer, giving residue
  ranges in seq1 coordinates. Lets the JSD be restricted to the columns of the *novel* domain
  specifically (the reviewer's position-level request).

Data sources, all fetched idempotently into ``cfg.ANNOTATION_DIR``:

* SIFTS ``pdb_chain_pfam.tsv.gz``     — (pdb, chain) -> Pfam accession(s)
* Pfam  ``Pfam-A.clans.tsv.gz``       — Pfam family -> clan
* SCOPe ``dir.cla.scope...txt``       — (pdb, chain) -> SCOP class.fold.superfamily (curated, ~38% cov)
* ECOD  ``ecod.latest.domains.txt``   — (pdb, chain) -> X/H/T groups (auto, ~90% cov, has ranges)
* Pfam  ``Pfam-A.hmm``                — HMMs, for locating domains on seq1 (domain level only)

Structural classification uses SCOPe + ECOD. SIFTS SCOP2 is empty in the current release; CATH was
dropped in favor of ECOD, which covers ~90% of chains (vs CATH ~73%) with the same superfamily/fold
signal. The eval set carries few novel folds under any structural scheme (~15 novel superfamilies),
so those splits are underpowered; the finer Pfam-family split (81 novel) is the powered test.
"""

import gzip
import json
import os
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import paper_config as cfg

# --- remote sources -------------------------------------------------------------------
_SIFTS_BASE = "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv"
_PFAM_BASE = "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release"

_SIFTS_PFAM = "pdb_chain_pfam.tsv.gz"
_PFAM_CLANS = "Pfam-A.clans.tsv.gz"
_SCOPE_CLA = "dir.cla.scope.2.08-stable.txt"
_SCOPE_CLA_URL = f"https://scop.berkeley.edu/downloads/parse/{_SCOPE_CLA}"
# ECOD: near-complete PDB coverage (auto-classified), SCOP-style homology hierarchy. Large (~700 MB).
_ECOD_DOMAINS = "ecod.domains.txt"
_ECOD_URL = "http://prodata.swmed.edu/ecod/distributions/ecod.latest.domains.txt"
_PFAM_HMM_GZ = "Pfam-A.hmm.gz"
_PFAM_HMM = "Pfam-A.hmm"

_FAMILY_RE = re.compile(r"^([0-9a-z]{4})_\d+_([A-Za-z0-9]+)$")

# Novelty schemes: public name -> the label key on each family's label dict.
# Structural classifications are given at two granularities. CATH superfamily / SCOPe superfamily
# are the best-powered structural axes; the fold levels (CATH topology, SCOPe fold) come out with
# only a handful of novel held-out families — the eval set carries essentially no novel folds — so
# they are reported but not relied on.
SCHEMES = {
    "pfam_family": "pfam",           # sequence-domain novelty — the powered test (81 novel)
    "pfam_clan": "pfam_clan",        # homology-group novelty (23 novel)
    "ecod_hgroup": "ecod_h",         # structural, ECOD homology ~superfamily (~89% cov, 15 novel)
    "ecod_tgroup": "ecod_t",         # structural, ECOD topology/fold (16 novel)
    "scop_superfamily": "scop_sf",   # structural, SCOPe (~38% cov, 14 novel)
}


# ======================================================================================
# family lists and name parsing
# ======================================================================================
def parse_family(family: str) -> Tuple[str, str]:
    """``1a2t_1_A`` -> ``("1a2t", "A")``. Raises on an unexpected shape."""
    m = _FAMILY_RE.match(family)
    if not m:
        raise ValueError(f"Cannot parse PDB/chain from family name {family!r}.")
    return m.group(1), m.group(2)


def _load_family_list(path: Path) -> List[str]:
    """Read a family list from a ``{"families": [...]}`` JSON, a bare JSON list, or one-per-line text."""
    path = Path(path)
    text = path.read_text()
    if path.suffix == ".json":
        data = json.loads(text)
        return list(data["families"] if isinstance(data, dict) else data)
    return [line.strip() for line in text.splitlines() if line.strip()]


def train_families() -> List[str]:
    return _load_family_list(cfg.require(cfg.TRAIN_FAMILIES_FILE))


def eval_families() -> List[str]:
    """The 553 held-out families the simulations are evaluated on."""
    return _load_family_list(cfg.require(cfg.EVAL_FAMILIES_FILE))


# ======================================================================================
# fetching
# ======================================================================================
def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if url.startswith("ftp://"):
        # urllib handles ftp, but curl is more robust for some hosts, so prefer it when present.
        if shutil.which("curl"):
            subprocess.run(["curl", "-fsS", "--max-time", "900", "-o", str(tmp), url], check=True)
        else:  # pragma: no cover - fallback
            urllib.request.urlretrieve(url, tmp)
    else:
        with urllib.request.urlopen(url, timeout=900) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
    tmp.replace(dest)


def ensure_annotations() -> None:
    """Fetch the family-level annotation sources into ``cfg.ANNOTATION_DIR`` if absent.

    Idempotent: existing files are left untouched, so every script can call this at start-up
    and "fetch the data" without re-downloading. The Pfam HMMs (domain level) are handled
    separately by :func:`ensure_pfam_hmm`, since they are large and only the domain-JSD script
    needs them.
    """
    d = Path(cfg.ANNOTATION_DIR)
    d.mkdir(parents=True, exist_ok=True)
    for fname, url in [
        (_SIFTS_PFAM, f"{_SIFTS_BASE}/{_SIFTS_PFAM}"),
        (_PFAM_CLANS, f"{_PFAM_BASE}/{_PFAM_CLANS}"),
        (_SCOPE_CLA, _SCOPE_CLA_URL),
    ]:
        dest = d / fname
        if not dest.exists() or dest.stat().st_size == 0:
            print(f"[generalization] fetching {fname} ...")
            _download(url, dest)


# ======================================================================================
# family-level labels
# ======================================================================================
def _sifts_pfam_by_chain() -> Dict[Tuple[str, str], set]:
    out: Dict[Tuple[str, str], set] = {}
    with gzip.open(Path(cfg.ANNOTATION_DIR) / _SIFTS_PFAM, "rt") as fh:
        for line in fh:
            if line[0] == "#" or line.startswith("PDB\t"):
                continue
            p = line.rstrip("\n").split("\t")
            out.setdefault((p[0], p[1]), set()).add(p[3])
    return out


def _pfam_clan_map() -> Dict[str, str]:
    """Pfam family accession -> clan accession; families with no clan map to themselves."""
    out: Dict[str, str] = {}
    with gzip.open(Path(cfg.ANNOTATION_DIR) / _PFAM_CLANS, "rt") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                out[p[0]] = p[1] if p[1] else p[0]
    return out


def _scope_sf_by_chain() -> Dict[Tuple[str, str], set]:
    """(pdb, chain) -> set of SCOPe superfamily strings (sccs truncated to class.fold.superfamily)."""
    out: Dict[Tuple[str, str], set] = {}
    path = Path(cfg.ANNOTATION_DIR) / _SCOPE_CLA
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            pdb, chainfield, sccs = p[1].lower(), p[2], p[3]
            sf = ".".join(sccs.split(".")[:3])
            for seg in chainfield.split(","):
                ch = seg.split(":")[0].strip()
                if ch:
                    out.setdefault((pdb, ch), set()).add(sf)
    return out


def _ecod_groups_by_chain() -> Tuple[Dict[Tuple[str, str], set], Dict[Tuple[str, str], set]]:
    """(pdb, chain) -> ECOD H-group (~superfamily) and T-group (~topology) id sets, from f_id X.H.T.F."""
    path = _ensure_ecod()
    h_by: Dict[Tuple[str, str], set] = {}
    t_by: Dict[Tuple[str, str], set] = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("uid"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 6:
                continue
            parts = p[3].split(".")
            pc = (p[4].lower(), p[5])
            h_by.setdefault(pc, set()).add(".".join(parts[:2]))
            t_by.setdefault(pc, set()).add(".".join(parts[:3]))
    return h_by, t_by


def _ensure_ecod() -> Path:
    """Fetch the (large, ~700 MB) ECOD domains file if absent. Returns its path."""
    dest = Path(cfg.ANNOTATION_DIR) / _ECOD_DOMAINS
    if not dest.exists() or dest.stat().st_size == 0:
        print(f"[generalization] fetching {_ECOD_DOMAINS} (~700 MB, slow host) ...")
        _download(_ECOD_URL, dest)
    return dest


def build_family_labels(force: bool = False) -> Dict[str, Dict[str, List[str]]]:
    """Label all 15,051 families with Pfam family/clan and SCOPe superfamily + ECOD H/T groups.

    Returns ``{family: {"pfam", "pfam_clan", "scop_sf", "ecod_h", "ecod_t": [...]}}`` and caches it to
    ``cfg.ANNOTATION_DIR/family_labels.json``. Empty lists mean "no annotation under that scheme".
    Structural coverage differs (ECOD ~90%, SCOPe ~38%); the eval set carries few novel folds under
    either (~15 novel superfamilies), so the finer Pfam-family split (81 novel) is the powered test.
    """
    cache = Path(cfg.ANNOTATION_DIR) / "family_labels.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text())

    ensure_annotations()
    pfam = _sifts_pfam_by_chain()
    clan = _pfam_clan_map()
    scop = _scope_sf_by_chain()
    ecod_h, ecod_t = _ecod_groups_by_chain()

    labels: Dict[str, Dict[str, List[str]]] = {}
    for family in train_families() + eval_families():
        pc = parse_family(family)
        pf = sorted(pfam.get(pc, set()))
        labels[family] = {
            "pfam": pf,
            "pfam_clan": sorted({clan.get(x, x) for x in pf}),
            "scop_sf": sorted(scop.get(pc, set())),
            "ecod_h": sorted(ecod_h.get(pc, set())),
            "ecod_t": sorted(ecod_t.get(pc, set())),
        }
    cache.write_text(json.dumps(labels))
    return labels


# ======================================================================================
# novelty partition
# ======================================================================================
def _labels_for(labels: Dict[str, Dict[str, List[str]]], families: Sequence[str], key: str) -> Dict[str, set]:
    return {f: set(labels[f][key]) for f in families}


def partition(scheme: str, labels: Optional[Dict] = None) -> Dict[str, set]:
    """Split eval families into novel / seen / unlabeled under ``scheme``.

    ``novel`` = eval family whose (non-empty) label set is disjoint from every training label.
    """
    if scheme not in SCHEMES:
        raise KeyError(f"Unknown scheme {scheme!r}; choose from {sorted(SCHEMES)}.")
    key = SCHEMES[scheme]
    labels = labels or build_family_labels()

    train_lab = _labels_for(labels, train_families(), key)
    train_vocab = set().union(*train_lab.values()) if train_lab else set()

    eval_lab = _labels_for(labels, eval_families(), key)
    novel, seen, unlabeled = set(), set(), set()
    for fam, lab in eval_lab.items():
        if not lab:
            unlabeled.add(fam)
        elif lab.isdisjoint(train_vocab):
            novel.add(fam)
        else:
            seen.add(fam)
    return {"novel": novel, "seen": seen, "unlabeled": unlabeled, "train_vocab": train_vocab}


def novelty_table(labels: Optional[Dict] = None) -> pd.DataFrame:
    """One row per eval family: novelty flag per scheme (bool, or NA if unlabeled) + covariates."""
    labels = labels or build_family_labels()
    rows = {}
    parts = {s: partition(s, labels) for s in SCHEMES}
    cov = family_covariates(eval_families())
    for fam in eval_families():
        row = {}
        for scheme, part in parts.items():
            if fam in part["novel"]:
                row[scheme] = True
            elif fam in part["seen"]:
                row[scheme] = False
            else:
                row[scheme] = pd.NA
        row.update(cov.get(fam, {}))
        rows[fam] = row
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "family"
    return df


# ======================================================================================
# covariates (for matched comparison / confound reporting)
# ======================================================================================
def _read_a3m_query_and_depth(family: str) -> Tuple[str, int]:
    """Return (ungapped uppercase seq1, number of sequences) from the family's a3m."""
    path = Path(cfg.INPUT_A3M_DIR) / f"{family}.a3m"
    query = None
    depth = 0
    with open(cfg.require(path)) as fh:
        seq_chunks: List[str] = []
        for line in fh:
            if line.startswith(">"):
                depth += 1
                if depth == 1:
                    seq_chunks = []
                elif depth == 2:
                    query = "".join(seq_chunks)
            elif depth == 1:
                seq_chunks.append(line.strip())
        if query is None:  # single-sequence a3m
            query = "".join(seq_chunks)
    # seq1 is the query: uppercase, no insertions/gaps.
    return re.sub(r"[^A-Za-z]", "", query).upper(), depth


def family_covariates(families: Sequence[str]) -> Dict[str, Dict[str, int]]:
    """Per-family confound descriptors: sequence length and original MSA depth.

    (Simulations use a fixed 512-leaf tree for every family, so tree depth is not a confound;
    sequence length is the covariate that varies and is matched on downstream.)
    """
    out = {}
    for fam in families:
        try:
            seq, depth = _read_a3m_query_and_depth(fam)
            out[fam] = {"length": len(seq), "msa_depth": depth}
        except FileNotFoundError:
            out[fam] = {"length": np.nan, "msa_depth": np.nan}
    return out


# ======================================================================================
# domain level: Pfam domains on seq1 via pyhmmer
# ======================================================================================
def ensure_pfam_hmm() -> Path:
    """Fetch + gunzip ``Pfam-A.hmm`` (needed only for the domain-level JSD). Returns its path."""
    d = Path(cfg.ANNOTATION_DIR)
    hmm = d / _PFAM_HMM
    if hmm.exists() and hmm.stat().st_size > 0:
        return hmm
    gz = d / _PFAM_HMM_GZ
    if not gz.exists() or gz.stat().st_size == 0:
        print(f"[generalization] fetching {_PFAM_HMM_GZ} (~400 MB) ...")
        _download(f"{_PFAM_BASE}/{_PFAM_HMM_GZ}", gz)
    print(f"[generalization] decompressing {_PFAM_HMM_GZ} ...")
    with gzip.open(gz, "rb") as fin, open(hmm, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    return hmm


def scan_pfam_domains(
    families: Sequence[str], cpus: int = 0, force: bool = False
) -> Dict[str, List[dict]]:
    """Locate Pfam domains on each family's ``seq1`` (Pfam gathering thresholds), with residue ranges.

    Returns ``{family: [{"acc","name","start","end","ievalue","bitscore"}, ...]}`` where start/end
    are 1-based positions in the *ungapped* seq1. Cached to ``pfam_domains_seq1.json``. Uses
    hmmsearch (HMMs as queries over the seq1 database) — the efficient direction — so the whole
    held-out set scans in one pass.
    """
    import pyhmmer

    cache = Path(cfg.ANNOTATION_DIR) / "pfam_domains_seq1.json"
    if cache.exists() and not force:
        cached = json.loads(cache.read_text())
        if all(f in cached for f in families):
            return {f: cached[f] for f in families}

    hmm_path = ensure_pfam_hmm()
    alphabet = pyhmmer.easel.Alphabet.amino()

    seqs = []
    for fam in families:
        seq, _ = _read_a3m_query_and_depth(fam)
        ts = pyhmmer.easel.TextSequence(name=fam.encode(), sequence=seq)
        seqs.append(ts.digitize(alphabet))
    seq_block = pyhmmer.easel.DigitalSequenceBlock(alphabet, seqs)

    def _s(x):  # pyhmmer returns bytes on some versions, str on others.
        if x is None:
            return None
        return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)

    results: Dict[str, List[dict]] = {f: [] for f in families}
    with pyhmmer.plan7.HMMFile(hmm_path) as hmm_file:
        for hits in pyhmmer.hmmer.hmmsearch(
            hmm_file, seq_block, bit_cutoffs="gathering", cpus=cpus
        ):
            name = _s(hits.query.name)
            acc = _s(hits.query.accession) or name
            for hit in hits:
                if not hit.included:
                    continue
                fam = _s(hit.name)
                for dom in hit.domains:
                    if not dom.included:
                        continue
                    results[fam].append(
                        {
                            "acc": acc.split(".")[0],  # strip Pfam version suffix
                            "name": name,
                            "start": int(dom.env_from),
                            "end": int(dom.env_to),
                            "ievalue": float(dom.i_evalue),
                            "bitscore": float(dom.score),
                        }
                    )

    merged = {}
    if cache.exists():
        merged = json.loads(cache.read_text())
    merged.update(results)
    cache.write_text(json.dumps(merged))
    return results


def hmm_pfam_vocab(families: Sequence[str]) -> set:
    """Pfam accessions found on these families' seq1s by pyhmmer (gathering thresholds).

    Used to build a training vocabulary from the *same* method as the held-out domain calls, so
    the domain-level novelty test is not confounded by SIFTS vs hmmscan disagreeing on coverage.
    """
    domains = scan_pfam_domains(families)
    return {h["acc"] for hits in domains.values() for h in hits}


def seq1_residue_to_column(family: str, msa_dir: Optional[str] = None) -> Dict[int, int]:
    """Map 1-based seq1 residue index -> 1-based alignment column in the reference frame.

    The reference alignment (``mafft_add/old_sequences``) contains seq1; its non-gap positions,
    walked left to right, are seq1 residues 1..L. Columns are numbered as ``paper.jsd`` numbers
    them (1-based), so the result indexes directly into that module's per-site tables.
    """
    from protevo.utils import read_msa, gap_character

    msa_dir = msa_dir or str(Path(cfg.MAFFT_ADD_DIR) / "old_sequences")
    msa = read_msa(os.path.join(msa_dir, f"{family}.txt"))
    if "seq1" not in msa:
        raise KeyError(f"seq1 not found in reference alignment for {family}.")
    aligned = msa["seq1"]
    mapping: Dict[int, int] = {}
    resid = 0
    for col, ch in enumerate(aligned, start=1):
        if ch != gap_character:
            resid += 1
            mapping[resid] = col
    return mapping


# ======================================================================================
# fast per-family JSD (numpy reimplementation of the paper.jsd frequency step)
# ======================================================================================
# paper.jsd computes site frequencies with a per-column pandas value_counts, which is ~2 s/family
# and makes the 553-family recompute take tens of minutes. This is the identical computation with a
# vectorized numpy frequency count (validated bit-for-bit against paper.jsd), turning the whole
# recompute into ~1 minute single-threaded. Everything else (split filtering, conserved-site union,
# gap handling, the JS distance) reuses paper.jsd / paper.splits verbatim so the two cannot drift.
def _fast_site_frequencies(seqs: List[str], vocab_codes: "np.ndarray") -> "np.ndarray":
    """(len(VOCAB), n_cols) column-normalized frequencies, matching ``paper.jsd.site_frequencies``."""
    if not seqs:
        raise ValueError("Cannot compute site frequencies from an empty MSA.")
    mat = np.frombuffer("".join(seqs).encode("ascii"), dtype=np.uint8).reshape(len(seqs), -1)
    counts = (mat[None, :, :] == vocab_codes[:, None, None]).sum(axis=1).astype(float)
    return counts / len(seqs)  # value_counts(normalize=True) divides by the column height


def fast_family_jsd(
    msa_dirs: Dict[str, str],
    family: str,
    tree_split: Dict[str, List[str]],
    threshold: float,
) -> Tuple[Dict[str, float], "pd.DataFrame"]:
    """Drop-in fast equivalent of ``paper.jsd.family_jsd`` (same return shape and values)."""
    from protevo.utils import read_msa
    from paper.jsd import RESIDUES, VOCAB, jsd, msa_path
    from paper.splits import (
        REAL, REAL_OTHER_SPLIT, SPLIT_A, SPLIT_B, filter_msa_based_on_split,
    )

    if REAL not in msa_dirs:
        raise KeyError(f"msa_dirs must contain a {REAL!r} entry; got {sorted(msa_dirs)}.")
    vocab_codes = np.frombuffer("".join(VOCAB).encode("ascii"), dtype=np.uint8)
    n_res = len(RESIDUES)  # residue rows come first in VOCAB; the gap is last

    raw = {model: read_msa(msa_path(d, family)) for model, d in msa_dirs.items()}
    raw[REAL_OTHER_SPLIT] = read_msa(msa_path(msa_dirs[REAL], family))
    filtered = {
        model: filter_msa_based_on_split(
            msa, tree_split, SPLIT_B if model == REAL_OTHER_SPLIT else SPLIT_A
        )
        for model, msa in raw.items()
    }

    freqs = {m: _fast_site_frequencies(list(msa.values()), vocab_codes) for m, msa in filtered.items()}

    def conserved(f):  # columns where some residue (excluding gap) exceeds the threshold
        return set(np.where(f[:n_res].max(axis=0) > threshold)[0].tolist())

    sites = sorted(conserved(freqs[REAL]) | conserved(freqs[REAL_OTHER_SPLIT]))
    if not sites:
        raise ValueError(f"No conserved sites for {family} at threshold {threshold}.")

    def distributions(f):  # drop gap row, renormalize each column over residues, 0 for all-gap cols
        res = f[:n_res][:, sites]
        col = res.sum(axis=0, keepdims=True)
        return np.divide(res, col, out=np.zeros_like(res), where=col != 0)

    dists = {m: distributions(f) for m, f in freqs.items()}
    real = dists[REAL]
    per_site = pd.DataFrame(
        {
            model: [jsd(real[:, i], dist[:, i]) for i in range(len(sites))]
            for model, dist in dists.items()
            if model != REAL
        },
        index=[str(s + 1) for s in sites],  # 1-based column labels, matching paper.jsd
    )
    return per_site.mean().to_dict(), per_site


def collect_family_jsd_fast(
    families: Sequence[str], msa_dirs: Dict[str, str], threshold: float, want_persite: bool = False,
) -> Tuple[Dict[str, dict], Dict[str, "pd.DataFrame"], int]:
    """Serial fast per-family JSD over ``families``. Returns ``(means, per_sites, n_skipped)``."""
    from paper.splits import generate_tree_split

    tree_dir = str(cfg.require(cfg.TREE_DIR))
    means, per_sites, skipped = {}, {}, 0
    for family in families:
        try:
            mean, per_site = fast_family_jsd(
                msa_dirs, family, generate_tree_split(tree_dir, family), threshold
            )
        except (FileNotFoundError, ValueError, KeyError):
            skipped += 1
            continue
        means[family] = mean
        if want_persite:
            per_sites[family] = per_site
    return means, per_sites, skipped


# ======================================================================================
# statistics + plotting helpers (shared by the per-metric scripts)
# ======================================================================================
def mann_whitney(novel: Sequence[float], seen: Sequence[float]) -> dict:
    """Two-sided Mann-Whitney U plus Cliff's delta effect size (novel relative to seen)."""
    from scipy.stats import mannwhitneyu

    a = np.asarray([x for x in novel if np.isfinite(x)], dtype=float)
    b = np.asarray([x for x in seen if np.isfinite(x)], dtype=float)
    if len(a) == 0 or len(b) == 0:
        return {
            "n_novel": len(a), "n_seen": len(b),
            "median_novel": float(np.median(a)) if len(a) else np.nan,
            "median_seen": float(np.median(b)) if len(b) else np.nan,
            "U": np.nan, "p": np.nan, "cliffs_delta": np.nan,
        }
    U, p = mannwhitneyu(a, b, alternative="two-sided")
    # Cliff's delta = 2U/(n1 n2) - 1, using this test's U for `a` (novel).
    delta = 2.0 * U / (len(a) * len(b)) - 1.0
    return {
        "n_novel": len(a),
        "n_seen": len(b),
        "median_novel": float(np.median(a)),
        "median_seen": float(np.median(b)),
        "U": float(U),
        "p": float(p),
        "cliffs_delta": float(delta),
    }


def matched_subsample(
    df: pd.DataFrame,
    novel_mask: pd.Series,
    covariate: str = "length",
    tolerance: float = 0.25,
    seed: int = 0,
) -> pd.DataFrame:
    """Greedy 1:1 nearest-covariate matching of seen families to novel families.

    Controls for the confound that novel families might be intrinsically harder (e.g. longer).
    Returns the novel rows plus one matched seen row each (within ``tolerance`` relative distance);
    rows with a missing covariate are dropped from matching. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    novel = df[novel_mask & df[covariate].notna()]
    seen = df[~novel_mask & df[covariate].notna()].copy()
    picked = []
    available = seen.index.tolist()
    for fam, row in novel.sample(frac=1.0, random_state=seed).iterrows():
        if not available:
            break
        target = row[covariate]
        dists = {i: abs(seen.at[i, covariate] - target) for i in available}
        best = min(dists, key=dists.get)
        if target > 0 and dists[best] / target > tolerance:
            continue
        picked.append(best)
        available.remove(best)
    keep = list(novel.index) + picked
    return df.loc[keep]


def strata_of(df: pd.DataFrame, family_col: str, scheme: str) -> pd.Series:
    """Label each row of ``df`` 'novel'/'seen'/'unlabeled' by its family, under ``scheme``."""
    part = partition(scheme)
    lut = {f: "novel" for f in part["novel"]}
    lut.update({f: "seen" for f in part["seen"]})
    lut.update({f: "unlabeled" for f in part["unlabeled"]})
    return df[family_col].map(lut)


def report_stratified(
    df_long: pd.DataFrame,
    value_col: str,
    out_stem: str,
    *,
    schemes: Sequence[str] = ("pfam_family", "ecod_hgroup", "scop_superfamily"),
    focus_models: Optional[Sequence[str]] = None,
    family_col: str = "family",
    model_col: str = "model",
    value_label: str = "",
    out_dir: Optional[str] = None,
    match_covariate: str = "length",
) -> pd.DataFrame:
    """Test novel vs seen for every model under each scheme; write a stats table and a boxplot.

    ``df_long`` is one row per (family, model) with a numeric ``value_col``. For each scheme and
    model we run a Mann-Whitney U (novel vs seen) plus a covariate-matched version (seen families
    matched to novel families on ``match_covariate``), reporting medians, p-values and Cliff's
    delta. The plot shows ``focus_models`` (default: PEINT + the Real baseline + LG+S256) with
    novel/seen side by side, under the first scheme — the "does PEINT's novel/seen gap look like
    the empirical baseline's gap" view. Returns the stats DataFrame.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from paper.plot_style import _set_publication_style
    from paper.splits import REAL_OTHER_SPLIT

    out_dir = Path(out_dir or cfg.GENERALIZATION_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    cov = family_covariates(sorted(df_long[family_col].unique()))
    length = {f: c.get(match_covariate, np.nan) for f, c in cov.items()}

    records = []
    strata_by_scheme = {}
    for scheme in schemes:
        strat = strata_of(df_long, family_col, scheme)
        strata_by_scheme[scheme] = strat
        tagged = df_long.assign(_stratum=strat.values)
        for model, g in tagged.groupby(model_col):
            novel = g.loc[g["_stratum"] == "novel", value_col]
            seen = g.loc[g["_stratum"] == "seen", value_col]
            rec = {"scheme": scheme, "model": model, **mann_whitney(novel.values, seen.values)}
            # covariate-matched re-test
            per_fam = (
                g[g["_stratum"].isin(["novel", "seen"])]
                .groupby(family_col)
                .agg(value=(value_col, "mean"), stratum=("_stratum", "first"))
            )
            per_fam["length"] = per_fam.index.map(length)
            matched = matched_subsample(per_fam, per_fam["stratum"] == "novel", covariate="length")
            m = mann_whitney(
                matched.loc[matched["stratum"] == "novel", "value"].values,
                matched.loc[matched["stratum"] == "seen", "value"].values,
            )
            rec["p_matched"] = m["p"]
            rec["n_novel_matched"] = m["n_novel"]
            rec["n_seen_matched"] = m["n_seen"]
            records.append(rec)
    stats = pd.DataFrame(records)
    stats.to_csv(out_dir / f"{out_stem}_stats.csv", index=False)

    # --- figure: focus models, novel vs seen, under the primary scheme ---
    _set_publication_style()
    primary = schemes[0]
    focus_models = list(focus_models or ["PEINT (Progressive)", REAL_OTHER_SPLIT, "LG+S256"])
    tagged = df_long.assign(_stratum=strata_by_scheme[primary].values)
    plot_df = tagged[
        tagged["_stratum"].isin(["novel", "seen"]) & tagged[model_col].isin(focus_models)
    ].copy()
    plot_df[model_col] = pd.Categorical(plot_df[model_col], categories=focus_models, ordered=True)

    fig, ax = plt.subplots(figsize=(1.6 * len(focus_models), 3))
    sns.boxplot(
        data=plot_df, x=model_col, y=value_col, hue="_stratum",
        hue_order=["seen", "novel"], palette={"seen": "#9ecae1", "novel": "#fb6a4a"},
        showfliers=False, width=0.7, linewidth=0.5, ax=ax,
    )
    sns.stripplot(
        data=plot_df, x=model_col, y=value_col, hue="_stratum",
        hue_order=["seen", "novel"], dodge=True, size=1.5, alpha=0.35,
        palette={"seen": "#3182bd", "novel": "#cb181d"}, ax=ax, legend=False,
    )
    ax.set_xlabel("")
    ax.set_ylabel(value_label or value_col, fontsize=9)
    ax.set_title(f"Novel vs seen ({primary.replace('_', ' ')})", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    handles, labels_ = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels_[:2], title="", fontsize=7, loc="best", frameon=False)
    sns.despine(ax=ax)
    fig.savefig(out_dir / f"{out_stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{out_stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    return stats


def main() -> None:
    """Build the labels for all 15,051 families and print the coverage / novelty summary."""
    labels = build_family_labels()
    cov = family_covariates(eval_families())
    print(f"Built labels for {len(labels)} families "
          f"(train {len(train_families())}, eval {len(eval_families())}).")
    print(f"\n{'scheme':<20}{'eval labeled':>14}{'novel':>8}{'seen':>8}")
    for scheme in SCHEMES:
        p = partition(scheme, labels)
        labeled = len(p["novel"]) + len(p["seen"])
        print(f"{scheme:<20}{labeled:>14}{len(p['novel']):>8}{len(p['seen']):>8}")
    lengths = [c["length"] for c in cov.values() if c["length"] == c["length"]]
    print(f"\neval seq length: median {int(np.median(lengths))}, range {int(min(lengths))}-{int(max(lengths))}")
    print(f"labels cached at {Path(cfg.ANNOTATION_DIR) / 'family_labels.json'}")


if __name__ == "__main__":
    main()
