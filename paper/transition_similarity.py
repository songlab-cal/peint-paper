"""DIAMOND search of held-out transition sequences against a database of every training sequence,
at full-chain and Pfam-domain granularity, to measure how close held-out data is to training.
"""

import gzip
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd

import paper_config as cfg

GAP = "-"

# A domain slice shorter than this after gap removal is a stub, not a domain: the row is mostly
# gaps across those columns. DIAMOND would also struggle to place it. Dropped rather than searched.
MIN_DOMAIN_AA = 20

# Fasta ids are "<family>|<idx>" (full) or "<family>|<idx>|<pfam_acc>|<start>-<end>" (domain).
# '|' cannot occur in a family name (``<pdb>_<n>_<chain>``) or a Pfam accession, so the split is
# unambiguous, and DIAMOND takes everything up to the first whitespace as the id.
ID_SEP = "|"

LEVELS = ("full", "domain")

# DIAMOND tabular fields, in the order requested on the command line.
BLAST_FIELDS = [
    "qseqid", "sseqid", "pident", "length", "nident",
    "qlen", "slen", "evalue", "bitscore",
]


# ======================================================================================
# reading transitions
# ======================================================================================
def transitions_path(family: str, split: str, root: Optional[Path] = None) -> Path:
    """Path to a family's aligned transition file. ``split`` is 'train' or 'test'."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    root = Path(root) if root is not None else Path(cfg.ALIGNED_TRANSITIONS_DIR)
    return root / f"{split}_transitions_dir" / f"{family}.txt"


def unique_aligned_sequences(path) -> List[str]:
    """Sorted unique gapped rows from a transitions file, pooling the x and y columns.

    Deduping is what removes the deliberate ``x,y,t`` / ``y,x,t`` duplication: both directions
    of a cherry contribute the same two sequences, so the union of the columns is the set of
    distinct sequences the file touches. Sorted so downstream sampling is reproducible.
    """
    seen = set()
    with open(path) as fh:
        header = fh.readline()
        if "transitions" not in header:
            raise ValueError(f"{path}: expected a '<n> transitions' header, got {header!r}")
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(" ")
            if len(parts) != 3:
                raise ValueError(f"{path}: expected 'x y t' rows, got {len(parts)} fields")
            seen.add(parts[0])
            seen.add(parts[1])
    return sorted(seen)


def seq1_length(family: str) -> int:
    """Ungapped length of the a3m query (``seq1``), i.e. the reference PDB chain."""
    with open(Path(cfg.INPUT_A3M_DIR) / f"{family}.a3m") as fh:
        fh.readline()                      # ">seq1"
        return len(fh.readline().strip())


def check_frame(family: str, ncols: int, domains: Sequence[dict], strict: bool = True) -> None:
    """Assert the aligned transition columns really are seq1 residues 1..L.

    Two independent checks: every cached Pfam domain end must fall inside the row (cheap, always
    on), and, when the a3m is available, the row width must equal the ungapped seq1 length. The
    domain slicing below is only correct if both hold.
    """
    for d in domains:
        if d["end"] > ncols:
            raise ValueError(
                f"{family}: Pfam domain {d['acc']} ends at seq1 residue {d['end']} but the "
                f"aligned transitions have only {ncols} columns -- the two are not in the same frame."
            )
    if not strict:
        return
    try:
        length = seq1_length(family)
    except FileNotFoundError:
        return                              # no a3m available (deposit tier); domain check stands
    if length != ncols:
        raise ValueError(
            f"{family}: aligned transitions have {ncols} columns but seq1 is {length} residues. "
            f"Column i is no longer seq1 residue i, so Pfam ranges cannot be sliced directly."
        )


# ======================================================================================
# fasta records
# ======================================================================================
def _sample(seqs: List[str], max_seqs: int, family: str, seed: int) -> List[str]:
    """Deterministic per-family subsample. ``max_seqs<=0`` keeps everything."""
    if max_seqs <= 0 or len(seqs) <= max_seqs:
        return seqs
    rng = random.Random(f"{seed}:{family}")
    return sorted(rng.sample(seqs, max_seqs))


def family_records(
    family: str,
    split: str,
    level: str,
    domains: Sequence[dict],
    *,
    max_seqs: int = 0,
    seed: int = 0,
    root: Optional[Path] = None,
    strict_frame: bool = True,
) -> List[Tuple[str, str]]:
    """``(fasta_id, ungapped_sequence)`` for one family at one granularity.

    ``full`` yields one record per distinct aligned row; ``domain`` yields one record per
    (row, Pfam domain), sliced by the domain's seq1 residue range read as column indices.
    Subsampling is applied to the *rows*, before slicing, so a family's full-length and domain
    records describe the same underlying sequences.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    path = transitions_path(family, split, root)
    seqs = unique_aligned_sequences(path)
    if not seqs:
        return []
    check_frame(family, len(seqs[0]), domains, strict=strict_frame)
    seqs = _sample(seqs, max_seqs, family, seed)

    out: List[Tuple[str, str]] = []
    if level == "full":
        for i, aligned in enumerate(seqs):
            plain = aligned.replace(GAP, "")
            if plain:
                out.append((f"{family}{ID_SEP}{i}", plain))
        return out

    for i, aligned in enumerate(seqs):
        for d in domains:
            # Cached ranges are 1-based inclusive seq1 residues == 1-based inclusive columns.
            plain = aligned[d["start"] - 1:d["end"]].replace(GAP, "")
            if len(plain) >= MIN_DOMAIN_AA:
                out.append(
                    (f"{family}{ID_SEP}{i}{ID_SEP}{d['acc']}{ID_SEP}{d['start']}-{d['end']}", plain)
                )
    return out


def parse_ids(ids: pd.Series) -> pd.DataFrame:
    """Vectorized fasta-id -> (family, seq_idx, pfam_acc) for a whole column of hits."""
    parts = ids.str.split(ID_SEP, expand=True)
    out = pd.DataFrame({"family": parts[0]})
    out["seq_idx"] = pd.to_numeric(parts[1], errors="coerce").astype("Int32")
    out["pfam_acc"] = parts[2] if parts.shape[1] > 2 else pd.NA
    return out


# ======================================================================================
# fasta writing
# ======================================================================================
def _write_chunk(handle, records: Iterable[Tuple[str, str]], width: int = 0) -> int:
    n = 0
    for sid, seq in records:
        handle.write(f">{sid}\n{seq}\n")
        n += 1
    return n


def _worker(args):
    family, split, level, domains, max_seqs, seed, root, strict = args
    try:
        return family, family_records(
            family, split, level, domains,
            max_seqs=max_seqs, seed=seed, root=root, strict_frame=strict,
        ), None
    except (FileNotFoundError, ValueError) as exc:
        return family, [], f"{type(exc).__name__}: {exc}"


def write_fasta(
    out_path,
    families: Sequence[str],
    split: str,
    level: str,
    domains_by_family: Dict[str, List[dict]],
    *,
    max_seqs: int = 0,
    seed: int = 0,
    workers: int = 1,
    root: Optional[Path] = None,
    strict_frame: bool = True,
    progress_every: int = 2000,
) -> Dict[str, int]:
    """Write one fasta for a (family set, split, level). Returns counts + skipped families.

    Families are read in parallel and written by the parent in the order given, so the file is
    byte-identical across runs with the same inputs.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".partial")
    jobs = [
        (f, split, level, domains_by_family.get(f, []), max_seqs, seed, root, strict_frame)
        for f in families
    ]
    n_seq = 0
    n_fam = 0
    skipped: List[str] = []

    with open(tmp, "w") as fh:
        if workers > 1:
            from multiprocessing import Pool

            with Pool(workers) as pool:
                it = pool.imap(_worker, jobs, chunksize=16)
                for k, (family, records, err) in enumerate(it, 1):
                    if err:
                        skipped.append(f"{family}: {err}")
                        continue
                    if records:
                        n_seq += _write_chunk(fh, records)
                        n_fam += 1
                    if progress_every and k % progress_every == 0:
                        print(f"    {k}/{len(jobs)} families, {n_seq:,} sequences", flush=True)
        else:
            for k, job in enumerate(jobs, 1):
                family, records, err = _worker(job)
                if err:
                    skipped.append(f"{family}: {err}")
                    continue
                if records:
                    n_seq += _write_chunk(fh, records)
                    n_fam += 1
                if progress_every and k % progress_every == 0:
                    print(f"    {k}/{len(jobs)} families, {n_seq:,} sequences", flush=True)

    os.replace(tmp, out_path)
    if skipped:
        print(f"    WARNING: skipped {len(skipped)} families, e.g. {skipped[:3]}")
    return {"sequences": n_seq, "families": n_fam, "skipped": skipped}


# ======================================================================================
# DIAMOND
# ======================================================================================
def diamond_binary() -> str:
    """The DIAMOND executable: ``PEINT_PAPER_DIAMOND`` if set, else whatever is on PATH."""
    explicit = os.environ.get("PEINT_PAPER_DIAMOND")
    if explicit:
        if not os.access(explicit, os.X_OK):
            raise FileNotFoundError(f"PEINT_PAPER_DIAMOND={explicit} is not executable.")
        return explicit
    found = shutil.which("diamond")
    if not found:
        raise FileNotFoundError(
            "DIAMOND not found. Put `diamond` on PATH or set PEINT_PAPER_DIAMOND."
        )
    return found


def _run(cmd: Sequence[str]) -> None:
    print("    $ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def makedb(fasta, db, threads: int = 8) -> Path:
    """Build a DIAMOND database. Skips the build if the ``.dmnd`` is already newer than the fasta."""
    fasta, db = Path(fasta), Path(db)
    dmnd = db.with_suffix(".dmnd")
    if dmnd.exists() and dmnd.stat().st_mtime >= fasta.stat().st_mtime:
        print(f"    {dmnd} is up to date, skipping makedb")
        return dmnd
    db.parent.mkdir(parents=True, exist_ok=True)
    _run([diamond_binary(), "makedb", "--in", fasta, "-d", db, "--threads", threads])
    return dmnd


def blastp(
    db,
    query,
    out,
    *,
    threads: int = 8,
    sensitivity: str = "very-sensitive",
    max_target_seqs: int = 6,
    evalue: float = 1e-3,
    block: Optional[float] = None,
    index_chunks: Optional[int] = None,
    tmpdir: Optional[str] = None,
    hit_membuf: bool = True,
) -> Path:
    """Search ``query`` against ``db``, writing gzipped tabular output to ``out``.

    ``hit_membuf`` keeps DIAMOND's intermediate seed hits in RAM instead of spilling them to
    ``tmpdir``. Those intermediates dwarf both the database and the output -- a ``--very-sensitive``
    pass over this database wrote 158 GB of them, against a 3 GB database and a final table of a
    few hundred MB, because sensitive mode sweeps 16 seed shapes and every seed hit is staged
    before ranking. Holding them in memory is the right trade here: the node has far more RAM
    than the scratch quota has free space.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".partial")
    cmd = [
        diamond_binary(), "blastp",
        "-d", Path(db).with_suffix(".dmnd"),
        "-q", query,
        "-o", tmp,
        "--outfmt", "6", *BLAST_FIELDS,
        "--max-target-seqs", max_target_seqs,
        "--evalue", evalue,
        "--threads", threads,
        f"--{sensitivity}",
        "--compress", "1",           # DIAMOND appends .gz to -o itself
    ]
    if hit_membuf:
        cmd += ["--hit-membuf"]
    if block is not None:
        cmd += ["-b", block]
    if index_chunks is not None:
        cmd += ["-c", index_chunks]
    if tmpdir:
        os.makedirs(tmpdir, exist_ok=True)
        cmd += ["--tmpdir", tmpdir]
    _run(cmd)
    os.replace(str(tmp) + ".gz", out)
    return out


# ======================================================================================
# reducing hits
# ======================================================================================
def reduce_hits(tsv_gz, chunksize: int = 2_000_000) -> pd.DataFrame:
    """Stream a DIAMOND tabular file down to the single best hit (by bitscore) per query.

    Alongside the local percent identity we record ``qcov`` and ``gident``. A local alignment
    can be 90% identical over 20 of 200 residues, and reporting ``pident`` alone would call that
    sequence 90% similar to training. ``gident = nident/qlen`` counts identities over the whole
    query, which is the honest number for "how much of this sequence already exists in training".
    ``qcov = length/qlen`` can exceed 100 because the alignment length counts gap columns.
    """
    best: Dict[str, tuple] = {}
    reader = pd.read_csv(
        tsv_gz, sep="\t", header=None, names=BLAST_FIELDS,
        dtype={"qseqid": str, "sseqid": str}, chunksize=chunksize,
    )
    for chunk in reader:
        top = chunk.loc[chunk.groupby("qseqid", sort=False)["bitscore"].idxmax()]
        for row in top.itertuples(index=False):
            cur = best.get(row.qseqid)
            if cur is None or row.bitscore > cur[0]:
                best[row.qseqid] = (
                    row.bitscore, row.sseqid, row.pident,
                    row.length, row.nident, row.qlen, row.evalue,
                )
    if not best:
        return pd.DataFrame(columns=["qseqid"])
    f = pd.DataFrame(
        [(q, *v) for q, v in best.items()],
        columns=["qseqid", "bitscore", "sseqid", "pident", "length", "nident", "qlen", "evalue"],
    )
    f["qcov"] = 100.0 * f["length"] / f["qlen"]
    f["gident"] = 100.0 * f["nident"] / f["qlen"]
    return f


def hits_table(tsv_gz, query_fasta) -> pd.DataFrame:
    """Per-query best-hit table, reindexed onto **every** submitted query.

    Queries that returned no hit survive as NaN rows. That matters: "no detectable homolog in
    the entire training set" is the strongest possible statement of difference, and dropping
    those rows would silently delete it from the distribution.
    """
    best = reduce_hits(tsv_gz)
    ids = [ln[1:].strip() for ln in _open_text(query_fasta) if ln.startswith(">")]
    frame = pd.DataFrame({"qseqid": ids})
    if not best.empty:
        frame = frame.merge(best, on="qseqid", how="left")
    meta = parse_ids(frame["qseqid"])
    return pd.concat([frame, meta], axis=1)


def _open_text(path) -> Iterator[str]:
    path = str(path)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            yield line
