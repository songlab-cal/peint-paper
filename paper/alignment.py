"""MAFFT alignment helpers for the paper benchmarks.

Ported from the model repo's ``peint/datasets/_msa.py``. MAFFT is a *benchmarking*
dependency (expected as ``mafft`` on PATH), not a dependency of the ``peint`` model
library. Only the three helpers actually used by the paper figures are kept here
(``run_mafft``, ``run_mafft_add``, ``sanitize_fastas``); the SeqKernel / patristic /
combine_msas / clean_msa / fsa helpers from the original module are intentionally dropped.
"""

import multiprocessing
import os
import re
import subprocess
import tempfile
from typing import List, Optional, Tuple, Dict

import tqdm

from peint import caching as peint_caching
from peint.caching import secure_parallel_output
from peint.utils import (
    get_process_args,
    read_msa,
    write_msa,
    amino_acids,
    ambiguous_mapping,
)


def _map_func_run_mafft(args):
    data_dir, families, extra_command_line_args, output_msa_dir = args
    mafft_path = 'mafft'  # should be on PATH, otherwise set to the full mafft path

    for family in families:
        input_sequence_path = os.path.join(data_dir, family + ".txt")
        output_sequence_path = os.path.join(output_msa_dir, family + ".txt")

        with open(output_sequence_path, "w") as output_file:
            mafft_cmd = [mafft_path, input_sequence_path]
            if extra_command_line_args:
                mafft_cmd.extend(extra_command_line_args)
            subprocess.run(mafft_cmd, stdout=output_file, stderr=subprocess.DEVNULL)

        secure_parallel_output(output_msa_dir, family)


@peint_caching.cached_parallel_computation(
    parallel_arg="families",
    output_dirs=["output_msa_dir"],
    exclude_args=["num_processes"],
    exclude_args_if_default=["extra_command_line_args"],
)
def run_mafft(
    data_dir: str,
    families: List[str],
    num_processes: int = 1,
    extra_command_line_args: Optional[str] = None,
    output_msa_dir: Optional[str] = None,
):
    """Runs MAFFT on the given set of alignments for the given families."""
    map_args = [
        [
            data_dir,
            get_process_args(process_rank, num_processes, families),
            extra_command_line_args,
            output_msa_dir,
        ]
        for process_rank in range(num_processes)
    ]

    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(tqdm.tqdm(pool.imap(_map_func_run_mafft, map_args), total=len(map_args)))
    else:
        list(tqdm.tqdm(map(_map_func_run_mafft, map_args), total=len(map_args)))


def _map_func_run_mafft_add(args):
    (existing_alignment_dir, new_sequences_dirs, new_sequences_names, families,
     extra_command_line_args, output_all_sequences_msa_dir,
     output_new_sequences_msa_dir, output_old_sequences_msa_dir) = args
    mafft_path = 'mafft'

    # Process multiple sequence directories that we want to MAFFT add if they exist
    with tempfile.TemporaryDirectory() as new_sequences_dir:

        # Create a set of combined sequences from all directories for each family, with a flag
        for family in families:
            msa = {}
            temp_msa_path = os.path.join(new_sequences_dir, family + ".txt")
            for i, directory in enumerate(new_sequences_dirs):
                input_msa_path = os.path.join(directory, family + ".txt")
                temp_msa = read_msa(input_msa_path)
                for k, v in temp_msa.items():
                    msa[f"{k}_{i}"] = v
            write_msa(msa, temp_msa_path)

            existing_alignment_path = os.path.join(existing_alignment_dir, family + ".txt")
            new_sequences_path = os.path.join(new_sequences_dir, family + ".txt")
            output_all_sequences_path = os.path.join(output_all_sequences_msa_dir, family + ".txt")
            output_new_sequences_path = os.path.join(output_new_sequences_msa_dir, family + ".txt")
            output_old_sequences_path = os.path.join(output_old_sequences_msa_dir, family + ".txt")

            with open(output_all_sequences_path, "w") as output_file:
                mafft_cmd = [mafft_path, "--add", new_sequences_path, existing_alignment_path]
                if extra_command_line_args:
                    if len(extra_command_line_args) == 1 and extra_command_line_args[0] == "--keeplength":
                        mafft_cmd = [mafft_path, "--add", new_sequences_path, "--keeplength", existing_alignment_path]
                    else:
                        mafft_cmd.extend(extra_command_line_args)
                subprocess.run(mafft_cmd, stdout=output_file, stderr=subprocess.DEVNULL)
            secure_parallel_output(output_all_sequences_msa_dir, family)

            # Remove the original sequences from the new-sequence-only alignment, dropping the "added" flag
            existing_msa = read_msa(existing_alignment_path)
            all_msa = read_msa(output_all_sequences_path)
            added_msa = {k: v for k, v in all_msa.items() if k not in existing_msa}
            with open(output_new_sequences_path, "w") as f:
                for seq_id, seq in added_msa.items():
                    match = re.match(r"(.+)_(\d+)", seq_id)
                    if match is None:
                        continue
                    seq_id = match.group(1)
                    f.write(f">{seq_id}\n{seq}\n")
            secure_parallel_output(output_new_sequences_msa_dir, family)

            # Remove the new sequences from the old-sequence-only alignment
            existing_msa = read_msa(existing_alignment_path)
            all_msa = read_msa(output_all_sequences_path)
            original_msa = {k: v for k, v in all_msa.items() if k in existing_msa}
            with open(output_old_sequences_path, "w") as f:
                for seq_id, seq in original_msa.items():
                    f.write(f">{seq_id}\n{seq}\n")
            secure_parallel_output(output_old_sequences_msa_dir, family)

            # If multiple new-sequence directories were combined, emit per-directory output too
            if len(new_sequences_dirs) > 1:
                base_out_dir = os.path.dirname(output_all_sequences_msa_dir)
                for i in range(len(new_sequences_dirs)):
                    outfolder_path = os.path.join(base_out_dir, f"{new_sequences_names[i]}_dir")
                    os.makedirs(outfolder_path, exist_ok=True)
                    out_path = os.path.join(outfolder_path, family + ".txt")
                    with open(out_path, "w") as f:
                        for seq_id, seq in added_msa.items():
                            match = re.match(rf"(.+)_({i})", seq_id)
                            if match is None:
                                continue
                            seq_id = match.group(1)
                            f.write(f">{seq_id}\n{seq}\n")
                    secure_parallel_output(outfolder_path, family)


@peint_caching.cached_parallel_computation(
    parallel_arg="families",
    output_dirs=[
        "output_all_sequences_msa_dir",
        "output_new_sequences_msa_dir",
        "output_old_sequences_msa_dir",
    ],
    exclude_args=["num_processes"],
    exclude_args_if_default=["extra_command_line_args"],
)
def run_mafft_add(
    existing_alignment_dir: str,
    new_sequences_dirs: Tuple[str],
    new_sequences_names: Tuple[str],
    families: List[str],
    num_processes: int = 1,
    extra_command_line_args: Optional[str] = None,
    output_all_sequences_msa_dir: Optional[str] = None,
    output_new_sequences_msa_dir: Optional[str] = None,
    output_old_sequences_msa_dir: Optional[str] = None,
):
    """Runs MAFFT --add to align new sequences relative to an existing alignment.

    ``new_sequences_dirs`` should contain **unaligned** sequences in fasta format.
    """
    map_args = [
        [
            existing_alignment_dir,
            new_sequences_dirs,
            new_sequences_names,
            get_process_args(process_rank, num_processes, families),
            extra_command_line_args,
            output_all_sequences_msa_dir,
            output_new_sequences_msa_dir,
            output_old_sequences_msa_dir,
        ]
        for process_rank in range(num_processes)
    ]

    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(tqdm.tqdm(pool.imap(_map_func_run_mafft_add, map_args), total=len(map_args)))
    else:
        list(tqdm.tqdm(map(_map_func_run_mafft_add, map_args), total=len(map_args)))


def _map_func_clean_msa(args):
    data_dir, families, chars_to_exclude, output_msa_dir = args

    for family in families:
        msa = read_msa(os.path.join(data_dir, family + ".txt"))
        msa = {
            seq_id: "".join(c for c in seq if c not in chars_to_exclude)
            for seq_id, seq in msa.items()
        }
        write_msa(msa, os.path.join(output_msa_dir, family + ".txt"))
        secure_parallel_output(output_msa_dir, family)


@peint_caching.cached_parallel_computation(
    parallel_arg="families",
    output_dirs=["output_msa_dir"],
    exclude_args=["num_processes"],
)
def clean_msa(
    data_dir: str,
    families: List[str],
    chars_to_exclude: Tuple[str] = ('-'),
    num_processes: int = 1,
    output_msa_dir: Optional[str] = None,
):
    """Strip unwanted characters (gaps, ambiguous codes) from each family's MSA.

    Used to turn an alignment back into raw sequences before re-aligning with MAFFT.
    """
    if len(chars_to_exclude) == 0:
        raise ValueError(f"chars_to_exclude must be nonempty! Received {chars_to_exclude}")

    map_args = [
        [
            data_dir,
            get_process_args(process_rank, num_processes, families),
            chars_to_exclude,
            output_msa_dir,
        ]
        for process_rank in range(num_processes)
    ]

    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(tqdm.tqdm(pool.imap(_map_func_clean_msa, map_args), total=len(map_args)))
    else:
        list(tqdm.tqdm(map(_map_func_clean_msa, map_args), total=len(map_args)))


def sanitize_sequences(sequences: Dict[str, str]) -> Dict[str, str]:
    """Remove ambiguous characters from an MSA / fasta (maps ambiguous codes, drops non-AA)."""
    cleaned_sequences = {}
    for seq_id, sequence in sequences.items():
        cleaned = []
        for char in sequence.upper():
            if char in ambiguous_mapping:
                cleaned.append(ambiguous_mapping[char])
            elif char in amino_acids:
                cleaned.append(char)
            else:
                continue  # skip characters that are not amino acids or ambiguous codes
        cleaned_sequences[seq_id] = ''.join(cleaned)
    return cleaned_sequences


def _map_func_sanitize_fastas(args):
    data_dir, families, output_sequences_dir = args
    for family in families:
        input_path = os.path.join(data_dir, family + ".txt")
        output_path = os.path.join(output_sequences_dir, family + ".txt")
        sequences = read_msa(input_path)
        sanitized_sequences = sanitize_sequences(sequences)
        write_msa(sanitized_sequences, output_path)
        secure_parallel_output(output_sequences_dir, family)


@peint_caching.cached_parallel_computation(
    parallel_arg="families",
    output_dirs=["output_sequences_dir"],
    exclude_args=["num_processes"],
)
def sanitize_fastas(
    data_dir: str,
    families: List[str],
    num_processes: int = 1,
    output_sequences_dir: Optional[str] = None,
):
    """Sanitize the sequences in ``data_dir`` by removing ambiguous characters."""
    map_args = [
        [data_dir, get_process_args(process_rank, num_processes, families), output_sequences_dir]
        for process_rank in range(num_processes)
    ]
    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(tqdm.tqdm(pool.imap(_map_func_sanitize_fastas, map_args), total=len(map_args)))
    else:
        list(tqdm.tqdm(map(_map_func_sanitize_fastas, map_args), total=len(map_args)))
