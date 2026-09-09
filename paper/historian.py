import glob
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tempfile
import re
import json
import multiprocessing
from collections import defaultdict
import logging
from dataclasses import dataclass

from Bio import SeqIO
from tqdm import tqdm
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
from ete3 import Tree

from peint import caching as peint_caching
from peint.caching import secure_parallel_output
from peint.io import read_tree, write_msa, write_tree
from paper.alignment import run_mafft
from peint.datasets._datasets import (
    find_optimal_edge_split,
    split_tree_on_edge
)
from peint.utils import get_process_args, read_msa


#logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EvolutionaryEvent:
    """Represents a single evolutionary event"""
    sequence_id: str
    branch_length: float
    position: int
    event_type: str  # 'substitution', 'insertion', 'deletion'
    ancestral_residue: str
    evolved_residue: str
    node_pair: Tuple[str, str] = None  # Optional, for pairwise events

def parse_sequence_name_and_time(seq_name: str) -> Tuple[str, float]:
    """
    Parse sequence name to extract base name and evolutionary time
    
    Example: 'family-sample0-0_05' -> ('family-sample0-0', 0.05)
    """
    # Split by underscore and get the last part as time
    parts = seq_name.split('-')
    if len(parts) >= 2:
        time_str = parts[-1]
        base_name = '_'.join(parts[:-1])
        try:
            # Replace underscore with decimal point for time
            time_value = round(float(time_str.replace('_', '.')), 2)
            return base_name, time_value
        except ValueError:
            # If parsing fails, assume no time component
            return seq_name, 0.0
    return seq_name, 0.0

def create_temp_files(seq1_name: str, seq1: str, seq2_name: str, seq2: str, 
                     branch_length: float, temp_dir: Path) -> Tuple[str, str]:
    """
    Create temporary FASTA and Newick files for pairwise analysis
    
    Returns:
        Tuple of (fasta_file_path, newick_file_path)
    """
    # Create FASTA file
    fasta_file = temp_dir / f"pair_{seq1_name}_{seq2_name}.fasta"
    records = [
        SeqRecord(Seq(seq1), id=seq1_name),
        SeqRecord(Seq(seq2), id=seq2_name)
    ]
    with open(fasta_file, 'w') as f:
        SeqIO.write(records, f, "fasta")
    #Dummy newick
    newick_file = temp_dir / f"pair_{seq1_name}_{seq2_name}.newick"
    internal_node = f"node_{seq1_name}_{seq2_name}"
    newick_tree = f"({seq1_name}:{branch_length},{seq2_name}:{branch_length}){internal_node};\n"
    
    logger.info(f"Created temp files for {seq1_name} vs {seq2_name} (branch_length={branch_length})")
    logger.info(f"newick tree: {newick_tree.strip()}")

    with open(newick_file, 'w') as f:
        f.write(newick_tree)
    
    return str(fasta_file), str(newick_file)

def run_historian_pairwise(fasta_file: str, newick_file: str, 
                          historian_path: str = "historian") -> str:
    """
    Run Historian reconstruction on pairwise sequences
    
    Returns:
        Stockholm format alignment as string
    """
    cmd = [
        historian_path,
        "recon",
        "-seqs", fasta_file,
        "-tree", newick_file,
        "-output", "stockholm"
    ]
        
    try:
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            check=True
        )
        
        if result.stderr:
            logger.debug(f"Historian stderr: {result.stderr}")
            
        return result.stdout
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Historian failed: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        raise RuntimeError(f"Historian failed with error: {e.stderr.strip()}")

def parse_stockholm_alignment(stockholm_text: str) -> Dict[str, str]:
    """
    Parse Stockholm format alignment text
    
    Returns:
        Dict mapping sequence_id -> aligned_sequence
    """
    sequences = {}
    
    for line in stockholm_text.split('\n'):
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('//'):
            parts = line.split()
            if len(parts) >= 2:
                seq_id = parts[0]
                sequence = parts[1]
                
                if seq_id in sequences:
                    sequences[seq_id] += sequence
                else:
                    sequences[seq_id] = sequence
    
    return sequences

def extract_pairwise_events(alignment: Dict[str, str], 
                            evolved_seq_name: str, 
                           root_seq_name: str,
                           branch_length: float) -> List[EvolutionaryEvent]:
    """
    Extract evolutionary events between root and evolved sequence
    
    Args:
        alignment: Dict with aligned sequences
        evolved_seq_name: Name of evolved sequence
        root_seq_name: Name of root/ancestral sequence  
        branch_length: Branch length between sequences
        
    Returns:
        List of evolutionary events
    """
    if evolved_seq_name not in alignment or root_seq_name not in alignment:
        logger.warning(f"Missing sequences in alignment: {evolved_seq_name}, {root_seq_name}")
        return []
    
    evolved_seq = alignment[evolved_seq_name]
    root_seq = alignment[root_seq_name]
    
    events = []
    position = 0  # Position in unaligned sequence root
    
    for i in range(min(len(evolved_seq), len(root_seq))):
        root_char = root_seq[i]
        evolved_char = evolved_seq[i]
        
        if root_char != evolved_char:
            if root_char != '-' and evolved_char != '-':
                # Substitution
                events.append(EvolutionaryEvent(
                    sequence_id=evolved_seq_name,
                    branch_length=branch_length,
                    position=position,
                    event_type='substitution',
                    ancestral_residue=root_char,
                    evolved_residue=evolved_char
                ))
            elif root_char != '-' and evolved_char == '-':
                # Deletion
                events.append(EvolutionaryEvent(
                    sequence_id=evolved_seq_name,
                    branch_length=branch_length,
                    position=position,
                    event_type='deletion',
                    ancestral_residue=root_char,
                    evolved_residue='-'
                ))
            elif root_char == '-' and evolved_char != '-':
                # Insertion
                events.append(EvolutionaryEvent(
                    sequence_id=evolved_seq_name,
                    branch_length=branch_length,
                    position=position,
                    event_type='insertion',
                    ancestral_residue='-',
                    evolved_residue=evolved_char
                ))
        
        # Only increment position for non-gap characters in root
        if root_char != '-':
            position += 1
    
    return events

def analyze_star_topology(sequences: Dict[str, str], 
                         root_sequence_name: str = "initial_sequence",
                         historian_path: str = "historian") -> pd.DataFrame:
    """
    Analyze star topology by running pairwise comparisons
    
    Args:
        sequences: List of (sequence_name, sequence) tuples
        root_sequence_name: Name of the root/initial sequence
        output_dir: Directory for output files
        historian_path: Path to Historian executable
        
    Returns:
        DataFrame with evolutionary events
    """
    # output_path = Path(output_dir)
    # output_path.mkdir(parents=True, exist_ok=True)
    
    if root_sequence_name not in sequences:
        raise ValueError(f"Root sequence '{root_sequence_name}' not found in sequences")
    
    root_sequence = sequences[root_sequence_name]
    
    # Collect all events
    all_events = []
    
    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        for seq_name, sequence in sequences.items():
            if seq_name == root_sequence_name:
                continue  # Skip root sequence
            
            logger.info(f"Analyzing {seq_name} vs {root_sequence_name}")
            
            # Parse branch length from sequence name
            base_name, branch_length = parse_sequence_name_and_time(seq_name)
            
            # Create temporary files
            fasta_file, newick_file = create_temp_files(
                seq_name, sequence, 
                root_sequence_name, root_sequence,
                branch_length, temp_path
            )
            
            try:
                # Run Historian
                stockholm_output = run_historian_pairwise(fasta_file, newick_file, historian_path)
                
                # Parse alignment
                alignment = parse_stockholm_alignment(stockholm_output)
                
                # Extract events
                events = extract_pairwise_events(alignment, seq_name, root_sequence_name, branch_length)
                all_events.extend(events)
                
                logger.info(f"Found {len(events)} events for {seq_name}")
                
            except Exception as e:
                logger.error(f"Failed to analyze {seq_name}: {e}")
                continue
    
    # Convert to DataFrame
    if all_events:
        df_data = []
        for event in all_events:
            df_data.append({
                'sequence_id': event.sequence_id,
                'branch_length': event.branch_length,
                'position': event.position,
                'event_type': event.event_type,
                'ancestral_residue': event.ancestral_residue,
                'evolved_residue': event.evolved_residue
            })
        
        df = pd.DataFrame(df_data)
        
        # Save to CSV
        # output_file = output_path / "evolutionary_events.csv"
        # df.to_csv(output_file, index=False)
        # logger.info(f"Saved {len(df)} events to {output_file}")
        
        return df
    else:
        logger.warning("No evolutionary events found")
        return pd.DataFrame()

def create_summary_statistics(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary statistics for evolutionary events
    """
    if events_df.empty:
        return pd.DataFrame()
    
    summary = events_df.groupby(['sequence_id', 'branch_length']).agg({
        'event_type': ['count'],
        'position': ['nunique']
    }).reset_index()
    
    # Flatten column names
    summary.columns = ['sequence_id', 'branch_length', 'total_events', 'positions_affected']
    
    # Add event type counts
    event_counts = events_df.groupby(['sequence_id', 'event_type']).size().unstack(fill_value=0)
    event_counts = event_counts.reset_index()
    
    # Merge
    summary = summary.merge(event_counts, on='sequence_id', how='left')
    
    # Calculate rates
    summary['events_per_unit_time'] = summary['total_events'] / summary['branch_length']
    
    return summary


@peint_caching.cached_parallel_computation(
        parallel_arg='families',
        output_dirs=['output_tree_dir'],
        exclude_args=['num_processes'],
)
def reroot_trees_from_fasttree(
    tree_dir: str,
    families: List[str],
    num_processes: int = 1,
    reroot_command: List[str] = ['perl', 'reroot.pl', '-midpoint'],
    output_tree_dir: Optional[str] = None,
):

    if not os.path.exists(output_tree_dir):
        os.makedirs(output_tree_dir)

    for family in tqdm(families, desc="Rerooting trees"):
        input_tree_file = os.path.join(tree_dir, f"{family}.txt")
        
        output_tree_file = os.path.join(output_tree_dir, f"{family}.txt")

        command_addon = ['<', input_tree_file, '>', output_tree_file]
        command = reroot_command + command_addon

        os.system(' '.join(command))
        
        #The reroot script leads to a duplicate node name, so we need to fix that.
        #The duplicate is an unnamed internal node, so the choice of new name is arbitrary.
        #This part is more general and can detect an number of duplicate nodes
        with open(output_tree_file, 'r') as f:
            newick_str = f.read().strip()
        tree = Tree(newick_str, format=1)
        node_distances = defaultdict(list)
        max_internal_num = 0
        internal_pattern = re.compile(r'^internal-(\d+)$')
        
        for node in tree.traverse():
            if node.name:
                distance = tree.get_distance(node)
                node_distances[node.name].append((node, distance))
                match = internal_pattern.match(node.name)
                if match:
                    max_internal_num = max(max_internal_num, int(match.group(1)))
        

        next_internal_num = max_internal_num + 1
        for _, nodes_with_dist in node_distances.items():
            if len(nodes_with_dist) > 1:
                # Sort by distance, keep closest to root unchanged
                nodes_with_dist.sort(key=lambda x: x[1])
                
                # Rename all but the first (closest to root)
                for node, _ in nodes_with_dist[1:]:
                    node.name = f"internal-{next_internal_num}"
                    next_internal_num += 1

        tree.write(format=1, outfile=output_tree_file)

def _prep_seqs_for_historian(
        simulated_tree_dir: str,
        simulated_sequences_dir: str,
        root_sequences_dir: str,
        real_tree_dir: str,
        real_sequences_dir: str,
        family: str,
):
    """
    Prepare seuqences for historian to compare evolutionay events
    between real and simulated data.
    For simulated data, we have all events, as we know the internal nodes,
    but for real data, this must be inferred.
    We'll restrict ourselves to a subtree to make it easier for the inference
    part.
    """

    original_tree = read_tree(
        os.path.join(real_tree_dir, f"{family}.txt")
    )
    original_msa = read_msa(
        os.path.join(real_sequences_dir, f"{family}.txt")
    )

    rerooted_tree = read_tree(
        os.path.join(simulated_tree_dir, f"{family}.txt")
    )

    simulated_msa = read_msa(
        os.path.join(simulated_sequences_dir, f"{family}.txt")
    )

    root_seq = read_msa(
        os.path.join(root_sequences_dir, f"{family}.txt")
    )
    rs_name = list(root_seq.keys())[0]

    optimal_edge = find_optimal_edge_split(original_tree)
    tree_a, tree_b = split_tree_on_edge(original_tree, optimal_edge)
    
    #want the opposite subtree to the one used for root
    if rs_name in tree_a.leaves():
        tree_to_keep = tree_b
        nodes_to_keep = tree_b.nodes()
    else:
        tree_to_keep = tree_a
        nodes_to_keep = tree_a.nodes()

    rerooted_tree_ete = rerooted_tree.to_ete3()
    tree_to_keep_ete = tree_to_keep.to_ete3()
    #prune
    rerooted_tree_ete.prune(tree_to_keep.leaves(), preserve_branch_length=True)

    #rename
    rerooted_tree_ete.name = tree_to_keep_ete.name

    #check if there's a noname node in here, this happens sometimes
    if "" in [n.name for n in rerooted_tree_ete.traverse()]:
        noname_node = rerooted_tree_ete&""
        noname_children = set(n.name for n in noname_node.children)
        for n in tree_to_keep_ete.traverse():
            n_kids = set(c.name for c in n.children)
            if n_kids == noname_children:
                break
        #rename that node to the name of n
        noname_node.name = n.name
        #do the same for the MSA
        if "" in simulated_msa:
            noname_seq = simulated_msa.pop("")
            simulated_msa[n.name] = noname_seq

    #filter the MSAs to only include the nodes in the tree
    filtered_simulated_msa = {
        k: v for k, v in simulated_msa.items() if k in nodes_to_keep
    }

    filtered_original_msa = {
        k: v for k, v in original_msa.items() if k in nodes_to_keep
    }

    rrt_nodes = set(n.name for n in rerooted_tree_ete.traverse())
    original_nodes = set(n.name for n in tree_to_keep_ete.traverse())
    assert rrt_nodes == original_nodes, \
        f"Rerooted tree nodes {rrt_nodes} do not match original tree nodes {original_nodes} for family {family}"
    
    return filtered_original_msa, tree_to_keep_ete, filtered_simulated_msa, rerooted_tree_ete

def _map_func_prep_for_historian(args):
    """
    Helper function to prepare sequences for historian in parallel
    """
    [simulated_tree_dir,
     simulated_sequences_dir,
     root_sequences_dir,
     real_tree_dir,
     real_sequences_dir,
     families,
     output_real_tree_dir,
     output_real_sequences_dir,
     output_simulated_tree_dir,
     output_simulated_sequences_dir] = args
    
    for family in families:
        filtered_original_seqs, tree_to_keep_real, filtered_simulated_seqs, \
            rerooted_tree_simulated = _prep_seqs_for_historian(
                simulated_tree_dir,
                simulated_sequences_dir,
                root_sequences_dir,
                real_tree_dir,
                real_sequences_dir,
                family)
        # Write the filtered original MSA
        write_msa(filtered_original_seqs,
                  os.path.join(output_real_sequences_dir, f"{family}.txt")
        )
        #write the tree to keep
        newick_str = tree_to_keep_real.write(format=1)
        if newick_str.endswith(";"):
            newick_str = newick_str[:-1] + f"{tree_to_keep_real.name};"
        with open(os.path.join(output_real_tree_dir, f"{family}.txt"), 'w') as f:
            f.write(newick_str)

        # Write the filtered simulated MSA
        write_msa(filtered_simulated_seqs,
                  os.path.join(output_simulated_sequences_dir, f"{family}.txt")
        )
        #write the rerooted tree
        newick_str = rerooted_tree_simulated.write(format=1)
        if newick_str.endswith(";"):
            newick_str = newick_str[:-1] + f"{rerooted_tree_simulated.name};"

        with open(os.path.join(output_simulated_tree_dir, f"{family}.txt"), 'w') as f:
            f.write(newick_str)

@peint_caching.cached_parallel_computation(
        parallel_arg='families',
        output_dirs=['output_real_tree_dir', 'output_real_sequences_dir',
                     'output_simulated_tree_dir', 'output_simulated_sequences_dir'],
        exclude_args=['num_processes'],
)
def prepare_simulated_vs_real_historian(
        simulated_tree_dir: str,
        simulated_sequences_dir: str,
        root_sequences_dir: str,
        real_tree_dir: str,
        real_sequences_dir: str,
        families: List[str],
        num_processes: int = 1,
        output_real_tree_dir: Optional[str] = None,
        output_real_sequences_dir: Optional[str] = None,
        output_simulated_tree_dir: Optional[str] = None,
        output_simulated_sequences_dir: Optional[str] = None,
):
    """
    Prepare sequences for historian to compare evolutionary events
    between real and simulated data.
    """
    
    if not os.path.exists(output_real_tree_dir):
        os.makedirs(output_real_tree_dir)
    if not os.path.exists(output_real_sequences_dir):
        os.makedirs(output_real_sequences_dir)
    if not os.path.exists(output_simulated_tree_dir):
        os.makedirs(output_simulated_tree_dir)
    if not os.path.exists(output_simulated_sequences_dir):
        os.makedirs(output_simulated_sequences_dir)

    map_args = [
        [
            simulated_tree_dir,
            simulated_sequences_dir,
            root_sequences_dir,
            real_tree_dir,
            real_sequences_dir,
            get_process_args(process_rank, num_processes, families),
            output_real_tree_dir,
            output_real_sequences_dir,
            output_simulated_tree_dir,
            output_simulated_sequences_dir,
        ]
        for process_rank in range(num_processes)
    ]

    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(
                tqdm(
                    pool.imap(_map_func_prep_for_historian, map_args),
                    total=len(map_args),
                )
            )
    else:
        list(
            tqdm(
                map(_map_func_prep_for_historian, map_args),
                total=len(map_args),
            )
        )

@peint_caching.cached_parallel_computation(
        parallel_arg = 'families',
        output_dirs = ['output_sequences_dir', 'output_tree_dir'],
        exclude_args = ['num_processes'],
)
def prepare_sequences_for_historian(
        sequences_dir: str,
        tree_dir: str,
        root_sequence_dir: str,
        families: List[str],
        num_processes: int = 1,
        output_tree_dir: Optional[str] = None,
        output_sequences_dir: Optional[str] = None,
):
    
    if not os.path.exists(output_sequences_dir):
        os.makedirs(output_sequences_dir)
    if not os.path.exists(output_tree_dir):
        os.makedirs(output_tree_dir)
        
    for family in tqdm(families, desc="Preparing sequences for Historian"):
        seqs = read_msa(os.path.join(sequences_dir, f"{family}"))
        tree = read_tree(os.path.join(tree_dir, f"{family}")).to_ete3() #using our tree format
        root_sequence = read_msa(os.path.join(root_sequence_dir, f"{family}"))

        # Handle the unnamed node from rerooting
        if "" in seqs:
            noname_seq = seqs.pop("")
            
            # Find the next available internal-X number
            existing_internal_nums = set()
            for node in tree.traverse():
                if node.name and node.name.startswith("internal-"):
                    try:
                        num = int(node.name.split("-")[1])
                        existing_internal_nums.add(num)
                    except (IndexError, ValueError):
                        pass
            
            # Find the next available number
            next_internal_num = 0
            while next_internal_num in existing_internal_nums:
                next_internal_num += 1
            
            new_internal_name = f"internal-{next_internal_num}"
            seqs[new_internal_name] = noname_seq
            
            for node in tree.traverse():
                if node.name == "":
                    node.name = new_internal_name
                    break
        
        # Add the root sequence
        root_sequence_name = list(root_sequence.keys())[0]
        root_seq = root_sequence[root_sequence_name]
        seqs[tree.name] = root_seq

        # Generate the newick string
        newick_str = tree.write(format=1)
        if newick_str.endswith(";"):
            newick_str = newick_str[:-1] + f"{tree.name};"

        # Write sequences
        write_msa(seqs, os.path.join(output_sequences_dir, f"{family}.txt"))

        # Write tree (fixed typo: outout_tree_file -> output_tree_file)
        output_tree_file = os.path.join(output_tree_dir, f"{family}.txt")
        with open(output_tree_file, 'w') as f:
            f.write(newick_str)

@peint_caching.cached_parallel_computation(
        parallel_arg = 'families',
        exclude_args_if_default = ['dummy_branch_length'],
        exclude_args = ['num_processes'],
        output_dirs = ['output_sequences_dir', 'output_tree_dir'],
)
def add_dummy_nodes(
        sequences_dir: str,
        tree_dir: str,
        families: List[str],
        num_processes: int = 1,
        output_tree_dir: Optional[str] = None,
        output_sequences_dir: Optional[str] = None,
        dummy_branch_length: float = 0.005,
):
    
    if not os.path.exists(output_sequences_dir):
        os.makedirs(output_sequences_dir)
    if not os.path.exists(output_tree_dir):
        os.makedirs(output_tree_dir)

    for family in tqdm(families, desc="Adding dummy nodes"):
        # Read sequences and tree
        msa = read_msa(os.path.join(sequences_dir, f"{family}.txt"))
        with open(os.path.join(tree_dir, f"{family}.txt"), 'r') as f:
            newick_tree = f.read().strip()
        
        # Add dummy nodes
        dummy_sequences, dummy_newick_tree = _add_dummy_nodes_to_tree(
            msa, newick_tree, dummy_branch_length=dummy_branch_length
        )
        
        # Write output files
        write_msa(dummy_sequences, os.path.join(output_sequences_dir, f"{family}.txt"))
        with open(os.path.join(output_tree_dir, f"{family}.txt"), 'w') as f:
            f.write(dummy_newick_tree)

def _map_func_run_historian(args):
    """
    Helper function to run Historian in parallel
    """
    sequences_dir, tree_dir, families, output_sequences_dir, historian_path, extra_command_line_args = args
    for family in families:
        # Read sequences and tree - these have dummy nodes already added
        msa_file = os.path.join(sequences_dir, f"{family}.txt")
        if tree_dir is not None:
            tree_file = os.path.join(tree_dir, f"{family}.txt")
        else:
            tree_file = None

        output_file = os.path.join(output_sequences_dir, f"{family}.txt")

        historian_command = [
            historian_path,
            "recon",
            "-guide", msa_file,
            "-tree", tree_file,
            "-ancseq",
            "-output", "fasta"
        ]

        if tree_file is None:
            #pop the tree argument if not provided
            historian_command.pop(historian_command.index("-tree"))
            historian_command.pop(historian_command.index(tree_file))

        # Add extra command line arguments if provided
        if extra_command_line_args:
            historian_command.extend(extra_command_line_args)

        output = subprocess.run(
            historian_command,
            capture_output=True,
            text=True
        )

        # Write stdout to file
        with open(output_file, 'w') as f:
            f.write(output.stdout)

        # Check for errors
        process_return_code = output.returncode
        if process_return_code != 0:
            logger.error(f"Historian failed for family {family} with return code {process_return_code}")
            logger.error(f"Output: {output.stdout}")
            logger.error(f"Error: {output.stderr}")
            raise RuntimeError(f"Historian failed for family {family} with return code {process_return_code}")

        #should be written already to file, but check
        secure_parallel_output(output_sequences_dir, family)

@peint_caching.cached_parallel_computation(
        parallel_arg = 'families',
        exclude_args = ['num_processes'],
        output_dirs = ['output_sequences_dir'],
)
def run_historian(
        sequences_dir: str,
        tree_dir: str,
        families: List[str],
        num_processes: int = 1,
        output_sequences_dir: Optional[str] = None,
        historian_path: str = "historian",
        extra_command_line_args: Optional[List[str]] = None,
):
    """
    Run Historian on the sequences and trees for each family.
    These have dummy nodes already added, and they should be aligned using mafft as a guide alignment.
    """

    if not os.path.exists(output_sequences_dir):
        os.makedirs(output_sequences_dir)

    map_args = [
        [
            sequences_dir,
            tree_dir,
            get_process_args(process_rank, num_processes, families),
            output_sequences_dir,
            historian_path,
            extra_command_line_args or []
        ]
        for process_rank in range(num_processes)
    ]

    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(
                tqdm(
                    pool.imap(_map_func_run_historian, map_args),
                    total=len(map_args),
                )
            )
    else:
        list(
            tqdm(
                map(_map_func_run_historian, map_args),
                total=len(map_args),
            )
        )

def _add_dummy_nodes_to_tree(msa: Dict[str, str], 
                   newick_tree: str,
                   dummy_branch_length: float = 0.005) -> Tuple[Dict[str, str], str]:
    """
    Add dummy leaf nodes for all internal nodes using proper binary tree structure
    
    For each internal node we want to sample, we insert a new dummy internal node
    just above it, creating:
    parent -> dummy_internal (original_length - eps) -> [original_node (eps), dummy_leaf (eps)]
    
    Args:
        sequences: Original sequences
        newick_tree: Original tree
        dummy_branch_length: Branch length for dummy connections (eps)
    
    Returns:
        Tuple of (dummy_sequences, dummy_newick_tree)
    """

    def _insert_dummy_node(tree: Tree, 
                           node: Tree, 
                           dummy_internal_name: str, 
                           dummy_leaf_name: str,
                           dummy_branch_length: float
                           ) -> Tree:
        """Insert a dummy internal node above the given node. Returns the tree (which may be modified)."""
        original_dist = node.dist
        parent = node.up
        
        if parent is not None:
            # Remove the node from its parent
            parent.remove_child(node)
            
            # Create new dummy internal node
            dummy_internal = parent.add_child(
                name=dummy_internal_name, 
                dist=max(0.001, original_dist - dummy_branch_length)
            )
            
            # Add the original node and dummy leaf as children of dummy internal
            dummy_internal.add_child(node, dist=dummy_branch_length)
            dummy_internal.add_child(name=dummy_leaf_name, dist=dummy_branch_length)
            
            return tree
            
        else:
            # Handle root node - use same naming convention
            new_root = Tree(name=dummy_internal_name)
            
            # Add original tree and dummy leaf as children
            new_root.add_child(node, dist=dummy_branch_length)
            new_root.add_child(name=dummy_leaf_name, dist=dummy_branch_length)
            
            return new_root
        
    tree = Tree(newick_tree, format=1)
    
    # Add names to unnamed internal nodes
    already_named = set()
    for node in tree.traverse():
        if node.name.startswith("internal-"):
            # Extract the number from the name
            try:
                num = int(node.name.split("-")[1])
                already_named.add(num)
            except (IndexError, ValueError):
                continue
            
    for node in tree.traverse():
        if not node.name or node.name == "":
            # Find the next available internal-X number
            next_internal_num = 0
            while next_internal_num in already_named:
                next_internal_num += 1
            node.name = f"internal-{next_internal_num}"
            already_named.add(next_internal_num)
    
    # Create dummy sequences
    dummy_sequences = msa.copy()
    
    # Find internal nodes that need dummy leaves
    nodes_to_modify = [
        node for node in tree.traverse() 
        if not node.is_leaf() and node.name in msa and len(node.get_children()) > 0
    ]
    
    dummy_counter = 0
    
    # Process each internal node that has a sequence and has children
    for node in nodes_to_modify:
        dummy_counter += 1

        dummy_internal_name = f"{node.name}_dummy_internal"
        dummy_leaf_name = f"{node.name}_dummy_leaf"

        # The dummy leaf is an *observed* copy of this internal node's known sequence,
        # pinning Historian's reconstruction of the ancestor. Without this the guide MSA
        # lacks the dummy-leaf sequence and Historian aborts ("can't find sequence for
        # leaf node <name>_dummy_leaf").
        dummy_sequences[dummy_leaf_name] = msa[node.name]

        tree = _insert_dummy_node(tree, node, dummy_internal_name, dummy_leaf_name,
                                 dummy_branch_length)
                
    # Convert back to Newick
    dummy_newick = tree.write(format=1)
    if dummy_newick.endswith(";"):
        dummy_newick = dummy_newick[:-1] + f"{tree.name};"
    
    return dummy_sequences, dummy_newick

def _remove_gap_columns(msa: Dict[str, str]) -> Dict[str, str]:
    """
    Remove columns that are all gaps from the MSA.
    
    Args:
        msa: Multiple sequence alignment
        
    Returns:
        MSA with all-gap columns removed
    """
    if not msa:
        return msa
    
    # Get sequence length (all sequences should be same length)
    seq_length = len(next(iter(msa.values())))
    
    # Check if all sequences are the same length
    if not all(len(seq) == seq_length for seq in msa.values()):
        raise ValueError("All sequences in MSA must be the same length")
    
    # Find columns to keep (not all gaps)
    columns_to_keep = []
    
    for col_idx in range(seq_length):
        column_chars = set(seq[col_idx] for seq in msa.values())
        if column_chars != {'-'}:
            # If column is not all gaps, keep it
            columns_to_keep.append(col_idx)
    
    # If no columns to remove, return original
    if len(columns_to_keep) == seq_length:
        return msa
    
    print(f"Removing {seq_length - len(columns_to_keep)} all-gap columns from MSA")
    
    # Build new MSA with only kept columns
    filtered_msa = {}
    for seq_id, sequence in msa.items():
        filtered_sequence = ''.join(sequence[i] for i in columns_to_keep)
        filtered_msa[seq_id] = filtered_sequence
    return filtered_msa

def remove_dummy_nodes_from_msa(msa: Dict[str, str]) -> Dict[str, str]:
    """Remove dummy sequences from MSA."""

    cleaned_msa = {}

    for seq_id, sequence in msa.items():
        # Skip all dummy sequences (both leaves and internal nodes)
        if seq_id.endswith('_dummy_leaf') or seq_id.endswith('_dummy_internal'):
            continue
            
        # # Handle internal nodes
        # if seq_id.startswith('internal-'):
        #     dummy_leaf_name = f"{seq_id}_dummy_leaf"
        #     if dummy_leaf_name in msa:
        #         cleaned_msa[seq_id] = msa[dummy_leaf_name].lower()
        #     else:
        #         cleaned_msa[seq_id] = sequence.lower() #fallback
        # else:
        #     # Regular leaf nodes - keep uppercase
        #     cleaned_msa[seq_id] = sequence.upper()

        cleaned_msa[seq_id] = sequence  # Keep all sequences uppercase
    
    # Remove all-gap columns
    cleaned_msa = _remove_gap_columns(cleaned_msa)

    
    return cleaned_msa

@peint_caching.cached_parallel_computation(
        parallel_arg = 'families',
        exclude_args = ['num_processes'],
        output_dirs = ['output_sequences_dir'],
)
def remove_dummy_nodes_from_historian_output(
        sequences_dir: str,
        families: List[str],
        num_processes: int = 1,
        output_sequences_dir: Optional[str] = None,
):
    if not os.path.exists(output_sequences_dir):
        os.makedirs(output_sequences_dir)
    
    for family in tqdm(families, desc="Removing dummy nodes from Historian output"):
        input_file = os.path.join(sequences_dir, f"{family}.txt")
        output_file = os.path.join(output_sequences_dir, f"{family}.txt")
        
        # Read the MSA
        msa = read_msa(input_file)
        
        # Remove dummy nodes
        cleaned_msa = remove_dummy_nodes_from_msa(msa)
        
        # Write cleaned MSA
        write_msa(cleaned_msa, output_file)

def add_dummy_nodes_linear(sequences: Dict[str, str], 
                          branch_lengths: List[float],
                          dummy_branch_length: float = 0.005) -> Tuple[Dict[str, str], str]:
    """
    Add dummy leaf nodes for linear topology directly to internal nodes
    
    For linear topology, we can add dummy leaves directly since it's not binary anyway.
    Creates a linear tree from the sequences and adds dummy leaves to all internal nodes.
    
    Args:
        sequences: Dict mapping node_id -> sequence (in order from root to leaf)
        branch_lengths: List of branch lengths between consecutive nodes
        dummy_branch_length: Branch length for dummy leaf connections
    
    Returns:
        Tuple of (dummy_sequences, dummy_newick_tree)
    """
    node_names = list(sequences.keys())
    
    if len(node_names) < 2:
        raise ValueError("Linear topology requires at least 2 sequences")
    
    # Ensure we have enough branch lengths
    if len(branch_lengths) < len(node_names) - 1:
        missing = len(node_names) - 1 - len(branch_lengths)
        branch_lengths.extend([0.1] * missing)
        logger.warning(f"Missing {missing} branch lengths, using default 0.1")
    
    # Create dummy sequences - start with original sequences
    dummy_sequences = sequences.copy()
    
    # Build linear tree structure using ete3
    # Start with the first node (root)
    tree = Tree(name=node_names[0])
    current_node = tree
    
    # Add subsequent nodes in linear fashion
    for i in range(1, len(node_names)):
        new_node = current_node.add_child(name=node_names[i], dist=branch_lengths[i-1])
        current_node = new_node
    
    # Add dummy leaves to all internal nodes (all nodes except the last leaf)
    for i, node_name in enumerate(node_names[:-1]):  # Exclude the final leaf
        dummy_leaf_name = f"{node_name}_dummy"
        
        # Add dummy leaf sequence (same as internal node)
        dummy_sequences[dummy_leaf_name] = sequences[node_name]
        
        # Find the node in the tree and add dummy leaf
        for node in tree.traverse():
            if node.name == node_name:
                node.add_child(name=dummy_leaf_name, dist=dummy_branch_length)
                logger.info(f"Added dummy leaf '{dummy_leaf_name}' to node '{node_name}'")
                break
    
    # Convert to Newick
    dummy_newick = tree.write(format=1)
    
    logger.info(f"Linear tree with dummy nodes: {dummy_newick}")
    return dummy_sequences, dummy_newick

def _map_func_historian_count_events(args):
    """
    Helper function to run Historian in parallel for counting events
    """
    sequences_dir, tree_dir, families, output_counts_dir, historian_path = args
    error_families = []
    for family in families:
        msa_file = os.path.join(sequences_dir, f"{family}.txt")
        tree_file = os.path.join(tree_dir, f"{family}.txt")
        output_file = os.path.join(output_counts_dir, f"{family}.txt")


        #reformat because Historian has a bug in their json output...
        
        historian_command = [
            historian_path,
            "count",
            "-recon", msa_file,
            "-tree", tree_file
        ]
        # Redirect stdout to file directly
        with open(output_file, 'w') as f:
            result = subprocess.run(
                historian_command,
                stdout=f,
                stderr=subprocess.PIPE,
                text=True
            )
        
        # Check for errors
        
        with open(output_file, 'r') as f:
            alllines = [l for l in f]
        if not alllines:
            logger.error(f"Historian count events failed for family {family} with no output.")
            logger.error(f"Error message: {result.stderr}")
            continue
        new_all_lines = []
        for l in alllines:
            #known error where historian's writer doesn't add a comma at the end of this line, which breaks json
            if 'insTime' in l:
                new_all_lines.append(l.replace('\n', '') + ',\n')
            else:
                l = l.replace('-nan', '0.0').replace('-inf', '0.0').replace('nan', '0.0').replace('inf', '0.0')  # Replace -nan with 0.0
                new_all_lines.append(l)
        json_str = ''.join(new_all_lines)
        good_json = json.loads(json_str)
        with open(output_file, 'w') as f:
            json.dump(good_json, f, indent=4)
    

@peint_caching.cached_parallel_computation(
        parallel_arg = 'families',
        exclude_args = ['num_processes'],
        output_dirs = ['output_counts_dir'],
)
def historian_count_events(
        sequences_dir: str,
        tree_dir: str,
        families: List[str],
        num_processes: int = 1,
        output_counts_dir: Optional[str] = None,
        historian_path: str = "historian",
):

    if not os.path.exists(output_counts_dir):
        os.makedirs(output_counts_dir)

    map_args = [
        [
            sequences_dir,
            tree_dir,
            get_process_args(process_rank, num_processes, families),
            output_counts_dir,
            historian_path
        ]
        for process_rank in range(num_processes)
    ]

    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(
                tqdm(
                    pool.imap(_map_func_historian_count_events, map_args),
                    total=len(map_args),
                )
            )
    else:
        list(
            tqdm(
                map(_map_func_historian_count_events, map_args),
                total=len(map_args),
            )
        )


def extract_evolutionary_events(alignment, 
                               newick_tree: str) -> List[EvolutionaryEvent]:
    """Extract evolutionary events from alignment.
    A bit repeptitive since we already have the pairwise events,
    but this should be more general, and the pairwise setup is 
    really more for the star topology."""
    tree = Tree(newick_tree, format=1)
    events = []
    
    # Extract events for each parent-child pair
    for node in tree.traverse():
        if node.up:  # Has parent
            parent_id = node.up.name
            child_id = node.name
            branch_length = node.dist
            
            if parent_id not in alignment or child_id not in alignment:
                logger.warning(f"Missing sequences for {parent_id} -> {child_id}")
                continue
            
            parent_seq = alignment[parent_id].upper()
            child_seq = alignment[child_id].upper()
            
            # Compare sequences position by position
            position = 0
            for i in range(len(parent_seq)):
                parent_char = parent_seq[i]
                child_char = child_seq[i]
                
                if parent_char != '-' and child_char != '-':
                    # Both present - check for substitution
                    if parent_char != child_char:
                        events.append(EvolutionaryEvent(
                            sequence_id=child_id,
                            event_type='substitution',
                            position=position,
                            ancestral_residue=parent_char,
                            evolved_residue=child_char,
                            branch_length=branch_length,
                            node_pair=(parent_id, child_id)
                        ))
                    position += 1
                elif parent_char != '-' and child_char == '-':
                    # Deletion
                    events.append(EvolutionaryEvent(
                        sequence_id=child_id,
                        event_type='deletion',
                        position=position,
                        ancestral_residue=parent_char,
                        evolved_residue='-',
                        branch_length=branch_length,
                        node_pair=(parent_id, child_id)
                    ))
                elif parent_char == '-' and child_char != '-':
                    # Insertion
                    events.append(EvolutionaryEvent(
                        sequence_id=child_id,
                        event_type='insertion',
                        position=position,
                        ancestral_residue='-',
                        evolved_residue=child_char,
                        branch_length=branch_length,
                        node_pair=(parent_id, child_id)
                    ))
    
    return events

def collate_evolutionary_events(events: List[EvolutionaryEvent]) -> pd.DataFrame:
    """Collate evolutionary events into a DataFrame."""
    # Group events by (sequence_id, node_pair, event_type, position, branch_length)
    grouped_events = defaultdict(list)
    
    for event in events:
        # For indels, group by position. For substitutions, keep separate
        if event.event_type in ['insertion', 'deletion']:
            key = (event.sequence_id, event.node_pair, event.event_type, 
                   event.position, event.branch_length)
        else:  # substitution
            # Use a unique key for each substitution to keep them separate
            key = (event.sequence_id, event.node_pair, event.event_type, 
                   event.position, event.branch_length, id(event))
        
        grouped_events[key].append(event)
    
    # Convert to DataFrame rows
    df_data = []
    
    for key, event_list in grouped_events.items():
        if not event_list:
            continue
            
        first_event = event_list[0]
        event_type = first_event.event_type
        
        if event_type == 'substitution':
            # Keep substitutions as individual events
            for event in event_list:
                df_data.append({
                    'sequence_id': event.sequence_id,
                    'parent_node': event.node_pair[0],
                    'child_node': event.node_pair[1],
                    'branch_length': event.branch_length,
                    'event_type': event.event_type,
                    'start_position': event.position,
                    'end_position': event.position,
                    'length': 1,
                    'ancestral_sequence': event.ancestral_residue,
                    'evolved_sequence': event.evolved_residue
                })
        else:
            # Combine insertions/deletions at the same position
            length = len(event_list)
            
            if event_type == 'insertion':
                ancestral_seq = '-' * length
                evolved_seq = ''.join(event.evolved_residue for event in event_list)
            else:  # deletion
                ancestral_seq = ''.join(event.ancestral_residue for event in event_list)
                evolved_seq = '-' * length
            
            df_data.append({
                'sequence_id': first_event.sequence_id,
                'parent_node': first_event.node_pair[0],
                'child_node': first_event.node_pair[1],
                'branch_length': first_event.branch_length,
                'event_type': event_type,
                'start_position': first_event.position,
                'end_position': first_event.position,  # Same position for indels at same site
                'length': length,
                'ancestral_sequence': ancestral_seq,
                'evolved_sequence': evolved_seq
            })
    
    # Convert to DataFrame and sort
    df = pd.DataFrame(df_data)
    
    if not df.empty:
        df = df.sort_values(['sequence_id', 'start_position']).reset_index(drop=True)
    
    return df


def _map_func_get_full_counts_historian(args):
    """
    Helper function to count events after historian has been run
    """
    sequences_dir, tree_dir, families, output_events_dir = args

    for family in families:
        msa_file = os.path.join(sequences_dir, f"{family}.txt")
        tree_file = os.path.join(tree_dir, f"{family}.txt")
        output_file = os.path.join(output_events_dir, f"{family}.txt")

        #tree is a newick

        with open(tree_file, 'r') as f:
            newick_tree = f.read().strip()

        events = extract_evolutionary_events(
            read_msa(msa_file),
            newick_tree #tree gets read
        )

        summary_df = collate_evolutionary_events(events)
        if summary_df.empty:
            logger.warning(f"No evolutionary events found for family {family}")
            continue
        # Write the summary DataFrame to a file
        summary_df.to_csv(output_file, index=False)

@peint_caching.cached_parallel_computation(
        parallel_arg='families',
        exclude_args=['num_processes'],
        output_dirs=['output_events_dir'],
)
def get_all_evolutionary_counts_from_historian_output(
        sequences_dir: str,
        tree_dir: str,
        families: List[str],
        num_processes: int = 1,
        output_events_dir: Optional[str] = None,
):
    """
    Get full counts of evolutionary events from historian output.
    This is a more general function that works with any topology.
    """
    
    if not os.path.exists(output_events_dir):
        os.makedirs(output_events_dir)

    map_args = [
        [
            sequences_dir,
            tree_dir,
            get_process_args(process_rank, num_processes, families),
            output_events_dir
        ]
        for process_rank in range(num_processes)
    ]

    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(
                tqdm(
                    pool.imap(_map_func_get_full_counts_historian, map_args),
                    total=len(map_args),
                )
            )
    else:
        list(
            tqdm(
                map(_map_func_get_full_counts_historian, map_args),
                total=len(map_args),
            )
        )



# ---------------------------------------------------------------------------
# Locating the revision-2 (ESM-C) Historian outputs
# ---------------------------------------------------------------------------
ESMC_HISTORIAN_VARIANTS = ("refine", "norefine")


def esmc_historian_dirs(r2_root, variant: str = "refine") -> Dict[str, str]:
    """Return ``{"events": ..., "reconstructions": ...}`` for one ESM-C Historian run.

    ``historian_esmc_progressive.py`` was run twice, with and without Historian's
    ``--refine``. Preferred layout is the repacked, self-describing one::

        <r2_root>/historian_esmc/<variant>/{events,reconstructions}

    Falling back to the peint content-hash cache, whose paths carry no label — there
    the two runs are told apart by the mtime of the cached output directory. Each run
    wrote its reconstructions and then its event tables within a minute of each other,
    and the two runs are a day apart, so pairing the directories in time order gives
    norefine first and refine second.

    Selecting these by ``sorted(glob(...))[0]`` (as several scripts used to) silently
    mixes the two: the events came from the norefine run while the reconstructions were
    hardcoded to the refine one.
    """
    if variant not in ESMC_HISTORIAN_VARIANTS:
        raise ValueError(f"variant must be one of {ESMC_HISTORIAN_VARIANTS}, got {variant!r}")

    packed = os.path.join(str(r2_root), "historian_esmc", variant)
    events, recons = os.path.join(packed, "events"), os.path.join(packed, "reconstructions")
    if os.path.isdir(events) and os.path.isdir(recons):
        return {"events": events, "reconstructions": recons}

    cache = os.path.join(str(r2_root), "simulations", "historian_progressive", "_cache")

    def _by_time(func_name, output_name):
        hits = glob.glob(os.path.join(cache, func_name, "*/*/*/*", output_name))
        if len(hits) != len(ESMC_HISTORIAN_VARIANTS):
            raise FileNotFoundError(
                f"Expected {len(ESMC_HISTORIAN_VARIANTS)} cached {output_name} dirs under "
                f"{func_name}, found {len(hits)}. Repack them into "
                f"{packed} (see esmc_historian_dirs) and re-run."
            )
        return [p for _, p in sorted((os.path.getmtime(p), p) for p in hits)]

    order = list(ESMC_HISTORIAN_VARIANTS[::-1])  # oldest run first == norefine
    i = order.index(variant)
    return {
        "events": _by_time("get_all_evolutionary_counts_from_historian_output",
                           "output_events_dir")[i],
        "reconstructions": _by_time("remove_dummy_nodes_from_historian_output",
                                    "output_sequences_dir")[i],
    }
