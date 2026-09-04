"""3Di structural-state annotation with ProstT5.

Ported from the model repo's ``peint/datasets/_3di.py``. ProstT5 is a *benchmarking*
dependency (transformers + sentencepiece), not part of the ``peint`` model library.

Two sets of weights are involved and they come from different places: the ProstT5 encoder
itself is pulled from HuggingFace into a cache directory, and the small CNN head that maps
embeddings to 3Di states is a file in the ProstT5 GitHub repo. Both are fetched by
``scripts/fetch_3di_weights.sh``. The original resolved the CNN head relative to the
*current working directory*, so it silently re-downloaded depending on where you ran from;
it now comes from ``paper_config``.
"""

import argparse
import time
from pathlib import Path
import os
import tempfile

from urllib import request
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing
import tqdm
from Bio import SeqIO
from transformers import T5EncoderModel, T5Tokenizer
from peint import caching as peint_caching
from peint.caching import secure_parallel_output
from peint.utils import get_process_args, read_msa, write_msa

import paper_config as cfg

def align_gapless_sequences_to_alignment(sequences, alignment):
    superimposed_sequences = {}
    for id, seq in sequences.items():
        aligned_seq = alignment[id]

        superimposed_seq = [x for x in aligned_seq]

        og_idx = 0
        for i in range(len(superimposed_seq)):
            if superimposed_seq[i] != '-':
                superimposed_seq[i] = seq[og_idx]
                og_idx += 1

        superimposed_seq = "".join(superimposed_seq)
        superimposed_sequences[id] = superimposed_seq
    return superimposed_sequences

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.classifier = nn.Sequential(
            nn.Conv2d(1024, 32, kernel_size=(7, 1), padding=(3, 0)),  # 7x32
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Conv2d(32, 20, kernel_size=(7, 1), padding=(3, 0))
        )

    def forward(self, x):
        """
            L = protein length
            B = batch-size
            F = number of features (1024 for embeddings)
            N = number of classes (20 for 3Di)
        """
        x = x.permute(0, 2, 1).unsqueeze(
            dim=-1)  # IN: X = (B x L x F); OUT: (B x F x L, 1)
        Yhat = self.classifier(x)  # OUT: Yhat_consurf = (B x N x L x 1)
        Yhat = Yhat.squeeze(dim=-1)  # IN: (B x N x L x 1); OUT: ( B x N x L )
        return Yhat


def get_T5_model(model_dir=None, device='cuda'):
    # model_dir is a HuggingFace cache directory, not a checkpoint file.
    model_dir = str(model_dir or cfg.PROSTT5_CACHE_DIR)
    print("Loading T5 from: {}".format(model_dir))
    model = T5EncoderModel.from_pretrained(
        "Rostlab/ProstT5_fp16", cache_dir=model_dir).to(device)
    model = model.eval()
    vocab = T5Tokenizer.from_pretrained(
        "Rostlab/ProstT5_fp16", do_lower_case=False, cache_dir=model_dir)
    return model, vocab


def read_fasta(fasta_path):
    '''
        Reads in fasta file containing multiple sequences.
        Returns dictionary of holding multiple sequences or only single 
        sequence, depending on input file.
    '''

    sequences = dict()
    with open(fasta_path, 'r') as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.replace(
                    '>', '').strip()
                # replace tokens that are mis-interpreted when loading h5
                # uniprot_id = uniprot_id.replace("/", "_").replace(".", "_")
                sequences[uniprot_id] = ''
            else:
                s = ''.join(line.split()).replace("-", "")

                if s.islower():  # sanity check to avoid mix-up of 3Di and AA input
                    print("The input file was in lower-case which indicates 3Di-input." +
                          "This predictor only operates on amino-acid-input (upper-case)." +
                          "Exiting now ..."
                          )
                    return None
                else:
                    sequences[uniprot_id] += s
    return sequences


def write_probs(predictions, out_path):
    with open(out_path, 'w+') as out_f:
        out_f.write('\n'.join(
            ["{},{}".format(seq_id, prob)
             for seq_id, (N, prob) in predictions.items()
             ]
        ))
    print(f"Finished writing probabilities to {out_path}")
    return None


def write_predictions(predictions, out_path):
    ss_mapping = {
        0: "A",
        1: "C",
        2: "D",
        3: "E",
        4: "F",
        5: "G",
        6: "H",
        7: "I",
        8: "K",
        9: "L",
        10: "M",
        11: "N",
        12: "P",
        13: "Q",
        14: "R",
        15: "S",
        16: "T",
        17: "V",
        18: "W",
        19: "Y"
    }

    with open(out_path, 'w+') as out_f:
        out_f.write('\n'.join(
            [">{}\n{}".format(
                seq_id, "".join(list(map(lambda yhat: ss_mapping[int(yhat)], yhats))))
             for seq_id, (yhats, _) in predictions.items()
             ]
        ))
    print(f"Finished writing results to {out_path}")
    return None


def toCPU(tensor):
    if len(tensor.shape) > 1:
        return tensor.detach().cpu().squeeze(dim=-1).numpy()
    else:
        return tensor.detach().cpu().numpy()


def download_file(url, local_path):
    local_path.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading: {}".format(url))
    req = request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'
    })

    with request.urlopen(req) as response, open(local_path, 'wb') as outfile:
        shutil.copyfileobj(response, outfile)
    return None


CNN_WEIGHTS_URL = "https://github.com/mheinzinger/ProstT5/raw/main/cnn_chkpnt/model.pt"


def load_predictor(weights_link=CNN_WEIGHTS_URL, device='cuda', checkpoint_path=None):
    model = CNN()
    checkpoint_p = Path(checkpoint_path or cfg.PROSTT5_CNN_CHECKPOINT)
    # Fetched by scripts/fetch_3di_weights.sh; downloaded on demand if still absent.
    if not checkpoint_p.exists():
        download_file(weights_link, checkpoint_p)

    # Torch load will map back to device from state, which often is GPU:0.
    # to overcome, need to explicitly map to active device
    state = torch.load(checkpoint_p, map_location=device)

    model.load_state_dict(state["state_dict"])

    model = model.eval()
    model = model.to(device)

    return model


def get_embeddings(seq_path, out_path, probs_path, model_dir, half_precision, device='cuda',
                   max_residues=4000, max_seq_len=1000, max_batch=500, include_internals=False):
    seq_dict = dict()
    predictions = dict()

    # Read in fasta
    seq_dict = read_fasta(seq_path)
    if not include_internals:
        seq_dict = {k: v for k, v in seq_dict.items() if not k.startswith('internal')}

    # ProstT5 needs >=2 residues per sequence: s_len=0 produces NaN over an empty
    # slice, and s_len=1 squeezes the prediction tensor to a 0-D scalar that
    # downstream `len(...)` chokes on. Skip these rows entirely (they're
    # uninformative anyway) and log so we can audit upstream gap patterns.
    _MIN_3DI_RESIDUES = 2
    degenerate = [pid for pid, seq in seq_dict.items() if len(seq) < _MIN_3DI_RESIDUES]
    if degenerate:
        print(f"[3di] dropping {len(degenerate)} sequences with <{_MIN_3DI_RESIDUES} residues from {seq_path}: {degenerate}")
        seq_dict = {pid: seq for pid, seq in seq_dict.items() if len(seq) >= _MIN_3DI_RESIDUES}
    if not seq_dict:
        print(f"[3di] no sequences with >={_MIN_3DI_RESIDUES} residues in {seq_path}; writing empty outputs")
        write_predictions({}, out_path)
        write_probs({}, probs_path)
        return True

    prefix = "<AA2fold>"

    model, vocab = get_T5_model(model_dir, device)
    predictor = load_predictor(device = device)

    if half_precision:
        model.half()
        predictor.half()
        print("Using models in half-precision.")
    else:
        model.to(torch.float32)
        predictor.to(torch.float32)
        print("Using models in full-precision.")

    print('########################################')
    print('Example sequence: {}\n{}'.format(next(iter(
        seq_dict.keys())), next(iter(seq_dict.values()))))
    print('########################################')
    print('Total number of sequences: {}'.format(len(seq_dict)))

    avg_length = sum([len(seq) for _, seq in seq_dict.items()]) / len(seq_dict)
    n_long = sum([1 for _, seq in seq_dict.items() if len(seq) > max_seq_len])
    seq_dict_names = list(seq_dict.keys())
    # sort sequences by length to trigger OOM at the beginning
    seq_dict = sorted(seq_dict.items(), key=lambda kv: len(
        seq_dict[kv[0]]), reverse=True)

    print("Average sequence length: {}".format(avg_length))
    print("Number of sequences >{}: {}".format(max_seq_len, n_long))

    start = time.time()
    batch = list()
    standard_aa = "ACDEFGHIKLMNPQRSTVWY"
    standard_aa_dict = {aa: aa for aa in standard_aa}
    count = 0
    for seq_idx, (pdb_id, seq) in enumerate(seq_dict, 1):
        # replace the non-standard amino acids with 'X'
        seq = ''.join([standard_aa_dict.get(aa, 'X') for aa in seq])
        seq_len = len(seq)
        seq = prefix + ' ' + ' '.join(list(seq))
        batch.append((pdb_id, seq, seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed
        n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len

        if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
            count += len(batch)
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            token_encoding = vocab.batch_encode_plus(seqs,
                                                     add_special_tokens=True,
                                                     padding="longest",
                                                     return_tensors='pt'
                                                     ).to(device)
            try:
                with torch.no_grad():
                    embedding_repr = model(token_encoding.input_ids,
                                           attention_mask=token_encoding.attention_mask
                                           )
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(
                    pdb_id, seq_len)
                )
                continue

            # ProtT5 appends a special tokens at the end of each sequence
            # Mask this also out during inference while taking into account the prefix
            for idx, s_len in enumerate(seq_lens):
                token_encoding.attention_mask[idx, s_len+1] = 0

            # extract last hidden states (=embeddings)
            residue_embedding = embedding_repr.last_hidden_state.detach()
            # mask out padded elements in the attention output (can be non-zero) for further processing/prediction
            residue_embedding = residue_embedding * \
                token_encoding.attention_mask.unsqueeze(dim=-1)
            # slice off embedding of special token prepended before to each sequence
            residue_embedding = residue_embedding[:, 1:]

            # IN: X = (B x L x F) - OUT: ( B x N x L )
            prediction = predictor(residue_embedding)
            probabilities = toCPU(torch.max(
                F.softmax(prediction, dim=1), dim=1, keepdim=True)[0])
            
            prediction = toCPU(torch.max(prediction, dim=1, keepdim=True)[
                               1]).astype(np.byte)

            # batch-size x seq_len x embedding_dim
            # extra token is added at the end of the seq
            for batch_idx, identifier in enumerate(pdb_ids):
                s_len = seq_lens[batch_idx]
                # Index dim-1 explicitly instead of .squeeze(): squeeze collapses
                # to a 0-D scalar when s_len == 1, breaking the len() assert below.
                pred = prediction[batch_idx, 0, 0:s_len]
                # nanmean + finiteness check: tolerate occasional NaN logits
                # without aborting the whole pipeline; -1 marks the bad row.
                mean_prob = np.nanmean(probabilities[batch_idx, :, 0:s_len])
                if not np.isfinite(mean_prob):
                    print(f"[3di] non-finite confidence for {identifier} (s_len={s_len}); using sentinel prob=-1")
                    prob = -1
                else:
                    prob = int(100 * mean_prob)
                predictions[identifier] = (pred, prob)
                assert s_len == len(predictions[identifier][0]), print(
                    f"Length mismatch for {identifier}: is:{len(predictions[identifier])} vs should:{s_len}")
                if len(predictions) == 1:
                    print(
                        f"Example: predicted for protein {identifier} with length {s_len}: {predictions[identifier]}")
            # print(f"Batch complete - total {count}")

    end = time.time()
    print('\n############# STATS #############')
    print('Total number of predictions: {}'.format(len(predictions)))
    print('Total time: {:.2f}[s]; time/prot: {:.4f}[s]; avg. len= {:.2f}'.format(
        end-start, (end-start)/len(predictions), avg_length))
    print("Writing results now to disk ...")

    # Sort the prediction as the input fasta file only if the name exists
    predictions = {seq_name: predictions[seq_name]
                   for seq_name in seq_dict_names if seq_name in predictions}

    write_predictions(predictions, out_path)
    write_probs(predictions, probs_path)

    return True

def _generate_3di_annotations(map_args):
    input_dir, model_checkpoint_path, families, output_3di_dir, output_probabilities_dir, use_half_precision, input_is_aligned, include_internals = map_args

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    assert not (use_half_precision and device == torch.device("cpu")), print(
        "Running fp16 on CPU is not supported yet")

    for family in families:
        seq_path = Path(os.path.join(input_dir, family + ".txt"))
        out_path = Path(os.path.join(output_3di_dir, family + ".txt"))
        probs_path = Path(os.path.join(output_probabilities_dir, family + ".txt"))

        if input_is_aligned:
            with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_file:
                with open(seq_path, 'r') as msa_file:
                    for line in msa_file:
                        if not line.startswith(">"):
                            line = line.replace("-", "")
                        temp_file.write(line)
                temp_seq_path = temp_file.name
        else:
            temp_seq_path = seq_path

        get_embeddings(
            seq_path=temp_seq_path,
            out_path=out_path,
            probs_path=probs_path,
            model_dir=model_checkpoint_path,
            half_precision=use_half_precision,
            device=device,
            include_internals=include_internals
        )

        if input_is_aligned:
            os.remove(temp_seq_path)

            # Write aligned sequences
            reference_msa = read_msa(seq_path)
            foldseek_sequences = read_msa(out_path)
            foldseek_msa = align_gapless_sequences_to_alignment(sequences=foldseek_sequences, alignment=reference_msa)

            out_path_aligned = Path(os.path.join(output_3di_dir, family + "_aligned.txt"))
            write_msa(foldseek_msa, out_path_aligned)

        secure_parallel_output(out_path, family)
        secure_parallel_output(probs_path, family)

@peint_caching.cached_parallel_computation(
    parallel_arg="families",
    exclude_args=["num_processes", "input_is_aligned"],
    output_dirs=[
        "output_3di_dir",
        "output_probabilities_dir"
    ],
    write_extra_log_files=True,
)
def generate_3di_annotations(
    input_dir,
    model_checkpoint_path,
    families,
    output_3di_dir = None,
    output_probabilities_dir = None,
    num_processes: int = 1,
    use_half_precision: bool = True,
    input_is_aligned: bool = False,
    include_internals: bool = False
):
    map_args = [
        [
            input_dir,
            model_checkpoint_path,
            get_process_args(process_rank, num_processes, families),
            output_3di_dir,
            output_probabilities_dir,
            use_half_precision,
            input_is_aligned,
            include_internals
        ]
        for process_rank in range(num_processes)
    ]

    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(
                tqdm.tqdm(
                    pool.imap(_generate_3di_annotations, map_args),
                    total=len(map_args),
                )
            )
    else:
        list(
            tqdm.tqdm(
                map(_generate_3di_annotations, map_args),
                total=len(map_args),
            )
        )