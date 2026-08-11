import pandas as pd 
import numpy as np 
import torch
import os
import sys
import re
import time
import csv
from datetime import datetime
# from esm.models.esmc import ESMC
# from esm.sdk.api import ESMProtein, LogitsConfig
import h5py

from pathlib import Path

from esm import pretrained, FastaBatchedDataset# locally defined FastaBatchedDataset bc of esm verioning diffs

import hydra
from omegaconf import DictConfig, OmegaConf
sys.path.append(os.path.expanduser('~/soderlinglab/utils/'))
from seqs import write_fasta

def load_and_write_fasta(config, save_type):
    model_name = config['esm'] ### 5th model. esm2_t33_650M_UR50D , esm2_t48_15B_UR50D, esm1v_t33_650M_UR90S_5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if 'csv' in config['seqfile']:
        data = pd.read_csv(config['seqfile'] )
    else:
        file=pd.ExcelFile(config['seqfile'])
        for i, s in enumerate(file.sheet_names):
            if i ==0:
                data = file.parse(s)
                continue
            data = pd.concat([data, file.parse(s)], axis=0)
    if str(config['IDcol']).isdigit() or str(config['seqcol']).isdigit():
        config['IDcol'] = data.columns[config['IDcol']]
        config['seqcol'] = data.columns[config['seqcol']]
    data = data.dropna(subset = [config['IDcol'], config['seqcol']]).reset_index(drop=True)
    id_seqs =  list(data[[config['IDcol'], config['seqcol']]].map(lambda x: x.strip('_').replace('*', ''))\
                .drop_duplicates().itertuples(index=False, name=None))
    embedding_path = config['output_dir']
    if not config['overwrite']:
        for i in reversed(range(len(id_seqs))):
            if os.path.exists(os.path.join(embedding_path, f'{id_seqs[i][0]}.{save_type}')):
                id_seqs.pop(i)
    id_seqs = dict(id_seqs)
    print(f'parsing {len(id_seqs)} after dropping duplicates')
    fasta_file_path = f'/cwork/pkp14/{config["project_name"]}/{Path(config["seqfile"]).stem}_{config["seqtype"]}.fasta'
    write_fasta.to_fasta(id_seqs=id_seqs, outfasta=fasta_file_path)
    print(f'fasta written to: {fasta_file_path}')
    return fasta_file_path

def esm_embeddings(config, tokens_per_batch=3096,
                       seq_length=9000, repr_layers=[33], save_type='h5',
                  average_pool=True):
    # "" save entire thing as one dictionary.
    esm_models = {'esm2_650m': 'esm2_t33_650M_UR50D',
               'esm2_3b':  'esm2_t36_3B_UR50D',
               'esm2_15b': 'esm2_t48_15B_UR50D',
               'esmc_300m': 'esmc_300m',
               'esmc_600m': 'esmc_600m'}
    model_name = esm_models[config['esm']] ### 5th model. esm2_t33_650M_UR50D , esm2_t48_15B_UR50D, esm1v_t33_650M_UR90S_5
    average_pool=config['average_pool']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fasta_file = load_and_write_fasta(config, save_type)
    print(f'fasta exists', os.path.exists(fasta_file))
    if os.path.getsize(fasta_file) == 0:
        print('fasta is empty')
        os.remove(fasta_file)
        return
    output_dir=Path(config['output_dir'])
    repr_layers=[int(re.search(r'^[^_]+_[^\d]*(\d+)', model_name).group(1))]
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()
    print(f'{model_name} loaded & eval mode')
    model = model.to(device)
    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(tokens_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        collate_fn=alphabet.get_batch_converter(seq_length), 
        batch_sampler=batches
    )
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        # labels = ids, strs = seqs, toks = numerical reps
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f'Processing batch {batch_idx + 1} of {len(batches)}')

            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)

            logits = out["logits"].to(device="cpu")
            representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}
    
            for i, label in enumerate(labels):
                entry_id = label.split(';')[0]
                
                filename = output_dir / f"{entry_id}.pt"
                truncate_len = min(seq_length, len(strs[i]))
                result = {}
                if average_pool:
                    result[entry_id] = {
                            layer: t[i, 1 : truncate_len + 1].mean(0).detach().clone()
                            for layer, t in representations.items()
                        }
                else:
                    result[entry_id] = {
                            layer: t[i, 1 : truncate_len + 1].detach().clone()
                            for layer, t in representations.items()
                        }
                wr = {f'{entry_id}_representation': result[entry_id][repr_layers[0]]}
                if save_type.lower() == 'pt':
                    filename = output_dir / f"{entry_id}.pt"
                    if  not config['overwrite']:
                        if filename.exists():
                            continue
                    torch.save(wr, filename)
                else:
                    filename = output_dir / f"{entry_id}.h5"
                    if not config['overwrite']:
                        if filename.exists():
                            continue
                    with h5py.File(output_dir / f"{entry_id}.h5", 'w') as file:
                        file.create_dataset(f'{entry_id}_representation', 
                                            data=wr[f'{entry_id}_representation'])
                        # file['mean_representations'] = wr['mean_representations']
    os.remove(fasta_file)
    return


def make_windows(seq, max_window=1022, overlap=100):
    """Slice `seq` into overlapping windows for sliding-window embedding.

    Returns a list of (start0, subseq) where start0 is the 0-indexed start of
    the window in the parent sequence. If len(seq) <= max_window there is a
    single window covering the whole sequence. Otherwise windows of length
    `max_window` step by stride = max_window - overlap, and the final window is
    right-anchored to the C-terminus so the tail is always covered.

    Example (max_window=1022, overlap=100 -> stride=922):
        starts 0, 922, 1844, ...  ->  labels {ID}_1-1022, {ID}_923-1944, ...
    """
    L = len(seq)
    if L <= max_window:
        return [(0, seq)]
    stride = max_window - overlap
    starts = list(range(0, L - max_window + 1, stride))
    if starts[-1] != L - max_window:
        starts.append(L - max_window)          # right-anchor last window
    return [(s, seq[s:s + max_window]) for s in starts]


def _run_id_from_dir(output_dir):
    """grandparent/parent of the h5 output dir, e.g. 'vt-interview-test/truncated'."""
    p = Path(output_dir)
    return f'{p.parent.name}/{p.name}'


def time_single_protein(fasta_file, esm='esm2_650m', repr_layers=None,
                        device=None, truncation_seq_length=None,
                        output_dir=None, average_pool=False,
                        window=False, max_window=1022, overlap=100,
                        csv_path=None, run_id=None):
    """Load an ESM-2 model (timed) and embed + benchmark proteins one-per-pass.

    Loads the model once (timing the load), then runs every sequence in
    `fasta_file` through the model ONE forward pass at a time, timing each
    inference and recording peak *GPU* memory. Optionally slides a window over
    over-length proteins and saves each piece as its own h5 shard, and appends
    the per-forward timings to a CSV.

    Parameters
    ----------
    fasta_file, esm, repr_layers, device : see below / same as before.
    truncation_seq_length : int or None
        Passed to the batch converter (None = no truncation). Ignored when
        `window=True` (windows are already <= max_window).
    output_dir : str or None
        Save each forward's embedding as `{output_dir}/{label}.h5`
        (dataset `{label}_representation`). None = time only, save nothing.
    average_pool : bool
        Mean-pool over residues when saving (default False = full matrix).
    window : bool
        If True, slide `make_windows(seq, max_window, overlap)` over each
        protein; over-length proteins become multiple shards labelled
        `{ID}_{start1}-{end1}` (1-indexed inclusive). If False, whole protein
        in one pass labelled `{ID}`.
    max_window, overlap : int
        Sliding-window size and residue overlap (defaults 1022 / 100).
    csv_path : str or None
        If given, append one row per forward pass to this CSV (header written
        if the file is new). Columns include run_id, parent_id, length,
        load_seconds, infer_seconds, load_plus_infer_seconds, peak_gpu_mb.
    run_id : str or None
        Identifier for this run in the CSV. Defaults to grandparent/parent of
        `output_dir` (e.g. 'vt-interview-test/truncated'); falls back to the
        model name when nothing is saved.

    Returns
    -------
    dict with keys: model, device, load_seconds, run_id, results.
    """
    esm_models = {'esm2_650m': 'esm2_t33_650M_UR50D',
                  'esm2_3b':  'esm2_t36_3B_UR50D',
                  'esm2_15b': 'esm2_t48_15B_UR50D'}
    model_name = esm_models.get(esm, esm)
    device = torch.device(device) if device is not None else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- time model load ---
    t0 = time.perf_counter()
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()
    model = model.to(device)
    load_seconds = time.perf_counter() - t0
    print(f'[load] {model_name} loaded & eval in {load_seconds:.2f}s on {device}')

    if repr_layers is None:
        repr_layers = [int(re.search(r'^[^_]+_[^\d]*(\d+)', model_name).group(1))]
    batch_converter = alphabet.get_batch_converter(truncation_seq_length)
    dataset = FastaBatchedDataset.from_file(fasta_file)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f'[save] embeddings -> {output_dir} '
              f'(window={window}, max_window={max_window}, overlap={overlap}, '
              f'average_pool={average_pool})')

    if run_id is None:
        run_id = _run_id_from_dir(output_dir) if output_dir is not None else model_name

    # --- prepare CSV (append; write header if new) ---
    csv_fields = ['timestamp', 'run_id', 'model', 'device', 'id', 'parent_id',
                  'length', 'window', 'max_window', 'overlap',
                  'load_seconds', 'infer_seconds', 'load_plus_infer_seconds',
                  'peak_gpu_mb']
    csv_writer = csv_fh = None
    if csv_path is not None:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        new_file = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
        csv_fh = open(csv_path, 'a', newline='')
        csv_writer = csv.DictWriter(csv_fh, fieldnames=csv_fields)
        if new_file:
            csv_writer.writeheader()

    results = []
    # 'load+infer_s' = one-time model-load time + this forward's inference time;
    # 'infer_s' = inference only (load excluded). peak_MB is GPU memory.
    print(f"{'id':<22}{'length':>8}{'load+infer_s':>14}{'infer_s':>10}{'peak_MB':>12}")
    with torch.no_grad():
        for idd, seq in dataset:
            windows = make_windows(seq, max_window, overlap) if window \
                else [(0, seq)]
            for start0, sub in windows:
                label = f'{idd}_{start0 + 1}-{start0 + len(sub)}' if window else idd
                _, _, toks = batch_converter([(label, sub)])
                toks = toks.to(device)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                t1 = time.perf_counter()
                out = model(toks, repr_layers=repr_layers, return_contacts=False)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                infer_seconds = time.perf_counter() - t1
                load_plus_infer_seconds = load_seconds + infer_seconds
                peak_gpu_mb = (torch.cuda.max_memory_allocated() / 1e6
                               if device.type == 'cuda' else float('nan'))
                length = len(sub)
                row = {'timestamp': datetime.now().isoformat(timespec='seconds'),
                       'run_id': run_id, 'model': model_name,
                       'device': str(device), 'id': label, 'parent_id': idd,
                       'length': length, 'window': window,
                       'max_window': max_window, 'overlap': overlap,
                       'load_seconds': round(load_seconds, 4),
                       'infer_seconds': round(infer_seconds, 4),
                       'load_plus_infer_seconds': round(load_plus_infer_seconds, 4),
                       'peak_gpu_mb': round(peak_gpu_mb, 1)}
                results.append(row)
                print(f"{label:<22}{length:>8}{load_plus_infer_seconds:>14.3f}"
                      f"{infer_seconds:>10.3f}{peak_gpu_mb:>12.1f}")
                if csv_writer is not None:
                    csv_writer.writerow(row)
                    csv_fh.flush()

                # save the embedding shard, timed separately from inference
                if output_dir is not None:
                    rep = out['representations'][repr_layers[0]][0, 1:length + 1].to('cpu')
                    if average_pool:
                        rep = rep.mean(0)
                    with h5py.File(output_dir / f'{label}.h5', 'w') as fh:
                        fh.create_dataset(f'{label}_representation',
                                          data=rep.float().numpy())

                del out, toks
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

    if csv_fh is not None:
        csv_fh.close()
        print(f'[csv] appended {len(results)} rows to {csv_path}')

    print(f'\n[load] {model_name}: {load_seconds:.2f}s (one-time) on {device}')
    return {'model': model_name, 'device': str(device),
            'load_seconds': load_seconds, 'run_id': run_id, 'results': results}


###
# with h5py.File(path, 'r') as f:
#     key = list(f.keys())[0]
#     embed = torch.from_numpy(f[key][:])
###
