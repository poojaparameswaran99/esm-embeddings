#!/usr/bin/env python
"""Aggregate sliding-window embedding shards back into full-length proteins.

Reads shard h5 files named `{ID}_{start1}-{end1}.h5` (1-indexed inclusive
residue ranges, as written by esm1_2.time_single_protein(window=True)), groups
them by parent protein ID, places each shard at its residue span, and averages
overlapping positions to reconstruct one `L x hidden` matrix per protein.

Usage:
    python aggregate_windows.py \
        --shard-dir /cwork/pkp14/vt-interview-test/truncated \
        --out-dir   /cwork/pkp14/vt-interview-test/reconstructed \
        [--average-pool]

Idempotent: overwrites reconstructed files. Verifies every residue is covered
by at least one shard before writing.
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

# {ID}_{start}-{end}; greedy ID so accessions/underscores in the ID are fine.
SHARD_RE = re.compile(r'^(?P<id>.+)_(?P<a>\d+)-(?P<b>\d+)$')


def _read_matrix(h5_path):
    """Return the (only) dataset in an h5 shard as a float32 numpy array."""
    with h5py.File(h5_path, 'r') as f:
        key = list(f.keys())[0]
        return np.asarray(f[key][:], dtype=np.float32)


def group_shards(shard_dir):
    """Map parent_id -> list of (start0, end_exclusive, path)."""
    groups = defaultdict(list)
    for p in sorted(Path(shard_dir).glob('*.h5')):
        m = SHARD_RE.match(p.stem)
        if m:
            pid = m.group('id')
            a, b = int(m.group('a')), int(m.group('b'))   # 1-indexed inclusive
            groups[pid].append((a - 1, b, p))              # -> 0-indexed, exclusive
        else:
            # not a windowed shard: treat whole file as a single full-length piece
            groups[p.stem].append((0, None, p))
    return groups


def stitch(pieces):
    """Average overlapping shards into one full-length matrix."""
    # resolve full length + hidden dim
    mats = {path: _read_matrix(path) for _, _, path in pieces}
    hidden = next(iter(mats.values())).shape[1]
    # for a lone non-windowed piece (end=None), end = its own length
    norm = []
    for start0, end, path in pieces:
        e = end if end is not None else start0 + mats[path].shape[0]
        norm.append((start0, e, path))
    full_len = max(e for _, e, _ in norm)

    acc = np.zeros((full_len, hidden), dtype=np.float64)
    cnt = np.zeros(full_len, dtype=np.int64)
    for start0, e, path in norm:
        mat = mats[path]
        span = e - start0
        if mat.shape[0] != span:
            raise ValueError(f'{path.name}: matrix rows {mat.shape[0]} != '
                             f'span {span} ({start0}-{e})')
        acc[start0:e] += mat
        cnt[start0:e] += 1
    if (cnt == 0).any():
        gaps = np.where(cnt == 0)[0]
        raise ValueError(f'uncovered residues at positions {gaps[:10]} ...')
    full = (acc / cnt[:, None]).astype(np.float32)
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--average-pool', action='store_true',
                    help='mean-pool the reconstructed matrix over residues')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = group_shards(args.shard_dir)
    print(f'found {sum(len(v) for v in groups.values())} shards '
          f'across {len(groups)} proteins in {args.shard_dir}')

    for pid, pieces in sorted(groups.items()):
        full = stitch(pieces)
        if args.average_pool:
            full = full.mean(0)
        with h5py.File(out_dir / f'{pid}.h5', 'w') as fh:
            fh.create_dataset(f'{pid}_representation', data=full)
        print(f'{pid:<16} shards={len(pieces):>3}  '
              f'reconstructed shape={full.shape}')

    print(f'\nreconstructed proteins written to {out_dir}')


if __name__ == '__main__':
    main()
