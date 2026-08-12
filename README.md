# esm-embeddings

A pipeline for generating ESM / ESM-C protein language-model embeddings for large
protein sets (e.g. the human kinome and substrate proteome), feeding downstream
representational analyses and co-embedding models. It runs on GPU nodes and
stores embeddings as HDF5 (`.h5`), one file per protein.

Given a table of sequences, the pipeline dedups them, embeds each unique protein
with an ESM model, and writes a per-protein embedding — either the full
per-residue matrix (`len × hidden`) or a mean-pooled vector (`1 × hidden`).


---

## Embedding modes

### Single model pass
Each protein is run through the model in one forward pass and its per-residue
representation is extracted. This is the natural mode for embedding a protein at
its true, full length.

### Batched pass (token-budget batching)
To use the GPU efficiently across a set of variable-length proteins, sequences
are grouped into batches under a cap on the **padded** tensor size
(`max_len_in_batch × n_seqs ≤ tokens_per_batch`) rather than a fixed batch count.
Short proteins pack many-per-batch; long proteins fall into small (often size-1)
batches. This keeps GPU memory roughly constant across a mixed-length input and
avoids the underutilization of a fixed batch size. Every sequence lands in
exactly one batch and is never split.

### Truncation (optional — and usually unnecessary)
Historically, sequences longer than a cap (`seq_length`) were hard-truncated at
tokenization, dropping the C-terminal tail. **This is no longer needed:** ESM-2
uses rotary position embeddings (RoPE), so it is not limited to its training
length and can embed essentially arbitrarily long sequences in a single pass —
bounded only by GPU memory (attention cost grows with length). Truncation
remains available as an option but is off the critical path.

### Sliding window (for when a bounded window is wanted)
When you do want to cap the window — to bound GPU memory on very long proteins,
or to keep each pass within the length regime the model was trained on — the
pipeline can slide a window across the sequence instead of truncating:

- ESM-2 was trained on sequences up to **1022** residues, so the window size is
  **1022**.
- Consecutive windows share a **100-residue overlap** (stride = 1022 − 100 =
  922); the final window is anchored to the C-terminus so the tail is always
  covered.
- Each window is embedded independently, so **peak GPU memory stays fixed at the
  ≤1022 level regardless of how long the protein is.**
- To reshape back into the full protein, the per-residue windows are laid out at
  their original positions and **overlapping residues are averaged**, producing a
  single `len × hidden` matrix identical in shape to a full single pass.

A protein shorter than 1022 is a single window — i.e. identical to the plain
single pass.

---

## Validation: windowed vs. full — cosine similarity

To check how faithfully the windowed-and-reshaped embedding reproduces a true
full-length single pass, the two are compared per protein with cosine similarity
(mean-pooled to a vector so a `len × hidden` matrix reduces to a single
comparable vector).

- Proteins ≤ 1022 are single-window, so they are **identical** (cosine = 1.0).
- Longer proteins **diverge**, and the divergence grows roughly with sequence
  length: a longer protein produces more windows, hence more window seams where
  each window only sees local (≤1022) context instead of the whole sequence. So
  cosine tends **downward as length / number of windows increases** — though the
  trend is not strictly monotonic, since it also depends on sequence content.
- In practice the effect is small at the protein (mean-pooled) level — typical
  cosine stays well above 0.95 even for multi-thousand-residue proteins — because
  averaging over the whole sequence washes out the local seam differences. The
  per-residue divergence is larger than the pooled divergence.

Takeaway: windowing trades a little global context for bounded, length-invariant
memory, and at the pooled level the representation is largely preserved.

---

## Other features

- **Dedup + cache-once**: duplicate `(ID, seq)` pairs are dropped before
  embedding, and each protein is written to its own `{ID}.h5`. If that file
  already exists (and `overwrite=False`) it is skipped, so cost scales with the
  number of *unique* proteins and re-runs are idempotent/resumable.
- **HDF5 storage**: one `.h5` per protein — either the full per-residue matrix or
  a mean-pooled vector. Read pattern:
  ```python
  import h5py, torch
  with h5py.File(path, 'r') as f:
      emb = torch.from_numpy(f[list(f.keys())[0]][:])   # len × hidden, or 1 × hidden if pooled
  ```
- **Window aggregation** (`postprocess/`): a standalone step that stitches window
  shards back into full-length per-protein embeddings (the overlap-averaging
  described above), verifying full residue coverage.
- **Timing / profiling**: an optional benchmarking mode records model-load time,
  per-protein inference time, and peak GPU memory as a function of sequence
  length, appended to a CSV — useful for quantifying the memory/time tradeoff
  between the full single pass and the sliding window.

---

## Environments

Two different Python packages both import as `esm` and cannot coexist in one
environment — pick the env by which model you run:

| Script | Package | Provides |
|---|---|---|
| `scripts/esm1_2.py` (ESM-2) | **fair-esm** | `esm.pretrained`, `esm.FastaBatchedDataset` |
| `scripts/esmC.py` (ESM-C / ESM3) | **EvolutionaryScale `esm` SDK** | `esm.models.esmc.ESMC`, `esm.sdk.*` |

(That is why `esmC.py` defines its own `FastaBatchedDataset` — the EvolutionaryScale
SDK does not ship one.)

---

## Layout

```
esm-embeddings/
├── configs                  # Hydra config
├── embed.sh                 # SLURM submission
├── src/src_embeddings.py    # entry point; dispatches ESM-2 vs ESM-C
├── scripts/
│   ├── esm1_2.py            # ESM-2 embedding + sliding window
│   └── esmC.py              # ESM-C embedding
├── postprocess/             # window-shard aggregation
└── docs/ROADMAP.md
```
