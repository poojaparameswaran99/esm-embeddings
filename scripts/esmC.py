
import torch
import os
import h5py
import sys
sys.path.append(os.path.expanduser('~/soderlinglab/utils'))
from seqs import write_fasta
import pandas as pd
from pathlib import Path
from esm.sdk.forge import ESM3ForgeInferenceClient
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import time
from functools import partial


def load_and_write_fasta(config, save_type):
    model_name = config['esm'] ### 5th model. esm2_t33_650M_UR50D , esm2_t48_15B_UR50D, esm1v_t33_650M_UR90S_5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pd.read_csv(config['seqfile'] ).dropna(subset = [config['IDcol'], config['seqcol']]).reset_index(drop=True)
    if '6b' in model_name: ## max seq length 2048
        data[config['seqcol']] = data[config['seqcol']].apply(lambda x: x[:2046] if len(x)> 2046 else x)
    id_seqs =  dict(data[[config['IDcol'], config['seqcol']]].drop_duplicates().itertuples(index=False))
    save_type='pt' if any(s for s in save_type.lower() if s in 'pt') else 'h5'
    print(f'parsing {len(id_seqs)} after dropping duplicates')
    check_existence_type = partial(check_existence, save_type=save_type, config=config)
    id_seqs = dict(filter(check_existence_type, id_seqs.items()))
    fasta_file_path = f'/cwork/pkp14/{config["project_name"]}/{Path(config["seqfile"]).stem}_{config["seqtype"]}.fasta'
    print(f'parsing {len(id_seqs)} after dropping existing IDs')
    write_fasta.to_fasta(id_seqs=id_seqs, outfasta=fasta_file_path)
    print(f'fasta written to: {fasta_file_path}')
    return fasta_file_path

def check_existence(iddandseq, save_type, config):
    idd, seq = iddandseq
    filename=Path(config['output_dir']) / f"{idd}.{save_type}"
    if filename.exists():
        return False
    return True

def esmc_collate_fn(client, device='cuda', average_pooling=True):
    # from esm.models.esmc import ESMC
    # from esm.sdk.api import ESMProtein, LogitsConfig
    ## call model
    # process batch
    def batch_converter(batch):
        ids, seqs = zip(*batch)
        final_embeddings = []
        for i, (idd, seq) in enumerate(batch):
            if i%2000==0:
                time.sleep(1)
            # create esm protein s (still in sequence format, w any other possibilites parse, structure, function, etc.)
            token=ESMProtein(sequence=seq)
            # into tensor (tokenizing here)
            protein_tensor= client.encode(token)
            # forward pass 
            logit=client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
            # extract embeddings — strip BOS (pos 0) and EOS (pos -1) special tokens
            # so that position i in the saved tensor corresponds to sequence residue i.
            embedding=logit.embeddings.squeeze().cpu()[1:-1]
            if average_pooling:
                embedding=logit.embeddings.squeeze().cpu()[1:-1].mean(0)
            final_embeddings.append(embedding)
        assert len(ids) == len(final_embeddings), f'length of ids ({len(ids)}) and embeddings ({len(final_embeddings)}) don"t match!'
        out = list(zip(ids, final_embeddings))
        
        return out

    
    # return the inner function to call upon each batch
    return batch_converter


def esmc_embeddings(config, average_pooling=True,
                    tokens_per_batch=4096, save_type='torch'):
    model_name = config['esm'] ### 5th model. esm2_t33_650M_UR50D , esm2_t48_15B_UR50D, esm1v_t33_650M_UR90S_5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fasta_file = load_and_write_fasta(config, save_type)
    output_dir=Path(config['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(tokens_per_batch, extra_toks_per_seq=1)
    ##     
    if '6b' in model_name.lower():
        client = ESM3ForgeInferenceClient(model="esmc-6b-2024-12", url="https://forge.evolutionaryscale.ai", 
                                          token="2XZ7Nr5BT3tl9mjTxz5aJB")
    else:
        client = ESMC.from_pretrained(model_name).to(device)
        client.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        collate_fn=esmc_collate_fn(client=client, device=device, 
                                   average_pooling=average_pooling), 
        batch_sampler=batches
    )
    for batch_idx, batch in enumerate(data_loader):
        print(f'Processing batch {batch_idx + 1} of {len(batches)}')
        for idd, embeddings in batch:
            if save_type.lower() == 'torch':
                filename = output_dir / f"{idd}.pt"
                if not config['overwrite']:
                    if filename.exists():
                        continue
                torch.save({'mean_representations': embeddings}, filename)
            else:
                filename = output_dir / f"{idd}.h5"
                if not config['overwrite']:
                    if filename.exists():
                        continue
                with h5py.File(output_dir / f"{idd}.h5", 'w') as file:
                    file.create_dataset('mean_representations', 
                                        data=embeddings)
    os.remove(fasta_file)
    return


class FastaBatchedDataset(object):
    def __init__(self, sequence_labels, sequence_strs):
        self.sequence_labels = list(sequence_labels)
        self.sequence_strs = list(sequence_strs)

    @classmethod
    def from_file(cls, fasta_file):
        sequence_labels, sequence_strs = [], []
        cur_seq_label = None
        buf = []

        def _flush_current_seq():
            nonlocal cur_seq_label, buf
            if cur_seq_label is None:
                return
            sequence_labels.append(cur_seq_label)
            sequence_strs.append("".join(buf))
            cur_seq_label = None
            buf = []

        with open(fasta_file, "r") as infile:
            for line_idx, line in enumerate(infile):
                if line.startswith(">"):  # label line
                    _flush_current_seq()
                    line = line[1:].strip()
                    if len(line) > 0:
                        cur_seq_label = line
                    else:
                        cur_seq_label = f"seqnum{line_idx:09d}"
                else:  # sequence line
                    buf.append(line.strip())

        _flush_current_seq()

        assert len(set(sequence_labels)) == len(
            sequence_labels
        ), "Found duplicate sequence labels"

        return cls(sequence_labels, sequence_strs)

    def __len__(self):
        return len(self.sequence_labels)

    def __getitem__(self, idx):
        return self.sequence_labels[idx], self.sequence_strs[idx]

    def get_batch_indices(self, toks_per_batch, extra_toks_per_seq=0):
        sizes = [(len(s), i) for i, s in enumerate(self.sequence_strs)]
        sizes.sort()
        batches = []
        buf = []
        max_len = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            max_len = 0

        for sz, i in sizes:
            sz += extra_toks_per_seq
            if max(sz, max_len) * (len(buf) + 1) > toks_per_batch:
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(i)

        _flush_current_buf()
        return batches
