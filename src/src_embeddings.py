import pandas as pd 
import numpy as np 
import torch

import os
import sys
import re
# from esm.models.esmc import ESMC
# from esm.sdk.api import ESMProtein, LogitsConfig
import h5py

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

## local imports 
sys.path.append(os.path.expanduser('~/soderlinglab/utils'))
from seqs import write_fasta

@hydra.main(version_base=None, config_path="configs/", config_name="embeddings_config")
def main(cfg):
    """ esm pkg in `kolossus` works for esm2"""
    """ esmenv works for esm3/esmc"""
    parse_seqs(cfg)
    return

def parse_seqs(config):
     ## load fasta file
    if 'esm2' in config['esm']:
        from embeddings import esm1_2
        print('starting embeddings extraction for esm2')
        esm1_2.esm_embeddings(config, save_type='h5')
    else:
        from embeddings import esmC
        print('starting embeddings extraction for esmc')
        esmC.esmc_embeddings(config, average_pooling=config.get('average_pool', True), save_type='h5')
    return


###
# with h5py.File(path, 'r') as f:
#     key = list(f.keys())[0]
#     embed = torch.from_numpy(f[key][:])
###
if __name__ == '__main__':
    main()

