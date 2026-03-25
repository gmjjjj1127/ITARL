import os
import os.path as osp
import numpy as np
import pandas as pd
import torch
import json
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from utils import get_3d_structure, compute_bonds
from torch_geometric.utils import to_dense_adj



def match_pair(structure):
    pair = [-1] * len(structure)
    pair_no = -1
    pair_no_stack = []
    for i, c in enumerate(structure):
        if c == '(':
            pair_no += 1
            pair[i] = pair_no
            pair_no_stack.append(pair_no)
        elif c == ')':
            pair[i] = pair_no_stack.pop()
    return pair


sequence_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}

def seq_encoding(sequence):
    numerical_sequences = [torch.tensor([sequence_map[nuc] for nuc in seq]) for seq in sequence]
    padded_sequences = pad_sequence(numerical_sequences, batch_first=True, padding_value=0)
    masks = pad_sequence([torch.ones_like(seq) for seq in numerical_sequences], batch_first=True, padding_value=0).bool()
    return padded_sequences, masks
