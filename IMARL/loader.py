import random
import torch
from torch_geometric.loader import DataLoader

from utils import random_split
from dataset import *




def batch_loader(dataset_name, dataset, batch_size, train_ratio, val_ratio, test_ratio):
    if 'tcribo' in dataset_name or 'fungal' in dataset_name:
        print("Using the fixed split for RNA dataset")
        train_dataset = [data for data in dataset if data.split == 'train']
        val_dataset = [data for data in dataset if data.split == 'val']
        test_dataset = [data for data in dataset if data.split == 'test']

        if train_ratio + val_ratio + test_ratio < 1:  # 0.2, 0.4, 0.6
            l = len(train_dataset) + len(val_dataset) + len(test_dataset)
            train_dataset = random.sample(train_dataset, int(l * train_ratio))
        print('Length:', len(train_dataset), len(val_dataset), len(test_dataset))

    else:
        train_idx, val_idx, test_idx = random_split(data=dataset,
                                                    train_ratio=train_ratio,
                                                    val_ratio=val_ratio,
                                                    test_ratio=test_ratio)
        
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader, test_loader