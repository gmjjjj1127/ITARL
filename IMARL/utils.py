# 文件名: utils.py
import numpy as np
import torch
import random
import warnings
from Bio.PDB import PDBParser
from Bio import pairwise2 

# 忽略 Biopython 的过时警告，保持输出整洁
warnings.filterwarnings("ignore", category=DeprecationWarning, module="Bio.pairwise2")

# --- 训练辅助类 ---

class EarlyStopping:
    def __init__(self, patience=50):
        self.patience = patience
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss >= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

# --- Loss Function ---

def mcrmse_loss(input, target, mask=None):
    if mask is not None:
        input = input[mask]
        target = target[mask]
    mse = torch.mean((input - target) ** 2, dim=0)
    columnwise_rmse = torch.sqrt(mse + 1e-8) 
    return torch.mean(columnwise_rmse)

# --- Graph Helpers ---

def fully_connected_edge_index(num_nodes):
    row = torch.arange(num_nodes).repeat_interleave(num_nodes)
    col = torch.arange(num_nodes).repeat(num_nodes)
    edge_index = torch.stack([row, col], dim=0)
    
    mask = row != col     # Remove self-loops
    edge_index = edge_index[:, mask]
    return edge_index

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def random_split(data, train_ratio, val_ratio, test_ratio):
    length = len(data)
    indices = torch.randperm(length)
    
    num_val = int(val_ratio * length)
    num_test = int(test_ratio * length)
    
    if abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5:
        num_train = length - num_val - num_test
    else:
        num_train = int(train_ratio * length)
    
    train_idx = indices[: num_train]
    val_idx = indices[num_train: num_train + num_val]
    test_idx = indices[num_train + num_val: num_train + num_val + num_test]
    
    assert len(set(train_idx).intersection(set(val_idx))) == 0
    assert len(set(val_idx).intersection(set(test_idx))) == 0
    return train_idx, val_idx, test_idx

def random_split_sparse(data, train_ratio, val_ratio, test_ratio):
    length = len(data)
    indices = torch.randperm(length)
    
    num_train = int(train_ratio * length)
    num_val = int(val_ratio * length)
    num_test = int(test_ratio * length)
    
    train_idx = indices[: num_train]
    val_idx = indices[num_train: num_train + num_val]
    test_idx = indices[num_train + num_val: num_train + num_val + num_test]
    
    assert len(set(train_idx).intersection(set(val_idx))) == 0
    assert len(set(val_idx).intersection(set(test_idx))) == 0
    return train_idx, val_idx, test_idx

# --- 3D Structure Processing (PDB) ---

def atom_type_to_index(atom_type):
    atom_dict = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'P': 4}
    return atom_dict.get(atom_type, -1) 

def get_atomic_number(atom_type):
    atomic_number_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'P': 15}
    return atomic_number_dict.get(atom_type, 0)

# [核心修改逻辑]
# [修改 utils.py]

def get_3d_structure(file_path, full_sequence=None):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('RNA', file_path)

    atom_types = []
    atom_coords = []
    atomic_numbers = []
    
    pdb_sequence_list = []
    pdb_residues = [] 

    for model in structure:
        for chain in model:
            for residue in chain:
                res_name = residue.resname.strip()
                if res_name in ['A', 'U', 'G', 'C']:
                    pdb_sequence_list.append(res_name)
                    pdb_residues.append(residue)

    pdb_sequence_str = "".join(pdb_sequence_list)
    
    # === [CRITICAL FIX] 严格对齐逻辑 ===
    pdb_to_full_map = {} 
    
    if full_sequence is not None:
        # 1. 长度差异过大直接报错，不要硬凑
        if abs(len(pdb_sequence_str) - len(full_sequence)) > 50 and len(pdb_sequence_str) < 10:
             # 如果PDB太短或者差异太大，视为无效数据
             # 返回空，让 dataset.py 去处理跳过
             return torch.empty(0, 2), torch.empty(0, 3), "", torch.empty(0, dtype=torch.long)

        try:
            # 使用 Biopython 对齐
            alignments = pairwise2.align.globalms(full_sequence, pdb_sequence_str, 5, -4, -10, -0.5)
            
            if len(alignments) > 0:
                best_aln = alignments[0]
                seq_full_aln = best_aln.seqA
                seq_pdb_aln = best_aln.seqB
                
                full_ptr = 0
                pdb_ptr = 0
                
                for i in range(len(seq_full_aln)):
                    char_full = seq_full_aln[i]
                    char_pdb = seq_pdb_aln[i]
                    
                    # 只有当两者完全匹配时，才建立映射
                    if char_full == char_pdb and char_full != '-':
                        pdb_to_full_map[pdb_ptr] = full_ptr
                    
                    if char_full != '-': full_ptr += 1
                    if char_pdb != '-': pdb_ptr += 1
            else:
                # [关键修改] 比对失败直接放弃，不要回退到 i=i
                print(f"Alignment failed for {file_path}")
                return torch.empty(0, 2), torch.empty(0, 3), "", torch.empty(0, dtype=torch.long)
                
        except Exception as e:
            print(f"Alignment error: {e}")
            return torch.empty(0, 2), torch.empty(0, 3), "", torch.empty(0, dtype=torch.long)
    else:
        # 没有提供全序列，默认 1:1
        for i in range(len(pdb_sequence_str)): pdb_to_full_map[i] = i

    # --- 提取特征 ---
    atom_nucleotide_index = []
    
    for i, residue in enumerate(pdb_residues):
        # [关键] 只保留映射成功的残基
        if i not in pdb_to_full_map:
            continue
            
        real_seq_index = pdb_to_full_map[i]
        
        for atom in residue:
            atom_types.append(atom.element)
            atom_coords.append(atom.coord)
            atomic_numbers.append(get_atomic_number(atom.element))
            atom_nucleotide_index.append(real_seq_index)

    if len(atom_coords) == 0:
        return torch.empty(0, 2), torch.empty(0, 3), "", torch.empty(0, dtype=torch.long)

    atom_coords = torch.tensor(np.array(atom_coords), dtype=torch.float)
    atom_type_indices = torch.tensor([atom_type_to_index(atom) for atom in atom_types], dtype=torch.long)
    atom_type_one_hot = torch.nn.functional.one_hot(atom_type_indices, num_classes=5).float()
    atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.long).view(-1, 1)
    
    atom_nucleotide_index = torch.tensor(atom_nucleotide_index, dtype=torch.long)
    node_features = torch.cat((atomic_numbers, atom_type_one_hot), dim=1)
    
    return node_features, atom_coords, pdb_sequence_str, atom_nucleotide_index

# --- Bond Computation ---

def compute_bonds(coords, threshold):
    if coords.size(0) == 0: return torch.empty(2, 0, dtype=torch.long)
    coords = coords.unsqueeze(0) 
    distances = torch.norm(coords - coords.transpose(0, 1), dim=-1)
    mask = (distances <= threshold) & (distances > 0) 
    bonds = mask.nonzero(as_tuple=False).t().contiguous()  
    return bonds

def compute_bonds_batch(coords, threshold, batch):
    device = coords.device 
    if batch.numel() == 0: return torch.empty(2, 0, dtype=torch.long, device=device)

    num_graphs = batch.max().item() + 1
    
    # 修复 scatter_add_ 的设备问题
    num_atoms_per_graph = torch.zeros(num_graphs, dtype=torch.long, device=device)
    num_atoms_per_graph.scatter_add_(0, batch, torch.ones_like(batch, device=device))
    
    max_num_atoms = num_atoms_per_graph.max().item()

    padded_coords = torch.zeros((num_graphs, max_num_atoms, 3), dtype=coords.dtype, device=device)
    atom_mask = torch.zeros((num_graphs, max_num_atoms), dtype=torch.bool, device=device)
    
    atom_offsets = torch.cumsum(num_atoms_per_graph, dim=0) - num_atoms_per_graph
    for i in range(num_graphs):
        atom_indices = (batch == i).nonzero(as_tuple=False).squeeze()
        if atom_indices.dim() == 0: atom_indices = atom_indices.unsqueeze(0)
        if atom_indices.numel() > 0:
            padded_coords[i, :atom_indices.size(0)] = coords[atom_indices]
            atom_mask[i, :atom_indices.size(0)] = True
    
    diff = padded_coords.unsqueeze(2) - padded_coords.unsqueeze(1)
    distances = torch.norm(diff, dim=-1)
    
    bond_mask = (distances <= threshold) & (distances > 0) & atom_mask.unsqueeze(1) & atom_mask.unsqueeze(2)
    
    bonds = []
    for i in range(num_graphs):
        bond_indices = bond_mask[i].nonzero(as_tuple=False)
        if bond_indices.numel() > 0:
            bond_indices = bond_indices.t().contiguous()
            bonds.append(torch.stack([bond_indices[0] + atom_offsets[i], bond_indices[1] + atom_offsets[i]]))
    
    if not bonds:
        return torch.empty(2, 0, dtype=torch.long, device=coords.device)
    
    return torch.cat(bonds, dim=1)

# --- Evaluation Metrics (找回了这些缺失的函数) ---

def pearson_correlation(pred, target, mask=None):
    if mask is not None:
        pred = pred[mask].view(-1)
        target = target[mask].view(-1)
    else:
        pred = pred.view(-1)
        target = target.view(-1)

    vx = pred - torch.mean(pred)
    vy = target - torch.mean(target)
    
    numerator = torch.sum(vx * vy)
    denominator = torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))
    
    return numerator / (denominator + 1e-8)

def spearman_correlation(pred, target, mask=None):
    if mask is not None:
        pred = pred[mask].view(-1)
        target = target[mask].view(-1)
    else:
        pred = pred.view(-1)
        target = target.view(-1)

    if pred.numel() < 2: 
        return torch.tensor(0.0).to(pred.device)

    pred_rank = pred.argsort().argsort().float()
    target_rank = target.argsort().argsort().float()

    vx = pred_rank - torch.mean(pred_rank)
    vy = target_rank - torch.mean(target_rank)
    numerator = torch.sum(vx * vy)
    denominator = torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))
    return numerator / (denominator + 1e-8)

def r2_score(pred, target, mask=None):
    if mask is not None:
        pred = pred[mask].view(-1)
        target = target[mask].view(-1)
    else:
        pred = pred.view(-1)
        target = target.view(-1)

    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - pred) ** 2)
    
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return r2