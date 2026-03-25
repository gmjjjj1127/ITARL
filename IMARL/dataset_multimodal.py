# 文件名: dataset_multimodal.py
import os.path as osp
import numpy as np
import torch
import json
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
import pandas as pd

from dataset import seq_encoding, match_pair
from utils import get_3d_structure, compute_bonds

class MultiModalData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        # 修复 Batching 逻辑，确保不同图在 Batch 中索引正确递增
        if key == 'edge_index':
            return self.x.size(0)
        if key == 'edge_index_atom': 
            return self.x_atom.size(0)
        if key == 'atom_to_nuc_map':
            return self.x.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

class CovidVaccineMultiModal(InMemoryDataset):
    def __init__(self,
        root: str,
        transform = None,
        pre_transform = None,
    ):
        self.name = 'covid'
        self.include_bpp = True
        self.max_seq_len = 130 
        super(CovidVaccineMultiModal, self).__init__(root, transform, pre_transform)
        # 加载处理好的数据
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # [修复] 只使用 train.json，因为 test.json 没有标签
        return ['train.json'] 

    @property
    def processed_file_names(self):
        # 建议修改文件名以强制重新生成，或者请手动删除原来的 data.pt
        return ['data_clean.pt']
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name)

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name + '_multimodal')

    def download(self):
        pass

    def process(self):
        data_list = []
        sequences_1d = [] 

        # 遍历原始文件 (现在只有 train.json)
        for raw_file in self.raw_file_names:
            file_path = osp.join(self.raw_dir, raw_file)
            if not osp.exists(file_path):
                print(f"Warning: {file_path} not found.")
                continue

            with open(file_path, 'r') as file:
                for line in tqdm(file, desc=f"Processing {raw_file}"):
                    item = json.loads(line)
                    
                    # 过滤信号信噪比过低的数据
                    SN_filter = int(item.get('SN_filter', 1))
                    if SN_filter != 1:
                        continue

                    # [修复] 核心逻辑：如果数据没有标签，直接跳过
                    if 'reactivity' not in item:
                        continue

                    id = item['id']
                    sequence = item['sequence']
                    seq_length = item['seq_length']
                    seq_scored = item['seq_scored']

                    # --- 1. 准备 1D 数据和标签 ---
                    sequences_1d.append(sequence)
                    
                    reactivity = item['reactivity']
                    deg_Mg_pH10 = item['deg_Mg_pH10']
                    deg_Mg_50C = item['deg_Mg_50C']
                    
                    targets = []
                    label_mask = []
                    
                    for j in range(seq_length):
                        # 只对 scored 长度内的序列进行监督
                        if j < seq_scored:
                            label_mask.append(True)
                            targets.append([reactivity[j], deg_Mg_pH10[j], deg_Mg_50C[j]])
                        else:
                            label_mask.append(False)
                            # Padding 区域填 0，但 mask 为 False 所以不算 loss
                            targets.append([0.0, 0.0, 0.0]) 
                    
                    y = torch.tensor(targets, dtype=torch.float)
                    mask = torch.tensor(label_mask, dtype=torch.bool)

                    # --- 2. 准备 2D 图数据 ---
                    structure = item['structure']
                    pair_info = match_pair(structure)
                    predicted_loop_type = item['predicted_loop_type']
                    
                    # 检查 BPP 文件是否存在
                    bpp_path = osp.join(self.raw_dir, 'bpps', id + '.npy')
                    if not osp.exists(bpp_path):
                        continue # 缺失文件则跳过

                    bpps = np.load(bpp_path)
                    bpps_sum = bpps.sum(axis=0)
                    bpps_nb = (bpps > 0).sum(axis=0) / seq_length
                    # 使用预计算的均值和方差进行归一化
                    bpps_nb_mean = 0.053428839952351155
                    bpps_nb_std = 0.04034970027890967
                    bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std

                    node_features_2d = []
                    edge_idx_2d = []
                    edge_feat_2d = []
                    paired_nodes_2d = {}

                    for j in range(seq_length):
                        node_features_2d.append(self.get_2d_node_features(sequence[j], predicted_loop_type[j], bpps_sum[j], bpps_nb[j]))

                        if j + 1 < seq_length:
                            f1, f2 = self.get_2d_edge_features_seq()
                            self.add_edges(edge_idx_2d, edge_feat_2d, j, j+1, f1, f2)

                        if pair_info[j] != -1:
                            if pair_info[j] not in paired_nodes_2d:
                                paired_nodes_2d[pair_info[j]] = [j]
                            else:
                                paired_nodes_2d[pair_info[j]].append(j)

                    for pair in paired_nodes_2d.values():
                        bpps_value = bpps[pair[0], pair[1]]
                        f1, f2 = self.get_2d_edge_features_pair(bpps_value)
                        self.add_edges(edge_idx_2d, edge_feat_2d, pair[0], pair[1], f1, f2)
                    
                    x = torch.tensor(node_features_2d, dtype=torch.float)
                    edge_index = torch.tensor(edge_idx_2d, dtype=torch.long).t().contiguous()
                    edge_attr = torch.tensor(edge_feat_2d, dtype=torch.float)

                    # --- 3. 准备 3D 图数据 ---
                    pdb_path = osp.join(self.raw_dir, 'pdb', id, 'unrelaxed_model.pdb')
                    if not osp.exists(pdb_path):
                        continue # 缺失 PDB 则跳过

                    try:
                        x_atom, pos_atom, recon_seq, atom_to_nuc_map = get_3d_structure(pdb_path, full_sequence=sequence)
                    except Exception as e:
                        print(f"Error parsing PDB {id}: {e}")
                        continue
                    
                    # 简单校验序列长度，确保对齐
                    if len(recon_seq) != len(sequence):
                        # 如果 PDB 解析出的序列和 JSON 中的不一致，跳过以防对齐错误
                        continue
                        
                    edge_index_atom = compute_bonds(pos_atom, 1.6)
                    
                    # --- 4. 创建 Data 对象 ---
                    data = MultiModalData(
                        id=id,
                        sequence=sequence,
                        y=y,
                        mask=mask,
                        
                        # 2D features
                        x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        
                        # 3D features
                        x_atom=x_atom,
                        pos_atom=pos_atom,
                        edge_index_atom=edge_index_atom,
                        atom_to_nuc_map=atom_to_nuc_map
                    )
                    data_list.append(data)

        if len(data_list) == 0:
            raise RuntimeError("No valid data processed. Please check if train.json contains labeled data and if 'pdb' folder exists.")

        print(f"Successfully processed {len(data_list)} samples.")

        # --- 5. 1D 序列 Batch 处理 ---
        padded_sequences, seq_masks = seq_encoding(sequences_1d)
        for i, data in enumerate(data_list):
            data.padded_sequences = padded_sequences[i].unsqueeze(0)
            data.seq_masks = seq_masks[i].unsqueeze(0)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0], pickle_protocol=5) 

    # --- 辅助函数 ---
    def get_2d_node_features(self, sequence, predicted_loop_type, bpps_sum, bpps_nb):
        features = [
            sequence == 'A', sequence == 'C', sequence == 'G', sequence == 'U',
            predicted_loop_type == 'S', predicted_loop_type == 'M',
            predicted_loop_type == 'I', predicted_loop_type == 'B',
            predicted_loop_type == 'H', predicted_loop_type == 'E',
            predicted_loop_type == 'X'
        ]
        if self.include_bpp:
            features.extend([bpps_sum, bpps_nb])
        return features
        
    def add_edges(self, edge_index, edge_features, node1, node2, feature1, feature2):
        edge_index.append([node1, node2])
        edge_features.append(feature1)
        edge_index.append([node2, node1])
        edge_features.append(feature2)

    def get_2d_edge_features_seq(self):
        f1 = [0, 1]
        f2 = [0, -1]
        if self.include_bpp:
            f1.append(1)
            f2.append(1)
        return f1, f2

    def get_2d_edge_features_pair(self, bpps_value):
        f1 = [1, 0]
        f2 = [1, 0]
        if self.include_bpp:
            f1.append(bpps_value)
            f2.append(bpps_value)
        return f1, f2


class RNAMultiModal(InMemoryDataset):
    """
    用于 Tcribo 和 Fungal 数据集的多模态数据集
    """
    def __init__(self,
        root: str,
        name: str, # 'tcribo' or 'fungal'
        transform = None,
        pre_transform = None,
    ):
        self.name = name
        self.method = 'eternafold' 
        self.include_bpp = True
        
        # 根据原始 dataset.py 设置 BPP 统计数据
        if self.name == 'tcribo':
            self.bpps_nb_mean = 0.12498024106025696
            self.bpps_nb_std = 0.07428042590618134
            self.max_seq_len = 130 
        elif self.name == 'fungal':
            self.bpps_nb_mean = 0.062242209911346436
            self.bpps_nb_std = 0.039426613599061966
            self.max_seq_len = 3100 
        
        super(RNAMultiModal, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        if self.name == 'tcribo':
            return ['Tc-Riboswitches.csv'] 
        elif self.name == 'fungal':
            return ['fungal_expression.csv']
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name)

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name + '_multimodal')

    def download(self):
        pass

    def process(self):
        data_list = []
        sequences_1d = []
        skipped_count = 0 
        
        csv_path = osp.join(self.raw_dir, self.raw_file_names[0])
        # 兼容性读取
        if not osp.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at {csv_path}")

        df = pd.read_csv(csv_path)
        print(f"Processing {self.name} dataset: {len(df)} raw samples")
        
        for index, row in tqdm(df.iterrows(), total=len(df)):
            id = str(row['id'])
            sequence = row['sequence']
            # 清洗序列
            sequence = sequence.upper().replace('T', 'U')
            seq_length = len(sequence)
            
            label = float(row['label'])
            split = row['split'] 
            
            # [核心逻辑] 过滤：检查 3D 结构是否存在
            pdb_path = osp.join(self.raw_dir, 'pdb', id, 'unrelaxed_model.pdb')
            if not osp.exists(pdb_path):
                skipped_count += 1
                continue

            # [辅助逻辑] 检查 2D 结构是否存在 (BPP)
            bpp_path = osp.join(self.raw_dir, self.method, id, 'bpp.npy')
            if not osp.exists(bpp_path):
                # print(f"Warning: BPP not found for {id}, skipping.")
                skipped_count += 1
                continue
                
            # --- 1. 加载 3D 数据 ---
            try:
                x_atom, pos_atom, recon_seq, atom_to_nuc_map = get_3d_structure(pdb_path, full_sequence=sequence)
            except Exception as e:
                print(f"Warning: Failed to parse PDB for {id}: {e}, skipping.")
                skipped_count += 1
                continue

            # 序列长度校验
            if len(recon_seq) != len(sequence):
                # 如果 PDB 序列和 CSV 序列不一致，跳过
                skipped_count += 1
                continue

            edge_index_atom = compute_bonds(pos_atom, 1.6)

            # --- 2. 加载 2D 数据 ---
            bpps = np.load(bpp_path)
            bpps_sum = bpps.sum(axis=0)
            bpps_nb = (bpps > 0).sum(axis=0) / seq_length
            bpps_nb = (bpps_nb - self.bpps_nb_mean) / self.bpps_nb_std
            
            structure_path = osp.join(self.raw_dir, self.method, id, 'structure.txt')
            if osp.exists(structure_path):
                with open(structure_path, 'r') as f:
                    structure = f.read().strip()
            else:
                structure = '.' * seq_length
                
            pair_info = match_pair(structure)
            
            x_features = [] 
            edge_idx_list = []
            edge_feat_list = []
            paired_nodes = {}
            
            for j in range(seq_length):
                x_features.append(self.get_node_features(sequence[j], bpps_sum[j], bpps_nb[j]))
                if j + 1 < seq_length:
                    f1, f2 = self.get_edge_features_seq()
                    self.add_edges(edge_idx_list, edge_feat_list, j, j+1, f1, f2)
                if pair_info[j] != -1:
                    if pair_info[j] not in paired_nodes:
                        paired_nodes[pair_info[j]] = [j]
                    else:
                        paired_nodes[pair_info[j]].append(j)
            
            for pair in paired_nodes.values():
                bpps_value = bpps[pair[0], pair[1]]
                f1, f2 = self.get_edge_features_pair(bpps_value)
                self.add_edges(edge_idx_list, edge_feat_list, pair[0], pair[1], f1, f2)
            
            x = torch.tensor(x_features, dtype=torch.float)
            edge_index = torch.tensor(edge_idx_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_feat_list, dtype=torch.float)
            
            # --- 3. 1D 数据 ---
            sequences_1d.append(sequence)
            y = torch.tensor([[label]], dtype=torch.float)
            # Mask设为True，适配 loader
            mask = torch.ones(1, dtype=torch.bool) 

            # --- 4. 构建 Data ---
            data = MultiModalData(
                id=id,
                sequence=sequence,
                y=y,
                split=split,
                mask=mask, 
                
                # 2D
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                
                # 3D
                x_atom=x_atom,
                pos_atom=pos_atom,
                edge_index_atom=edge_index_atom,
                atom_to_nuc_map=atom_to_nuc_map
            )
            data_list.append(data)
        
        print(f"Processing complete. Total samples: {len(data_list)}. Skipped (missing 3D/2D/Mismatch): {skipped_count}")

        # --- 5. 1D Batch 处理 ---
        if len(data_list) > 0:
            padded_sequences, seq_masks = seq_encoding(sequences_1d)
            for i, data in enumerate(data_list):
                data.padded_sequences = padded_sequences[i].unsqueeze(0)
                data.seq_masks = seq_masks[i].unsqueeze(0)
            
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0], pickle_protocol=5)
        else:
            print("Error: No data processed!")

    def get_node_features(self, sequence, bpps_sum, bpps_nb):
        features = [
            sequence == 'A', sequence == 'C', sequence == 'G', sequence == 'U'
        ]
        if self.include_bpp:
            features.extend([bpps_sum, bpps_nb])
        return features 

    def add_edges(self, edge_index, edge_features, node1, node2, feature1, feature2):
        edge_index.append([node1, node2])
        edge_features.append(feature1)
        edge_index.append([node2, node1])
        edge_features.append(feature2)

    def get_edge_features_seq(self):
        f1 = [0, 1]
        f2 = [0, -1]
        if self.include_bpp:
            f1.append(1)
            f2.append(1)
        return f1, f2

    def get_edge_features_pair(self, bpps_value):
        f1 = [1, 0]
        f2 = [1, 0]
        if self.include_bpp:
            f1.append(bpps_value)
            f2.append(bpps_value)
        return f1, f2