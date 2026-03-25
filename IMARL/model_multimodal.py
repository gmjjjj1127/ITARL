# filename: model_multimodal.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.utils import to_dense_batch
from models.transformers import TransformerModel
from models.gnns import GCN
from models.egnn import EGNN_Pooling

# --- [Positional Encoding Module] ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) 

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# --- [Attention Modules] ---
class AlignedCrossModalLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super(AlignedCrossModalLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value, key_padding_mask=None, return_attn=False):
        attn_output, attn_weights = self.multihead_attn(
            query, key, value, 
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True
        )
        x = self.norm(query + self.dropout(attn_output))
        x = self.norm_ffn(x + self.ffn(x))
        if return_attn:
            return x, attn_weights
        return x, None

class TriModalFusionBlock(nn.Module):
    def __init__(self, dim, nhead=4, dropout=0.1):
        super(TriModalFusionBlock, self).__init__()
        self.attn_1d_3d = AlignedCrossModalLayer(dim, nhead, dropout)
        self.attn_1d_2d = AlignedCrossModalLayer(dim, nhead, dropout)
        
        self.fusion_proj = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.LayerNorm(dim)
        )

    def forward(self, feat_1d, feat_2d, feat_3d, mask_common, mask_3d=None, ablation_mode=None, return_attn=False):
        if ablation_mode:
            if 'no_3d' in ablation_mode: feat_3d = torch.zeros_like(feat_3d)
            if 'no_2d' in ablation_mode: feat_2d = torch.zeros_like(feat_2d)
            if 'no_1d' in ablation_mode: feat_1d = torch.zeros_like(feat_1d)

        if mask_3d is None: mask_3d = mask_common

        # 1D-3D Attention (使用 mask_3d)
        feat_1d_aware_3d, attn_3d = self.attn_1d_3d(feat_1d, feat_3d, feat_3d, key_padding_mask=mask_3d, return_attn=return_attn)
        
        # 1D-2D Attention (使用 mask_common)
        feat_1d_aware_2d, attn_2d = self.attn_1d_2d(feat_1d, feat_2d, feat_2d, key_padding_mask=mask_common, return_attn=return_attn)
        
        combined = torch.cat([feat_1d, feat_1d_aware_3d, feat_1d_aware_2d], dim=-1)
        fused = self.fusion_proj(combined)
        
        return fused, {'attn_3d': attn_3d, 'attn_2d': attn_2d}

# --- [Main Model Class] ---
class EndToEndRNAFusionModel(nn.Module):
    def __init__(self, d_model_1d, nhead_1d, num_layers_1d, dim_ff_1d, max_seq_len_1d,
                 in_channels_2d, hidden_2d, L_2d,
                 in_channels_3d, hidden_3d, L_atom_3d, L_nt_3d, thres_atom_3d, thres_nt_3d,
                 fusion_dim, out_channels, task='node', pool='mean'):
        super(EndToEndRNAFusionModel, self).__init__()
        self.task = task
        self.pool = pool
        
        # Encoders
        self.encoder_1d = TransformerModel(4, d_model_1d, nhead_1d, num_layers_1d, dim_ff_1d, max_seq_len_1d, d_model_1d, 'node')
        self.encoder_1d.fc = nn.Identity()
        
        self.encoder_2d = GCN(in_channels_2d, hidden_2d, hidden_2d, L_2d, 0.1, 'node')
        
        self.encoder_3d = EGNN_Pooling(in_channels_3d, hidden_3d, hidden_3d, L_atom_3d, L_nt_3d, thres_atom_3d, thres_nt_3d, 'node', pool)
        self.encoder_3d.dec = nn.Identity()

        # Projections
        self.proj_1d = nn.Linear(d_model_1d, fusion_dim)
        self.proj_2d = nn.Linear(hidden_2d, fusion_dim)
        self.proj_3d = nn.Linear(hidden_3d, fusion_dim)
        
        self.norm_1d = nn.LayerNorm(fusion_dim)
        self.norm_2d = nn.LayerNorm(fusion_dim)
        self.norm_3d = nn.LayerNorm(fusion_dim)
        
        # Positional Encoding
        self.shared_pos_encoder = PositionalEncoding(fusion_dim, max_len=max_seq_len_1d + 100)
        
        self.fusion_block = TriModalFusionBlock(dim=fusion_dim, nhead=4, dropout=0.1)
        
        # Main Predictor
        self.predictor = nn.Linear(fusion_dim, out_channels)

        # [新增] 3D Auxiliary Predictor (用于辅助 Loss)
        self.predictor_3d_aux = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, out_channels)
        )

    def forward(self, data, ablation_mode=None, return_attn=False, debug=False):
        # --- 1D Processing ---
        seq_padded = data.padded_sequences.squeeze(1) 
        seq_masks = data.seq_masks.squeeze(1)         
        seq_masks_bool = seq_masks.bool()
        
        feat_1d_padded = self.encoder_1d(seq_padded, mask=seq_masks_bool)
        feat_1d_proj = self.norm_1d(self.proj_1d(feat_1d_padded))
        
        # --- 2D Processing ---
        feat_2d = self.encoder_2d(data.x, data.edge_index, batch=data.batch)
        feat_2d_proj = self.norm_2d(self.proj_2d(feat_2d))
        
        # === [Modality Dropout] ===
        # 仅在训练模式下，随机丢弃 1D 或 2D，强迫模型看 3D
        if self.training:
            # 30% 概率丢弃
            if torch.rand(1).item() < 0.3:
                feat_1d_proj = torch.zeros_like(feat_1d_proj)
            # 30% 概率丢弃 2D
            if torch.rand(1).item() < 0.3:
                feat_2d_proj = torch.zeros_like(feat_2d_proj)

        # --- 3D Processing ---
        if data.x.shape[1] > 6: 
            nt_features = torch.cat([data.x[:, 0:4], data.x[:, 11:13]], dim=-1)
        else: 
            nt_features = data.x
        
        atom_batch_idx = data.batch[data.atom_to_nuc_map]
        feat_3d = self.encoder_3d(data.x_atom, data.pos_atom, data.atom_to_nuc_map, nt_features, atom_batch_idx)
        feat_3d_proj = self.norm_3d(self.proj_3d(feat_3d))
        
        # --- Alignment Strategy ---
        feat_2d_dense, _ = to_dense_batch(feat_2d_proj, data.batch)
        
        full_3d_features = torch.zeros_like(feat_2d_proj) 
        has_3d_data_mask = torch.zeros(feat_2d_proj.size(0), dtype=torch.bool, device=feat_2d_proj.device)
        
        unique_nuc_ids = torch.unique(data.atom_to_nuc_map)
        
        if full_3d_features.shape[0] >= unique_nuc_ids.max() + 1:
             full_3d_features[unique_nuc_ids] = feat_3d_proj
             has_3d_data_mask[unique_nuc_ids] = True 
        else:
             valid_indices = unique_nuc_ids < full_3d_features.shape[0]
             full_3d_features[unique_nuc_ids[valid_indices]] = feat_3d_proj[valid_indices]
             has_3d_data_mask[unique_nuc_ids[valid_indices]] = True 

        feat_3d_dense, _ = to_dense_batch(full_3d_features, data.batch)
        mask_3d_existence, _ = to_dense_batch(has_3d_data_mask, data.batch, fill_value=False)

        # --- Padding to Match 1D ---
        target_len = feat_1d_proj.size(1)
        
        def align_tensor(t, target_l):
            batch_s = t.size(0)
            curr_l = t.size(1)
            if curr_l == target_l: return t
            if curr_l < target_l:
                diff = target_l - curr_l
                if t.dim() == 3: padding = torch.zeros(batch_s, diff, t.size(2), device=t.device, dtype=t.dtype)
                else: padding = torch.zeros(batch_s, diff, device=t.device, dtype=t.dtype)
                return torch.cat([t, padding], dim=1)
            else:
                if t.dim() == 3: return t[:, :target_l, :]
                else: return t[:, :target_l]

        feat_2d_dense = align_tensor(feat_2d_dense, target_len)
        feat_3d_dense = align_tensor(feat_3d_dense, target_len)
        mask_3d_existence = align_tensor(mask_3d_existence, target_len) 
        
        # Positional Encoding
        feat_2d_dense = self.shared_pos_encoder(feat_2d_dense)
        feat_3d_dense = self.shared_pos_encoder(feat_3d_dense)
        
        # Masks
        padding_mask = ~seq_masks_bool 
        if padding_mask.size(1) != target_len:
             padding_mask = padding_mask[:, :target_len]

        mask_for_3d_attn = padding_mask | (~mask_3d_existence)

        # === [新增] 计算 3D Auxiliary Output ===
        # 使用对齐后的 3D 特征进行预测，强制 EGNN 学习
        if self.task == 'node':
            # 只在非 Padding 区域有意义，但为了保持 shape 对齐，我们全算，后面再 mask loss
            aux_output_3d = self.predictor_3d_aux(feat_3d_dense)
        elif self.task == 'graph':
            mask_float = (~padding_mask).unsqueeze(-1).float()
            pooled_3d = (feat_3d_dense * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-9)
            aux_output_3d = self.predictor_3d_aux(pooled_3d)

        # --- Fusion ---
        fused_dense, attn_dict = self.fusion_block(
            feat_1d_proj, 
            feat_2d_dense, 
            feat_3d_dense, 
            mask_common=padding_mask,
            mask_3d=mask_for_3d_attn,
            ablation_mode=ablation_mode,
            return_attn=return_attn
        )
        
        # --- Main Output ---
        if self.task == 'node':
            valid_mask = ~padding_mask
            # 注意：返回的是 dense 的结果，方便外部处理，或者这里直接 mask 也可以
            # 为了配合 run_multimodal 里面使用 data.mask，这里返回 dense 比较好，或者展平
            # 这里保持原逻辑：展平 valid 部分
            output = self.predictor(fused_dense[valid_mask])
            
            # Aux output 也需要展平以匹配 loss 形状
            aux_output_3d = aux_output_3d[valid_mask]

        elif self.task == 'graph':
            mask_float = (~padding_mask).unsqueeze(-1).float()
            if self.pool == 'mean': 
                sum_pooled = (fused_dense * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-9)
                pooled = sum_pooled
            elif self.pool == 'add': 
                pooled = (fused_dense * mask_float).sum(dim=1)
            elif self.pool == 'max':
                fused_dense = fused_dense.clone()
                fused_dense[padding_mask] = -1e9
                pooled, _ = fused_dense.max(dim=1)
            output = self.predictor(pooled)
            
        if return_attn:
            return output, aux_output_3d, attn_dict
        return output, aux_output_3d