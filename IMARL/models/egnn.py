import torch
from torch import nn
from torch_geometric.utils import scatter
from utils import compute_bonds_batch
import torch.nn.functional as F
from time import time
from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

def nucleotide_pool(h, coord, nucleotide_id, batch, reduce):
    max_nucleotide_id = nucleotide_id.max().item() + 1
    combined_index = batch * max_nucleotide_id + nucleotide_id
    _, new_index = torch.unique(combined_index, return_inverse=True)

    h = scatter(h, new_index, dim=0, reduce=reduce)
    coord = scatter(coord, new_index, dim=0, reduce=reduce)
    batch_nt = scatter(batch, new_index, dim=0, reduce='mean').long()
    return h, coord, batch_nt

class EGNNLayer(nn.Module):
    def __init__(self, input_nf, hidden, output_nf, act_fn, use_coord, recurrent=True, attention=True, normalize=False, tanh=False):
        super(EGNNLayer, self).__init__()
        self.use_coord = use_coord
        self.recurrent = recurrent
        self.attention = attention
        self.tanh = tanh
        self.normalize = normalize
        edge_coords_nf = 1
        self.epsilon = 1e-8

        # --- 修复核心：兼容字符串类型的 act_fn ---
        self.act_fn = act_fn
        if isinstance(act_fn, str):
            if act_fn.lower() == 'silu' or act_fn.lower() == 'swish':
                self.act_fn = nn.SiLU()
            elif act_fn.lower() == 'relu':
                self.act_fn = nn.ReLU()
            elif act_fn.lower() == 'tanh':
                self.act_fn = nn.Tanh()
            else:
                self.act_fn = nn.SiLU() # 默认
        # -------------------------------------

        self.edge_mlp = nn.Sequential(nn.Linear(input_nf * 2 + edge_coords_nf, hidden),
                                        self.act_fn,
                                        nn.Linear(hidden, hidden),
                                        self.act_fn)

        self.node_mlp = nn.Sequential(nn.Linear(input_nf + hidden, hidden),
                                        self.act_fn,
                                        nn.Linear(hidden, output_nf))
        
        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        
        layer = nn.Linear(hidden, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden, hidden))
        coord_mlp.append(self.act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

    def edge_model(self, source, target, radial):
        out = torch.cat([source, target, radial], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, h, edge_index, edge_feat):
        row, col = edge_index
        agg = scatter_sum(edge_feat, row, dim=0, dim_size=h.size(0))
        out = self.node_mlp(torch.cat([h, agg], dim=1))
        if self.recurrent:
            out += h
        return out
    
    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        agg = scatter_sum(trans, row, dim=0, dim_size=coord.size(0))
        coord += agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm
        return radial, coord_diff

    def forward(self, h, edge_index, coord):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        edge_feat = self.edge_model(h[row], h[col], radial)
        h = self.node_model(h, edge_index, edge_feat)
        
        if self.use_coord:
            coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
            return h, coord
        else:
            return h

class EGNN_Pooling(nn.Module):
    def __init__(self, 
                in_node_nf, 
                hidden, 
                output_nf, 
                L_atom, 
                L_nucleotide, 
                threshold_atom, 
                threshold_nt,
                act_fn=nn.SiLU(), 
                attention=True,
                task='node',
                pool='mean', 
            ):
        super(EGNN_Pooling, self).__init__()
        self.L_atom = L_atom
        self.L_nucleotide = L_nucleotide
        self.threshold_atom = threshold_atom
        self.threshold_nt = threshold_nt
        self.task = task
        self.pool = pool

        self.embedding_atom = nn.Linear(in_node_nf, hidden)
        self.embedding_nt = nn.Linear(hidden + 6, hidden) # 假设输入维度修正

        # 确保 act_fn 传递正确
        for i in range(0, self.L_atom):
            self.add_module("egnn_layer_atom_%d" % i, EGNNLayer(hidden, hidden, hidden, act_fn=act_fn, use_coord=True, recurrent=True, attention=attention, tanh=True))

        for i in range(0, self.L_nucleotide):
            self.add_module("egnn_layer_nt_%d" % i, EGNNLayer(hidden, hidden, hidden, act_fn=act_fn, use_coord=True, recurrent=True, attention=attention, tanh=True))

        # 这里的 act_fn 也要处理一下，如果它被传进来是字符串的话
        if isinstance(act_fn, str):
            if act_fn.lower() == 'silu': act_fn_mod = nn.SiLU()
            elif act_fn.lower() == 'relu': act_fn_mod = nn.ReLU()
            else: act_fn_mod = nn.SiLU()
        else:
            act_fn_mod = act_fn

        self.dec = nn.Sequential(nn.Linear(hidden, hidden),
                                act_fn_mod,
                                nn.Linear(hidden, output_nf))

    def forward(self, h, coord, nucleotide_id, nt_features, batch):
        h = self.embedding_atom(h)
        edge_index = compute_bonds_batch(coord, threshold=self.threshold_atom, batch=batch)

        for i in range(0, self.L_atom):
            h, coord = self._modules["egnn_layer_atom_%d" % i](h, edge_index, coord)

        h, coord, batch_nt = nucleotide_pool(h, coord, nucleotide_id, batch, reduce=self.pool)
        h = torch.cat([h, nt_features], dim=1)
        h = self.embedding_nt(h)
        edge_index = compute_bonds_batch(coord, threshold=self.threshold_nt, batch=batch_nt)

        for i in range(0, self.L_nucleotide):
            h, coord = self._modules["egnn_layer_nt_%d" % i](h, edge_index, coord)

        if self.task == 'graph':
            if self.pool == 'mean':
                h = global_mean_pool(h, batch_nt)
            elif self.pool == 'add':
                h = global_add_pool(h, batch_nt)
            elif self.pool == 'max':
                h = global_max_pool(h, batch_nt)
        h = self.dec(h)
        return h