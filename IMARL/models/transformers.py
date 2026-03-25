import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool


class TransformerEncoderLayerWithAttention(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super(TransformerEncoderLayerWithAttention, self).__init__(*args, **kwargs)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        output, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(output)
        src = self.norm1(src)
        src = src + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(src)))))
        src = self.norm2(src)
        return src



class TransformerModel(nn.Module):
    def __init__(self, num_tokens, d_model, nhead, num_encoder_layers, dim_feedforward, max_seq_len, out_channels, task='node', pool='mean'):
        super(TransformerModel, self).__init__()
        self.task = task
        self.pool = pool
        self.embedding = nn.Embedding(num_tokens, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        encoder_layer = TransformerEncoderLayerWithAttention(d_model=d_model, 
                                                             nhead=nhead, 
                                                             dim_feedforward=dim_feedforward, 
                                                             batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(in_features=d_model, out_features=out_channels)

    def forward(self, sequence, mask=None, batch=None):
        embedded_sequence = self.embedding(sequence) + self.positional_encoding[:, :sequence.size(1), :]
        output = embedded_sequence
        for layer in self.transformer_encoder.layers:
            output = layer(output, src_key_padding_mask=~mask if mask is not None else None)

        if self.task == 'graph':
            if self.pool == 'mean':
                output = global_mean_pool(output, batch)
            elif self.pool == 'add':
                output = global_add_pool(output, batch)
            elif self.pool == 'max':
                output = global_max_pool(output, batch)
        return self.fc(output)


