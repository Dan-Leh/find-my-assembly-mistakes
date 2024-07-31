import torch
import torch.nn as nn
from einops import rearrange

from loftr_files.linear_attention import FullAttention, LinearAttention
from loftr_files.position_encoding import PositionEncodingSine

class MSA(nn.Module):
    def __init__(self,
                 n_layers,
                 d_model,
                 nhead,
                 attention_type='linear'):
        super().__init__()
        
        self.pos_encoding = PositionEncodingSine(d_model)
        
        self.MSA_Layers = nn.ModuleList(
            [
                MSA_Layer(d_model, nhead, attention_type)
                for i in range(n_layers)
            ]
        )

    def forward(self, x):

        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        x_pos = rearrange(self.pos_encoding(x), 'n c h w -> n (h w) c')
        
        for MSA_layer in self.MSA_Layers:
            x_pos = MSA_layer(x_pos)
        
        # revert feature map to original whape
        return x_pos.reshape(x.size())
        
        
        

class MSA_Layer(nn.Module): # multi-headed self-attention
    def __init__(self,
                 d_model,
                 nhead,
                 attention_type='linear'):
        super().__init__()
        

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention_type == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        
    def forward(self, features):
        """
        Args:
            features (torch.Tensor): [N, C, H, W]
        """
        bs = features.size(0)
        
        # multi-head attention
        query = self.q_proj(features).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(features).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(features).view(bs, -1, self.nhead, self.dim)
        
        attended_features = self.attention(query, key, value)  # [N, L, (H, D)]
        attended_features = self.merge(attended_features.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        attended_features = self.norm1(attended_features)
        
        # feed-forward network
        attended_features = self.mlp(torch.cat([features, attended_features], dim=2))
        attended_features = self.norm2(attended_features)

        return features + attended_features