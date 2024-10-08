''' This code was adapted from and inspired by 'The Change You Want to See':
https://github.com/ragavsachdeva/The-Change-You-Want-to-See.git'''

import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional


class CrossAttentionModule(nn.Module):
    def __init__(self, input_channels:list, hidden_channels:list, 
                 attention_type:str, kernel_size:Optional[int]=None) -> None:
        super().__init__()
        if attention_type == "gca":  # global cross-attention
            self.attention_layer = CrossAttentionLayer(input_channels, 
                                                       hidden_channels)
        elif attention_type == "noam":  # no attention module, i.e. concatenation
            self.attention_layer = NoAttentionLayer()
        elif attention_type == "lca":  # local cross-attention
            self.attention_layer = LocalAttentionLayer(input_channels, 
                                                    hidden_channels, kernel_size)
        else:
            raise NotImplementedError(f"Unknown attention {attention_type}")

    def forward(self, left_features:torch.Tensor, right_features:torch.Tensor, 
                                                    vis_CA:bool) -> torch.Tensor:
        
        weighted_r = self.attention_layer(left_features, right_features, vis_CA)
        left_attended_features = rearrange(
            [left_features, weighted_r], "two b c h w -> b (two c) h w"
        )

        return left_attended_features 


class CrossAttentionLayer(nn.Module):
    def __init__(self, input_channels:list, hidden_channels:list) -> None:
        super().__init__()
        self.reference_dimensionality_reduction = nn.Conv2d(input_channels, 
                hidden_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.query_dimensionality_reduction = nn.Conv2d(input_channels, 
                hidden_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.n_times_called = 0    
            
    def forward(self, query_features:torch.Tensor, key_features:torch.Tensor, 
                                                    vis_CA:bool) -> torch.Tensor:

        Q = self.query_dimensionality_reduction(query_features)
        K = self.reference_dimensionality_reduction(key_features)
        
        V = rearrange(key_features, "b c h w -> b c (h w)")
        attention_map = torch.einsum("bcij,bckl->bijkl", Q, K)  # we now have an attention score corresponding to the dot product between each vector in Q and each vector in K
        original_size = attention_map.shape  # to reverse reshape in visualization
        attention_map = rearrange(attention_map, "b h1 w1 h2 w2 -> b h1 w1 (h2 w2)")  # attention map contains score at each i,j of how well all the k,l vectors match

        attention_map = nn.Softmax(dim=3)(attention_map)
        
        attended_features = torch.einsum("bijp,bcp->bcij", attention_map, V) # this is basically a dot produt that, for each cannel in V, multiplies the attention scores corresponding to each location i,j with the values that those attention scores are attending to, to end up with the features that, regardless of spatial location, matched most strongly with the features at i,j.
        return attended_features


class LocalAttentionLayer(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size:int):
        super().__init__()
        self.reference_dimensionality_reduction = nn.Conv2d(input_channels, 
                hidden_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.query_dimensionality_reduction = nn.Conv2d(input_channels, 
                hidden_channels, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.n_times_called = 0
        
        # how many feature vectors to compute cross attention with 
        # (the number depends on the network depth)
        input_chans_to_spatial_res = {512: 8, 256: 16, 128: 32}
        self.spatial_res = input_chans_to_spatial_res[input_channels]
        self.offset = (kernel_size-1)//2
        
    
    def cross_attention(self, Q, K, V, return_attn_maps=False):
        attention_map = torch.einsum("bc,bckl->bkl", Q, K)
        V = rearrange(V, "b c h w -> b c (h w)")
        original_shape = attention_map.size()
        attention_map = rearrange(attention_map, "b h w -> b (h w)") 
        attention_map = nn.Softmax(dim=1)(attention_map)
        attended_features = torch.einsum("bp,bcp->bc", attention_map, V)
        if return_attn_maps:
            attention_map = attention_map.view(original_shape)
            return attended_features, attention_map
        return attended_features
        
            
    def forward(self, query_features, key_features, vis_CA):

        Q = self.query_dimensionality_reduction(query_features)
        K = self.reference_dimensionality_reduction(key_features)
        V = key_features
        # to make sure attended features are on same device
        device = V.get_device() if V.get_device() >=0 else 'cpu' 
        attended_features = torch.zeros(key_features.size()).to(device)
        
        if vis_CA:
            return_attn_maps = True
            attn_maps = []
        else: return_attn_maps = False
        
        # split sample dimension into regions
        for h in range(key_features.size()[-2]): 
            h_min = max(0, h-self.offset)
            h_max = min(self.spatial_res, h+self.offset)
            for w in range(key_features.size()[-1]):
                w_min = max(0, w-self.offset)
                w_max = min(self.spatial_res, w+self.offset)
                q = Q[:,:, h, w]
                k = K[:,:, h_min:h_max+1, w_min:w_max+1]
                v = V[:,:, h_min:h_max+1, w_min:w_max+1]
                if vis_CA:
                    attended_features[:,:,h,w], attn_map = self.cross_attention(q, k, v, return_attn_maps)
                    attn_maps.append(attn_map.cpu().detach().numpy())
                else:
                    attended_features[:,:,h,w] = self.cross_attention(q, k, v)
        
        return attended_features
    

class NoAttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query_features, key_features, vis_CA):
        return key_features
    