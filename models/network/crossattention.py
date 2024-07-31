import torch
import torch.nn as nn
from einops import rearrange


class CrossAttentionModule(nn.Module):
    def __init__(self, input_channels, hidden_channels, attention_type, kernel_size=None, mistake_in_local_attn=False):
        super().__init__()
        if attention_type == "coam":
            self.attention_layer = CoAttentionLayer(input_channels, hidden_channels)
        elif attention_type == "noam":
            self.attention_layer = NoAttentionLayer()
        elif attention_type == "local":
            self.attention_layer = LocalAttentionLayer(input_channels, hidden_channels, kernel_size, mistake_in_local_attn)
        elif attention_type == 'argmax':
            self.attention_layer = CoAttentionLayer(input_channels, hidden_channels, argmax=True, make_mistake=mistake_in_local_attn)
        else:
            raise NotImplementedError(f"Unknown attention {attention_type}")

    def forward(self, left_features, right_features, vis_CA, self_attend, same_WqWk):
        weighted_r = self.attention_layer(left_features, right_features, vis_CA, same_WqWk)
        if self_attend: # use the cross-attention weights on fs1
            weighted_l = self.attention_layer(left_features, left_features, vis_CA, same_WqWk)
            left_attended_features = rearrange(
                [weighted_l, weighted_r], "two b c h w -> b (two c) h w"
            )
        else:
            left_attended_features = rearrange(
                [left_features, weighted_r], "two b c h w -> b (two c) h w"
            )
        # right_attended_features = rearrange(
        #     [right_features, weighted_l], "two b c h w -> b (two c) h w"
        # )
        return left_attended_features #, right_attended_features


class CoAttentionLayer(nn.Module):
    def __init__(self, input_channels, hidden_channels, argmax=False, make_mistake=False):
        super().__init__()
        self.reference_dimensionality_reduction = nn.Conv2d(
            input_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.query_dimensionality_reduction = nn.Conv2d(
            input_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.n_times_called = 0
        self.argmax = argmax
        self.make_mistake = make_mistake
    
            
    def forward(self, query_features, key_features, vis_CA, same_WqWk):

        Q = self.query_dimensionality_reduction(query_features)
        if same_WqWk:
            K = self.query_dimensionality_reduction(key_features)
        else: # default for cyws
            K = self.reference_dimensionality_reduction(key_features)
        
        V = rearrange(key_features, "b c h w -> b c (h w)")
        attention_map = torch.einsum("bcij,bckl->bijkl", Q, K) # we now have an attention score corresponding to the dot product between each vector in Q and each vector in K
        original_size = attention_map.shape # to reverse reshape in visualization
        attention_map = rearrange(attention_map, "b h1 w1 h2 w2 -> b h1 w1 (h2 w2)") # attention map contains score at each i,j of how well all the k,l vectors match
        if self.argmax:
            argmax = torch.argmax(attention_map, dim=3)
            device = attention_map.get_device() if attention_map.get_device() >=0 else "cpu" # get_device returns -1 for cpu and 0 for cuda
            attention_map = torch.zeros_like(attention_map).to(device)
            if self.make_mistake: # another mistake temporarily implemented for backward compatibility on models that were trained with mistake
                attention_map[:,:,:,argmax] = 1
            else:
                batch_idx, row_idx, col_idx = torch.meshgrid(
                    torch.arange(attention_map.size(0)),
                    torch.arange(attention_map.size(1)),
                    torch.arange(attention_map.size(2)),
                    indexing='ij')
                attention_map[batch_idx, row_idx, col_idx, argmax] = 1
        else:  # original implementation of attention
            attention_map = nn.Softmax(dim=3)(attention_map)
            
        ############### save attention map for visualizing query later #################
        if vis_CA: # save cross-attention weights for visualization in image pairs
            import numpy as np 
            import os
            save_root = "/shared/nl011006/res_ds_ml_restricted/dlehman/assembly-error-localization/visualize_cross-attention/cross-attention_maps"
            spatial_resolutions = {0: "8x8", 1: "16x16", 2: "32x32"}
            # remove files from previous run
            for i in range(3):
                file_name = f"sample{self.n_times_called}_spatial_res_{spatial_resolutions[i]}.npy"
                save_path = os.path.join(save_root, file_name)
                if not os.path.exists(save_path):
                    break
            map4saving = attention_map.view(original_size).cpu().detach().numpy()
            np.save(save_path, map4saving)
            self.n_times_called += 1
            print(f"SAVING ATTENTION MAP with shape {map4saving.shape}")
        
        ################################################################################
        
        attended_features = torch.einsum("bijp,bcp->bcij", attention_map, V) # this is basically a dot produt that, for each cannel in V, multiplies the attention scores corresponding to each location i,j with the values that those attention scores are attending to, to end up with the features that, regardless of spatial location, matched most strongly with the features at i,j.
        return attended_features


class LocalAttentionLayer(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, make_mistake):
        super().__init__()
        self.reference_dimensionality_reduction = nn.Conv2d(
            input_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.query_dimensionality_reduction = nn.Conv2d(
            input_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.n_times_called = 0
        self.make_mistake = make_mistake
        
        # how many feature vectors to compute cross attention with (the number depends on the network depth)
        input_chans_to_spatial_res = {512: 8, 256: 16, 128: 32}
        self.spatial_res = input_chans_to_spatial_res[input_channels]
        
        self.offset = (kernel_size-1)//2
        
    
    def cross_attention(self, Q, K, V, return_attn_maps=False):
        attention_map = torch.einsum("bc,bckl->bkl", Q, K)
        V = rearrange(V, "b c h w -> b c (h w)")
        original_shape = attention_map.size()
        attention_map = rearrange(attention_map, "b h w -> b (h w)") # attention map contains score at each i,j of how well all the k,l vectors match
        attention_map = nn.Softmax(dim=1)(attention_map)
        attended_features = torch.einsum("bp,bcp->bc", attention_map, V)
        if return_attn_maps:
            attention_map = attention_map.view(original_shape)
            return attended_features, attention_map
        return attended_features
        
            
    def forward(self, query_features, key_features, vis_CA, same_WqWk):

        Q = self.query_dimensionality_reduction(query_features)
        if same_WqWk:
            K = self.query_dimensionality_reduction(key_features)
        else: # default for cyws
            K = self.reference_dimensionality_reduction(key_features)
        V = key_features
        device = V.get_device() if V.get_device() >=0 else 'cpu' # to make sure attended features are on same device
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
                if self.make_mistake: # temporary fix for backward compatibility on the following 4 models: pretrained_enc_ROI_crops_localCA, localCA_mixed_kernels, localCA_larger_kernels, localCA_smaller_kernels
                    k = K[:,:, h_min:h_max, w_min:w_max]
                    v = V[:,:, h_min:h_max, w_min:w_max]
                else: # correct implementation, making sure the last index is included in making the 'kernel' on image 2
                    k = K[:,:, h_min:h_max+1, w_min:w_max+1]
                    v = V[:,:, h_min:h_max+1, w_min:w_max+1]
                if vis_CA:
                    attended_features[:,:,h,w], attn_map = self.cross_attention(q, k, v, return_attn_maps)
                    attn_maps.append(attn_map.cpu().detach().numpy())
                else:
                    attended_features[:,:,h,w] = self.cross_attention(q, k, v)
                
                    
                
        ############### save attention map for visualizing query later #################
        if vis_CA: # save cross-attention weights for visualization in image pairs
            import pickle
            import os
            save_root = "/shared/nl011006/res_ds_ml_restricted/dlehman/assembly-error-localization/visualize_cross-attention/cross-attention_maps"
            spatial_resolutions = {0: "8x8", 1: "16x16", 2: "32x32"}
            
            for i in range(3): # to figure out which spatial resolution we're currently at
                file_name = f"sample{self.n_times_called}_spatial_res_{spatial_resolutions[i]}"
                save_path = os.path.join(save_root, file_name)
                if not os.path.exists(save_path):
                    break
            with open(save_path, "wb") as fp:
                pickle.dump(attn_maps, fp)
            
            self.n_times_called += 1
            print(f"SAVING LOCAL ATTENTION MAPS with legnth {len(attn_maps)}")
        
        ################################################################################
        
        return attended_features
    

class NoAttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query_features, key_features, tmp_fix, tmp_fix2):
        return key_features
    