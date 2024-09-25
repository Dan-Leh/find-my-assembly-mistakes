''' This code was adapted from 'The Change You Want to See':
https://github.com/ragavsachdeva/The-Change-You-Want-to-See.git'''

import torch.nn as nn
import torch

from models.network.unet_model import Unet
from models.network.crossattention import CrossAttentionModule
from models.network.loftr_files.multi_headed_attn import MSA

class CYWS(nn.Module):
    def __init__(self, cyws_params, data_path):
        super().__init__()
        n_coam, coam_input_channels, coam_hidden_channels = \
                                                cyws_params['coam_layer_data']
        
        # config variables that were added, to make configs backward compatible
        
        encoder_weights = 'imagenet' if cyws_params['pretrained_encoder'] else None
        
        self.unet_model = Unet(
                    cyws_params['encoder'],
                    decoder_channels=(256, 256, 128, 128, 64),
                    encoder_depth=5,
                    encoder_weights=encoder_weights,
                    num_coam_layers=n_coam,
                    decoder_attention_type=cyws_params['decoder_attn_type'],
                    disable_segmentation_head=False,
                    classes = 2,
                    data_path = data_path
                )
        
        if cyws_params['attention'] == 'msa':
            assert cyws_params['n_SA_heads'] > 0 and cyws_params['n_MSA_layers']>0 \
                and cyws_params['self_attention'] in ['linear', 'full'], \
                "Mistake in configuration of multi-headed self-attention."
            self.add_msa = True
            self.msa_modules = nn.ModuleList(
                [
                    MSA(n_layers=cyws_params['n_MSA_layers'], 
                        d_model=coam_input_channels[i], 
                        nhead=cyws_params['n_SA_heads'], 
                        attention_type=cyws_params['self_attention'])
                    for i in range(n_coam)
                ]
            )
        else: self.add_msa = False
        
        attn_type = cyws_params['attention'] 
        if self.add_msa: attn_type = 'gca'
        self.coattention_modules = nn.ModuleList(
            [
                CrossAttentionModule(
                    input_channels=coam_input_channels[i],
                    hidden_channels=coam_hidden_channels[i],
                    attention_type=attn_type,
                    kernel_size=cyws_params['kernel_sizes'][i]
                    )
                for i in range(n_coam)
            ]
        )
        
        
    def forward(self, reference, sample, vis_CA=False):
        # pass images through encoder
        reference_latent = self.unet_model.encoder(reference)
        sample_latent = self.unet_model.encoder(sample)
        
        # get crossattention block, apply self-attention if in settings
        for i in range(len(self.coattention_modules)):
            if self.add_msa:
                reference_latent[-(i + 1)] = self.msa_modules[i](reference_latent[-(i + 1)])
                sample_latent[-(i + 1)] = self.msa_modules[i](sample_latent[-(i + 1)])
                
            reference_latent[-(i + 1)] = self.coattention_modules[i](reference_latent[-(i + 1)], 
                                                                        sample_latent[-(i + 1)],
                                                                        vis_CA, # whether to save cross-attention maps for visualization 
                                                                        ) 

        decoded_features = self.unet_model.decoder(*reference_latent)
        
        predicted_mask = self.unet_model.segmentation_head(decoded_features)
        
        return predicted_mask