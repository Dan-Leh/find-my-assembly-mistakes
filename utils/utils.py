import torch
from torchvision.transforms import v2


def denormalize(tensor:torch.Tensor, normalization_type:str) -> torch.Tensor:
    ''' Reverse normalization of input images to the network for visualization. '''
    
    if normalization_type == 'imagenet':
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229,0.224,0.225]
        
    denorm_sequence = v2.Compose([
    v2.Normalize(mean = [ 0., 0., 0. ],
                std = [1/val for val in norm_std]),
    v2.Normalize(mean = [-val for val in norm_mean],
                std = [ 1., 1., 1. ])
                ])
    return denorm_sequence(tensor)


def remove_train_augmentations(tf_cfg: dict) -> dict:
    ''' Remove train-only augmentations for validation set. '''
    
    val_transforms = tf_cfg.copy()
    for aug in ['hflip_probability', 'vflip_probability', 'brightness', 
                'contrast', 'saturation', 'hue', 'shear']:
        val_transforms[aug] = 0
    for aug in ['g_kernel_size', 'g_sigma_l', 'g_sigma_h']:
        val_transforms[aug] = 1
    for aug in ['rotation']:
        val_transforms[aug] = False
        
    return val_transforms