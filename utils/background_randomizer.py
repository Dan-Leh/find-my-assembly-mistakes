import torch
import numpy as np
from PIL import Image
from .transforms import Transforms
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229,0.224,0.225]

def randomize_background(imgs:tuple[torch.Tensor, torch.Tensor], 
                         segmasks:list[np.ndarray, np.ndarray], 
                         bg_imgs:list[Image.Image, Image.Image], 
                         prev_tf:Transforms, 
                         img_size:tuple) -> list[torch.Tensor, torch.Tensor]:
    ''' Return assembly object pasted on random background.
    
    Arguments:
        imgs (list of tensors): tensors correspond to original anchor and
            sample images
        segmasks (tuple of np.ndarray): segmentation mask of the anchor and
            sample images used for cutting out the assembly object
        bg_imgs (list of PIL Images): background images to use
        prev_tf (Transform class): the transform applied to image pairs
            prior to becoming eg. ROI crops, used to transform the 
            segmentation masks so they are aligned with the images
        img_size (tuple): height and width of output image
        
    Returns:
        merged_imgs (list of tensors): the anchor and sample images with
            a random background
    '''

    transforms4bgimgs = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(img_size, interpolation=InterpolationMode.BILINEAR),
        v2.Normalize(mean=norm_mean, std=norm_std)
        ])

    merged_imgs = []
    segmasks = [prev_tf(segmasks[0], 'label'),
                prev_tf(segmasks[1], 'segmask_sample')]
    
    for i in range(2):
        background = transforms4bgimgs(bg_imgs[i])
        
        merged = background
        foreground_mask = (segmasks[i]==1).expand_as(background)
        merged[foreground_mask] = imgs[i][foreground_mask]
        merged_imgs.append(merged)
    
    return merged_imgs
    