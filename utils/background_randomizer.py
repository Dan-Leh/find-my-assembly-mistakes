import torch
import numpy as np
from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229,0.224,0.225]

def replace_background(img:torch.Tensor, 
                    segmask:np.ndarray, 
                    bg_img:Image.Image, 
                    prev_tf, img_size:tuple,
                    img_name:str) -> torch.Tensor:
    ''' Return assembly object pasted on random background.
    
    Arguments:
        img (tensor): tensor corresponding to original anchor or sample
        segmask (tuple of np.ndarray): segmentation mask of the anchor or
            sample images used for cutting out the assembly object
        bg_img (PIL Image): background image to use
        prev_tf (custom Transform class): the transform applied to image 
            pairs prior to becoming eg. ROI crops, used to transform the 
            segmentation masks so they are aligned with the images
        img_size (tuple): height and width of output image
        img_name (str): whether this is the 'anchor' or 'sample' image
        
    Returns:
        merged_img (tensor): the anchor or sample image with a background
            from the COCO unlabeled 2017 dataset
    '''
    
    transforms4bgimgs = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(img_size, interpolation=InterpolationMode.BILINEAR),
        v2.Normalize(mean=norm_mean, std=norm_std)
        ])

    # transform the segmentation mask to the same geometry as the object
    if img_name == 'anchor':
        segmask = prev_tf(segmask, 'label')
    elif img_name == 'sample':
        segmask = prev_tf(segmask, 'segmask_sample')
    
    # cut out assembly object and paste on background image
    merged_img = transforms4bgimgs(bg_img)
    foreground_mask = (segmask==1).expand_as(merged_img)
    merged_img[foreground_mask] = img[foreground_mask]
    
    return merged_img
    