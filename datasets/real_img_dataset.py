
import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils.eval_transforms import EvalTransforms

class RealChangeDataset(Dataset):
    """ Dataset for pairing a synthetic anchor with a real sample image """
    def __init__(self, data_path:str = "", ROI: bool = False):
        ''' 
        Args:
            data_path (str): path to the folder containing the dataset
            ROI (bool): whether to create region-of-interest crops
        '''
        self.path_to_data = data_path
        self.ROI = ROI
        
        # construct a list of image information
        self.pairs_info = []
        for category in os.listdir(data_path):
            labels = self._load_json(os.path.join(category, "labels.json"))
            for img in labels:
                self.pairs_info.append({
                    'sample': os.path.join(data_path, category, img), # path to real sample image
                    'reference': os.path.join('/shared/nl011006/res_ds_ml_restricted/dlehman', 
                                              labels[img]['anchor']), # reference image path
                    'bbox': labels[img]['bbox'],
                    'category': labels[img]['category']
                })              
            
    def _load_json(self, name: str) -> list:
        ''' load json file with specified name from data directory '''
        
        path = os.path.join(self.path_to_data, name)
        with open(path, 'r') as f:
            data = json.load(f)
        f.close()
        return data
        
    def __len__(self):
        return len(self.pairs_info)
        
    def __getitem__(self, idx: int) -> tuple[Image.Image, Image.Image, str]: 
        ''' Return synthetic anchor image, real sample image, and error category '''
        
        # get information about image pairs
        img_info = self.pairs_info[idx]
        ref_img_path = img_info['reference']
        real_img_path = img_info['sample']
        bbox = img_info['bbox']

        ref_img = Image.open(ref_img_path).convert("RGB")
        real_img = Image.open(real_img_path).convert("RGB")

        # crop according to bounding box
        if self.ROI: 
            real_img = real_img.crop((bbox[0], bbox[1], bbox[0]+
                                      bbox[2], bbox[1]+bbox[3]))
        
        # load segmentation mask of anchor, to use for ROI cropping
        segmask_path = ref_img_path.replace('.png', '.instance segmentation.png')
        segmask = np.array(Image.open(segmask_path).convert('L'))  # grayscale
        segmask = (segmask > 0).astype(np.uint8)
        
        # TODO: requires fixing
        tf = EvalTransforms(segmask_ref=segmask, segmask_sample=segmask, real_sample=True,
                        aug_cfg = {'rotation': False, # randomly rotates 0, 90, 180 or 270 degrees
                                    'img_size': (256,256),
                                    'random_crop': False, 
                                    'hflip_probability': 0,
                                    'vflip_probability': 0,
                                    'brightness': 0,
                                    'contrast': 0,
                                    'saturation': 0,
                                    'hue': 0,
                                    'normalization': 'imagenet', # add 'industsynth' later
                                    'max_translation': 0,
                                    'rescale': 1,
                                    'g_kernel_size': 1,
                                    'g_sigma_l': 1,
                                    'g_sigma_h': 1,
                                    'ROI_crops': self.ROI,
                                    'center_roi': True,
                                    'shear': 0},
                         )
        
        ref_img = tf(ref_img, 'reference')
        real_img = tf(real_img, 'sample')
            
        return ref_img, real_img, img_info['category']