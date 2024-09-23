
import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils.eval_transforms import EvalTransforms

class RealChangeDataset(Dataset):
    """ Dataset for pairing a synthetic anchor with a real sample image """
    def __init__(self, data_path:str = "", ROI: bool = False, img_tf: dict = {}):
        ''' 
        Args:
            data_path (str):  path to the folder containing the dataset
            ROI (bool):  whether to create region-of-interest crops
            img_tf (dict):  dictionary containing values for the following 
                        image transforms, i.e. keys of the dict: 'img_size', 
                        'normalization'
        '''
        self.path_to_data = data_path
        self.ROI = ROI
        self.img_tf = img_tf
        
        # construct a list of image information
        self.pairs_info = []
        for category in os.listdir(data_path):
            labels = self._load_json(os.path.join(category, "labels.json"))
            for img in labels:
                self.pairs_info.append({
                    'sample': os.path.join(data_path, category, img),  # real img
                    'anchor': os.path.join(data_path, category, 'anchors', img.replace('.jpg','.camera.png')),  # synthetic anchor
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
        anchor_img_path = img_info['anchor']
        real_img_path = img_info['sample']
        bbox = img_info['bbox']

        anchor_img = Image.open(anchor_img_path).convert("RGB")
        real_img = Image.open(real_img_path).convert("RGB")

        # crop according to bounding box
        if self.ROI: 
            real_img = real_img.crop((bbox[0], bbox[1], bbox[0]+bbox[2], 
                                                        bbox[1]+bbox[3]))
        
        # load segmentation mask of anchor, to use for ROI cropping
        segmask_path = anchor_img_path.replace('.camera.png', '.segmentation.png')
        segmask = np.array(Image.open(segmask_path).convert('L'))  # grayscale
        segmask = (segmask > 0).astype(np.uint8)
        
        tf = EvalTransforms(test_type='roi_aligned', old_size=(0,0), 
                            img_size=self.img_tf['img_size'], 
                            norm_type=self.img_tf['normalization'],
                            segmask1=segmask, segmask2=segmask)
        
        anchor_img = tf(anchor_img, 'anchor')
        real_img = tf(real_img, 'sample', real_img=True)
            
        return anchor_img, real_img, img_info['category']