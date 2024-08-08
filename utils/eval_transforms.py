import torch
import numpy as np
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode

class EvalTransforms():
    ''' Image preprocessing for model evaluation. '''
    
    def __init__(self, test_type:str, old_size:tuple, img_size:tuple, 
                 norm_type:str, segmask1:np.ndarray, segmask2:np.ndarray):
        '''
        Arguments:
            test_type (str): if it is 'ROI_aligned', the roi cropping function 
                             is used
            old_size (tuple of ints): height and width of 'raw' image
            img_size (tuple of ints): height and width of images when they are
                                      fed to the model.
            norm_type (str): according to which dataset to normalize images
            segmask1 (np.ndarray): foreground/background segmentation of anchor,
                                   used for making roi crops
            segmask1 (np.ndarray): foreground/background segmentation of sample,
                                   used for making roi crops
            '''
        self.test_type = test_type
        
        # get normalization parameters
        if norm_type == 'imagenet':
            self.norm_mean = [0.485, 0.456, 0.406]
            self.norm_std = [0.229,0.224,0.225]
        else:
            raise NotImplementedError(f'No implementation for normalizing '
                                      f'according to {norm_type}')
        
        # functions for converting imgs and label to tensor
        self.img2tensor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
            ])
        self.label2tensor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=False)
            ])
        
        self.centercrop = v2.CenterCrop(min(old_size))
        
        # transforms
        self.transforms4imgs = v2.Compose([
            v2.Resize(img_size, interpolation=InterpolationMode.BILINEAR),
            v2.Normalize(mean=self.norm_mean, std=self.norm_std)
            ])
        self.transforms4label = v2.Resize(img_size, 
                                    interpolation=InterpolationMode.NEAREST)
        
        if test_type.startswith('roi'):
            self.crop_params = {'anchor': self._roi_crop_parameter(segmask1),
                                'sample': self._roi_crop_parameter(segmask2)}
        
    def __call__(self, img, img_name:str):
        ''' Apply transforms to anchor, sample and change mask. '''
        
        if img_name == 'label':
            img = self.label2tensor(img)/255
            if self.test_type.startswith('ROI'):
                p = self.crop_params['anchor']
                img = v2.functional.crop(img, p[0],p[1],p[2],p[3])
            else:
                img = self.centercrop(img)
            img = self.transforms4label(img)
        
        else: # anchor or sample image
            img = self.img2tensor(img)
            if self.test_type.startswith('ROI'):
                p = self.crop_params[img_name]
                img = v2.functional.crop(img, p[0],p[1],p[2],p[3])
            else: img = self.centercrop(img)
            img = self.transforms4imgs(img)
        
        return img
    
    def _roi_crop_parameter(self, segmask: np.ndarray) -> list:
        ''' Get parameters for region of interest (ROI) cropping.
        
        Function that takes the bounding box of the object, and adds 5% margins 
        on both sides of height and width
        
        Arguments:
            segmask (numpy array): segmentation mask of anchor or sample images
        Returns: 
            crop_params (list): [top, left, height, width] ''' 
        
        bbox = self._get_bbox_coordinates(segmask) # left, right, top, bottom
        
        # desired margins: 5% of object to each side of height/width
        x_margin = round(0.05*(bbox[1]-bbox[0]))
        y_margin = round(0.05*(bbox[3]-bbox[2]))
        
        left = bbox[0] - x_margin
        width = bbox[1] + x_margin - left
        top = bbox[2] - y_margin
        height = bbox[3] + y_margin - top

        crop_params = [top, left, height, width]
        return crop_params
    
    def _get_bbox_coordinates(self, segmask:np.ndarray) -> tuple[int,int,int,int]:
        ''' Extract object bounding box coordinates from segmentation mask'''
        
        segmentation = np.where(segmask == 1)
        if len(segmentation)>0 and len(segmentation[0])>0 and len(segmentation[1])>0:
            x_min = int(np.min(segmentation[1]))
            x_max = int(np.max(segmentation[1]))
            y_min = int(np.min(segmentation[0]))
            y_max = int(np.max(segmentation[0]))
            return x_min, x_max, y_min, y_max
        else: 
            raise ValueError("No segmentation is present in the provided image")