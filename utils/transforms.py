import random 
import torch
import math
import numpy as np
from PIL import Image
from torchvision.transforms import v2


class Transforms(torch.nn.Module):
    """ Transforms to apply same augmentations to anchor and label, while 
    controlling pose difference between anchor and sample. """
    
    def __init__(self, segmask_anchor:np.ndarray, 
                 segmask_sample:np.ndarray, tf_cfg:dict):
        super().__init__()
        '''
        Arguments:
            segmask_anchor (np.ndarray): the foreground/background segmentation mask  
                                         of the anchor image
            segmask_sample (np.ndarray): the foreground/background segmentation mask
                                         of sample image
            tf_cfg (dict): dictionary obtained from config file containing all the
                            parameter values for the image transforms performed
        '''
        
        self.segmask_anchor = segmask_anchor
        self.segmask_sample = segmask_sample
        
        self._check_for_mistake_in_config(tf_cfg)
        self.shear_params = self._get_shear_parameters(tf_cfg['shear'])
        self.rescale_params = self._get_scale_parameters(tf_cfg['rescale'], 
                                                segmask_anchor, segmask_sample)
        self.crop_params = self._get_crop_parameters(tf_cfg, segmask_anchor, 
                                                            segmask_sample)
        
        # randomly choose rotation (multiple of 90 degrees)
        self.n_rot90 = random.choice([0,1,2,3]) if tf_cfg['rotation'] else 0
        
        # get random flip parameters so all images get the same random flip
        hflip_bool = True if random.random()<tf_cfg['hflip_probability'] else False
        vflip_bool = True if random.random()<tf_cfg['vflip_probability'] else False
        
        # get normalization parameters
        if tf_cfg['normalization'] == 'imagenet':
            self.norm_mean = [0.485, 0.456, 0.406]
            self.norm_std = [0.229,0.224,0.225]
        else:
            raise NotImplementedError(f'No implementation for normalizing' \
                                      f'according to {tf_cfg['normalization']}')
            
        # define transforms for converting images to tensor
        self.img2tensor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
            ])
        self.label2tensor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=False)
            ])

        # define transform sequence for image augmentation
        self.transforms4imgs = v2.Compose([
            v2.Resize(tf_cfg['img_size'], 
                      interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
            v2.RandomHorizontalFlip(p = 1 if hflip_bool else 0),
            v2.RandomVerticalFlip(p = 1 if vflip_bool else 0),
            v2.ColorJitter(brightness=tf_cfg['brightness'], 
                            contrast=tf_cfg['contrast'], 
                            saturation=tf_cfg['saturation'], 
                            hue=tf_cfg['hue']),
            v2.GaussianBlur(kernel_size=tf_cfg['g_kernel_size'], 
                            sigma=(tf_cfg['g_sigma_l'], tf_cfg['g_sigma_h'])),
            v2.Normalize(mean=self.norm_mean, std=self.norm_std)
            ])
        self.transforms4label = v2.Compose([
            v2.Resize(tf_cfg['img_size'], 
                      interpolation=v2.InterpolationMode.NEAREST, antialias=True),
            v2.RandomHorizontalFlip(p = 1 if hflip_bool else 0), 
            v2.RandomVerticalFlip(p=1 if vflip_bool else 0),
            ])
        
    def __call__(self, img:Image.Image, img_name:str):
        ''' Apply transforms to image '''
        
        if img_name == 'label': # transforms applied to label image
            img = self.label2tensor(img)/255
            img = self._affine_transforms(img, img_name)
            top, left, height, width = self.crop_params['anchor']
            img = v2.functional.crop(img, top, left, height, width)
            img = torch.rot90(img, self.n_rot90, dims=[-2,-1])
            img = self.transforms4label(img)
            
        elif img_name == 'anchor' or img_name == 'sample': 
            img = self.img2tensor(img)
            img = self._affine_transforms(img, img_name)
            top, left, height, width = self.crop_params[img_name]
            img = v2.functional.crop(img, top, left, height, width)
            img = torch.rot90(img, self.n_rot90, dims=[-2,-1])
            img = self.transforms4imgs(img)
            
        else:
            raise ValueError("Image name needs to be either 'anchor', "\
                             "'sample' or 'label' ")
            
        return img
    
    def _check_for_mistake_in_config(self, tf_cfg:dict) -> None:
        ''' Make sanity checks to prevent transforms that cannot coexist '''
        
        if tf_cfg['random_crop']:
            assert tf_cfg['max_translation'] == 0, 'Specifying translation amount '\
                                    'in addition to random cropping not supported'
            assert tf_cfg['rescale'] == 1, 'Specifying rescale amount in addition '\
                                                    'to random cropping not supported'
            assert tf_cfg['ROI_crops'] == False, 'ROI cropping and random cropping '\
                                                    'cannot be used simultaneously'
        elif tf_cfg['ROI_crops']:
            assert tf_cfg['max_translation'] == 0, 'Specifying translation amount '\
                                        'in addition to ROI cropping not supported'
            assert tf_cfg['rescale'] == 1, 'Specifying rescale amount in addition '\
                                                    'to ROI cropping not supported'
    
    def _get_shear_parameters(self, max_amount:int) -> dict:
        ''' Randomize shear amount for both images based on max value from config.
        
        Returns:
            shear_params (dict of str:tuple): shear parameters (tuple containing 
                                    x- and y-amount) for keys 'anchor' and 'sample'
        '''
        if max_amount == 0:
            shear_params= {'anchor': (0, 0), 'sample': (0, 0)} # default values
        elif max_amount > 0:
            # get shear values and apply to segmentation mask of anchor
            shearx_anchor = random.randint(-max_amount, max_amount)
            max_y_anchor = max_amount-shearx_anchor
            sheary_anchor = random.randint(-max_y_anchor, max_y_anchor)
            shear_params = {'anchor': (shearx_anchor, sheary_anchor)}

            # get shear values and apply to segmentation mask of sample
            shearx_sample = random.randint(-max_amount, max_amount)
            max_y_sample = max_amount-shearx_sample
            sheary_sample = random.randint(-max_y_sample, max_y_sample)
            shear_params['sample'] = (shearx_sample, sheary_sample)
        else:
            raise ValueError("Maximum shear amount must be a positive integer")
        
        return shear_params
    
    def _get_scale_parameters(self, max_scale_difference:float, 
                              seg1:np.ndarray, seg2:np.ndarray) -> dict:
        ''' Get parameter value to use for randomly rescaling anchor and sample.
        
        Arguments:
            max_scale_difference (float): the maximum amount of scale difference
                                        between anchor and sample images
            seg1 (numpy array): segmentation mask of anchor image
            seg2 (numpy array): segmentation mask of sample image
        Returns:
            rescale (dict of str:float): the rescale coefficient for both anchor
                                        and sample images
        '''
        if max_scale_difference == 1:
            rescale={'anchor':1, 'sample':1}
            
        else:        
            # randomly choose scale difference
            desired_scale_diff = random.uniform(1, max_scale_difference)
                        
            def get_max_upscale(segmask):
                ''' Find how much the image can be scaled up without cropping object. '''
                
                segmentation = np.where(segmask == 1)
                if len(segmentation) > 0 and len(segmentation[0]) > 0 and \
                                                len(segmentation[1]) > 0:
                    x_min = int(np.min(segmentation[1]))
                    x_max = int(np.max(segmentation[1]))
                    y_min = int(np.min(segmentation[0]))
                    y_max = int(np.max(segmentation[0]))
                else: 
                    raise ValueError("There is no segmentation present in the image")
                center_x = segmask.shape[1]//2
                center_y = segmask.shape[0]//2
                dist_from_center_x = max(abs(x_min-center_x), abs(x_max-center_x))
                dist_from_center_y = max(abs(y_min-center_y), abs(y_max-center_y))
                bottleneck_from_center = max(dist_from_center_x, dist_from_center_y)
                h = min(segmask.shape)
                max_upscale = (h//2)/bottleneck_from_center
                
                return max_upscale
            
            max_upscale_anchor = get_max_upscale(seg1)
            max_upscale_sample = get_max_upscale(seg2)
            
            rescale = {}            
            # if either image has enough margin to be upscaled without cropping object
            if (max_upscale_anchor >= desired_scale_diff) and (max_upscale_sample >= 
                                                               desired_scale_diff): 
                # randomly pick which to upscale
                rescale['anchor'] = random.choice([desired_scale_diff, 1])
                rescale['sample'] = desired_scale_diff if rescale['anchor'] == 1 else 1
            # if only the anchor has sufficient margin, rescale anchor
            elif (max_upscale_anchor >= desired_scale_diff) and (max_upscale_sample < 
                                                                 desired_scale_diff): 
                rescale['anchor'] = desired_scale_diff
                rescale['sample'] = 1
            # if only sample has sufficient margin, rescale sample
            elif (max_upscale_anchor < desired_scale_diff) and (max_upscale_sample >= 
                                                                desired_scale_diff): 
                rescale['anchor'] = 1
                rescale['sample'] = desired_scale_diff
            # if neither image has sufficient margin, scale one of them needs down
            elif (max_upscale_anchor < desired_scale_diff) and (max_upscale_sample < 
                                                                desired_scale_diff): 
                if random.random()>0.5:
                    rescale['anchor'] = max_upscale_anchor
                    rescale['sample'] = max_upscale_anchor/desired_scale_diff
                else:
                    rescale['anchor'] = max_upscale_sample/desired_scale_diff
                    rescale['sample'] = max_upscale_sample
            else:
                raise ValueError('There is an error in the implementation')
            
        return rescale
    
    def _apply_scale_shear_to_segmask(self, segmask:np.ndarray, img_name:str) \
                                                                -> np.ndarray:
        ''' Transforms segmentation mask prior to finding crop params. 
        
        Arguments:
            segmask (numpy array): foreground/background segmentation mask
            img_name (str): whether it is sample or anchor image'''
        
        segmask = torch.unsqueeze(torch.from_numpy(segmask), dim=0)
        rescale = self.rescale_params[img_name]
        shear_amt = self.shear_params[img_name]
        segmask = v2.functional.affine(segmask, angle=0, 
                        translate=[0,0], scale=rescale, shear=shear_amt, 
                        interpolation=v2.InterpolationMode.NEAREST)
        
        return segmask.squeeze().numpy()
        
    def _get_crop_parameters(self, tf_cfg:dict, 
                             seg1:np.ndarray, seg2:np.ndarray) -> dict:
        ''' Get the right cropping function according to config parameters 
        
        Arguments:
            tf_cfg (dict): config parameters for all image transforms
            seg1 (numpy array): segmentation mask of anchor image
            seg2 (numpy array): segmentation mask of sample image
        Returns:
            crop_params (dict of str:float): the top, left, height, width 
                        cropping parameters for both anchor and sample images
        '''
        # apply prior transformations so that segmentation masks reflect image
        seg1 = self._apply_scale_shear_to_segmask(seg1, 'anchor')
        seg2 = self._apply_scale_shear_to_segmask(seg2, 'sample')
        
        # get crop parameters
        min_HW = min(seg1.shape) # get smallest side of image
        crop_params = {}
        if tf_cfg['random_crop']:
            crop_params['anchor'] = self._random_crop_params(seg1)
            crop_params['sample'] = self._random_crop_params(seg2)
        elif tf_cfg['max_translation'] > 0:
            # convert max translation amount from ratio to pixels & choose randomly
            max_translation_pixels = round(min_HW * tf_cfg['max_translation'])
            trans_pixels = random.randint(0, max_translation_pixels)
            crop_params = self._get_crop_for_translation(seg1, seg2, trans_pixels)
        elif tf_cfg['ROI_crops']:
            crop_params = {
                'anchor': self._roi_crop_parameter(seg1, tf_cfg['center_roi'],
                                                   tf_cfg['roi_margin']),
                'sample': self._roi_crop_parameter(seg2, tf_cfg['center_roi'],
                                                   tf_cfg['roi_margin'])}
        else: # make a square crop
            top = seg1.shape[0] // 2 - min_HW // 2
            left = seg2.shape[1] // 2 - min_HW // 2
            crop_params['anchor'] = [top, left, min_HW, min_HW]
            crop_params['sample'] = [top, left, min_HW, min_HW]
        
        return crop_params
        
    def _affine_transforms(self, img:torch.Tensor, name:str) -> torch.Tensor:
        
        if name == 'label': # use NEAREST
            img = v2.functional.affine(img, angle=0, translate=[0,0], 
                scale=self.rescale_params["anchor"], 
                shear=self.shear_params["anchor"], 
                interpolation=v2.InterpolationMode.NEAREST)
        else: # use BILINEAR
            img = v2.functional.affine(img, angle=0, translate=[0,0], 
                scale=self.rescale_params[name], shear=self.shear_params[name], 
                interpolation=v2.InterpolationMode.BILINEAR)
        return img
          
    def _random_crop_params(self, segmask:np.ndarray) -> tuple[int,int,int,int]:
        ''' Find random crop parameters such that assembly object stays in view. 
        
        Returns:
            top, left, height, width
        '''        
        x_min, x_max, y_min, y_max = self._get_bbox_coordinates(segmask)
        x_delta = x_max-x_min
        y_delta = y_max-y_min
                    
        # choose random size between assembly object bbox and image height/width
        orig_height, orig_width = segmask.shape
        small_side_size = min(orig_width, orig_height)
        if max(x_delta, y_delta) < small_side_size:
            out_size = random.randint(max(x_delta, y_delta), small_side_size)
            # get actual corner pixels of crop
            top_min = y_max-out_size if y_max > out_size else 0
            top_max = orig_height-out_size if out_size>(orig_height-y_min) else y_min
            top = random.randint(top_min, top_max)
            
            left_min = x_max-out_size if x_max > out_size else 0
            left_max = x_min if (orig_width-x_min)>out_size else orig_width-out_size
            left = random.randint(left_min, left_max)

        # if some pixels are out of crop bounds, take the bounding box 
        # and turn it into a square crop with zero padding
        else:  
            out_size = max(x_delta, y_delta)
            top_min = y_max-out_size
            top_max = y_min
            top = random.randint(top_min, top_max)
            
            left_min = x_max-out_size 
            left_max = x_min
            left = random.randint(left_min, left_max)
        
        return top, left, out_size, out_size
    
    def _get_crop_for_translation(self, seg1: np.ndarray, 
                                  seg2: np.ndarray, translation: int) -> dict: 
        ''' Get crop parameters for desired translation between anchor and sample.
        
        Arguments:
            seg1 (numpy array): segmentation mask of anchor image
            seg2 (numpy array): segmentation mask of sample image
            translation (float): the maximum number of pixels translation there 
                                 should be between anchor & sample image
        Returns: 
            crop_params (dict): 
                keys: 'anchor' and 'sample', 
                values: cropping parameters - top, left, height, width
        '''
        centercrop_size = min(seg1.shape)
        centercrop_top = seg2.shape[0] // 2 - centercrop_size // 2
        centercrop_left = seg2.shape[1] // 2 - centercrop_size // 2
        
        def get_translation_margins(segmask:np.ndarray) -> tuple[int,int,int,int]:
            ''' Get margin values between object and image edges '''
            
            x_min, x_max, y_min, y_max = self._get_bbox_coordinates(segmask)
            
            margin_left = max(0, x_min - centercrop_left)
            margin_right = max(0, centercrop_left + centercrop_size - x_max)
            margin_top = max(0, y_min - centercrop_top)
            margin_bottom = max(0, centercrop_size - y_max)
            return margin_left, margin_right, margin_top, margin_bottom
        
        def margins_to_crop_params(seg_margins:list, sideways_direction:str, 
                                   up_down_direction:str, img_name:str) -> tuple:
            if img_name == 'sample':
                invert_directions = {'up':'down', 'down':'up', 
                                     'left':'right', 'right':'left'}
                sideways_direction = invert_directions[sideways_direction]
                up_down_direction = invert_directions[up_down_direction]
                                
            if sideways_direction == 'left':
                left = centercrop_left + seg_margins[0]
            elif sideways_direction == 'right':
                left = centercrop_left - seg_margins[1]
            if up_down_direction == 'down':
                top = centercrop_top + seg_margins[2]
            elif up_down_direction == 'up':
                top = centercrop_top - seg_margins[3]
                
            return top, left, centercrop_size, centercrop_size
        
        seg1_margins = get_translation_margins(seg1)
        seg2_margins = get_translation_margins(seg2)
        max_sideways = max(seg1_margins[0]+seg2_margins[1], 
                           seg1_margins[1]+seg2_margins[0])
        if max_sideways == seg1_margins[0]+seg2_margins[1]:
            sideways_direction = 'left'  # direction on seg1 (swapped for seg2)
        else: sideways_direction = 'right' 
        max_up_down = max(seg1_margins[2]+seg2_margins[3], 
                          seg1_margins[3]+seg2_margins[2])
        if max_up_down == seg1_margins[2]+seg2_margins[3]: 
            up_down_direction = 'up' # direction on seg1 (swapped for seg2)
        else: up_down_direction = 'down' 
        
        crop_params = {}
        # if we want more translation than we can have, crop into opposite corners
        if (translation+1)**2 > max_sideways**2+max_up_down**2: 
            crop_params['anchor'] = margins_to_crop_params(seg1_margins, 
                            sideways_direction, up_down_direction, 'anchor')
            crop_params['sample'] = margins_to_crop_params(seg2_margins, 
                            sideways_direction, up_down_direction, 'sample')
        
        else:  # randomly sample the amount of horizontal and vertical translation:
            min_sideways_translation = math.ceil(max(0, np.sqrt(translation**2-
                                                                max_up_down**2)))
            max_sideways_translation = min(translation, max_sideways)
            sideways_translation = random.randint(min_sideways_translation, 
                                                  max_sideways_translation)
            up_down_translation = math.floor(max(0, np.sqrt(translation**2-
                                                    sideways_translation**2)))
            
            # split sideways translation across both images:
            if sideways_direction == 'left':
                min_anchor_sideways = max(0, sideways_translation - seg2_margins[1])
                max_anchor_sideways = min(sideways_translation, seg1_margins[0])
                anchor_sideways = random.randint(min_anchor_sideways, 
                                                 max_anchor_sideways)
                anchor_left = centercrop_left+anchor_sideways
                sample_left = centercrop_left - (sideways_translation-anchor_sideways)
            elif sideways_direction == 'right':
                min_anchor_sideways = max(0, sideways_translation - seg2_margins[0])
                max_anchor_sideways = min(sideways_translation, seg1_margins[1])
                anchor_sideways = random.randint(min_anchor_sideways, 
                                                 max_anchor_sideways)
                anchor_left = centercrop_left-anchor_sideways
                sample_left = centercrop_left + (sideways_translation-anchor_sideways)
            if up_down_direction == 'up':
                min_anchor_up_down = max(0, up_down_translation - seg2_margins[3])
                max_anchor_up_down = min(up_down_translation, seg1_margins[2])
                anchor_up_down = random.randint(min_anchor_up_down, 
                                                max_anchor_up_down)
                anchor_top = centercrop_top+anchor_up_down
                sample_top = centercrop_top - (up_down_translation-anchor_up_down)
            elif up_down_direction == 'down':
                min_anchor_up_down = max(0, up_down_translation - seg2_margins[2])
                max_anchor_up_down = min(up_down_translation, seg1_margins[3])
                anchor_up_down = random.randint(min_anchor_up_down, 
                                                max_anchor_up_down)
                anchor_top = centercrop_top-anchor_up_down
                sample_top = centercrop_top + (up_down_translation-anchor_up_down)
            
            crop_params['anchor'] = [anchor_top, anchor_left, 
                                     centercrop_size, centercrop_size]
            crop_params['sample'] = [sample_top, sample_left, 
                                     centercrop_size, centercrop_size]
        
        return crop_params

    def _roi_crop_parameter(self, segmask: np.ndarray, center_roi: bool,
                            percent_margin: int = 10) -> list:
        ''' Get parameters for region of interest (ROI) cropping.
        
        Function that takes the bounding box of the object, and adds 10% margins 
        on to height and width and randomly moves object within those bounds to
        create some small random cropping when possible (i.e. without 0 padding).
        
        Arguments:
            segmask (numpy array): segmentation mask of anchor or sample images
            center_roi (bool): whether to randomly position object within crop 
                               (if False) or have it always be centered (if True)
            percent_margin (int): the size of margins around the assembly object,
                both above & below, and to the right & left, expressed as a 
                percentage of the height and width of the assembly object. Eg. 
                a percent_margin of 10% corresponds to 10% of the object height 
                being added above/below and 10% of the object width being added 
                to the right/left, with the object randomly translated therein.
        Returns: 
            crop_params (list): [top, left, height, width] '''
                
        def get_allowed_margins(bbox_coordinates, im_height, im_width):
            x_left = bbox_coordinates[0]
            x_right = im_width-bbox_coordinates[1]
            y_top = bbox_coordinates[2]
            y_bottom = im_height-bbox_coordinates[3]
            return x_left, x_right, y_top, y_bottom     
        
        bbox = self._get_bbox_coordinates(segmask) # left, right, top, bottom
        
        # get the number of pixels to append to object in x & y directions
        margin_frac1 = random.randint(10, percent_margin)/100
        margin_frac2 = random.randint(10, percent_margin)/100
        if bbox[1]-bbox[0] > bbox[3]-bbox[2]:  # if bbox is wide
            x_margin_frac = min(margin_frac1, margin_frac2)
            y_margin_frac = max(margin_frac1, margin_frac2)
        else:  # if bbox is tall
            x_margin_frac = max(margin_frac1, margin_frac2)
            y_margin_frac = min(margin_frac1, margin_frac2)
        
        x_margin = round(x_margin_frac*(bbox[1]-bbox[0]))
        y_margin = round(y_margin_frac*(bbox[3]-bbox[2]))
        
        # get allowable margins so that we do not crop outside the image boundaries
        H, W = segmask.shape[0:2]
        allowed_margins = get_allowed_margins(bbox, H, W)
        
        # get the horizontal crop parameters
        if x_margin >= allowed_margins[0] + allowed_margins[1]:  # too little margin
            left = 0; width = W  # do not crop
        else:  # there is enough room to create margins on one of the sides
            add_left = random.randint(max(0, x_margin-allowed_margins[1]),
                                      min(allowed_margins[0], x_margin)) 
            if center_roi:  # place bbox in center of crop for aligned image pairs
                add_left = x_margin//2 
            add_right = x_margin - add_left
            left = bbox[0] - add_left
            width = bbox[1] + add_right - left
        
        # get the vertical crop parameters
        if y_margin >= allowed_margins[2] + allowed_margins[3]:  # insufficient margin
            top = 0; height = H  # do not crop
        else:  # there is enough room to create margins on one of the sides
            add_top = random.randint(max(0, y_margin-allowed_margins[3]), 
                                     min(allowed_margins[2], y_margin)) 
            if center_roi:  # place bbox in center of crop for aligned image pairs
                add_top = y_margin//2 
            add_bottom = y_margin - add_top
            top = bbox[2] - add_top
            height = bbox[3] + add_bottom - top
            
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