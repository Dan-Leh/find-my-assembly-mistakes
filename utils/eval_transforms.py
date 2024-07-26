import random 
import torch
import math
import numpy as np
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode

class EvalTransforms():
    def __init__(self, test_type, old_size, img_size, scale_ratio, norm_type, 
                 translation_amt, segmask1, segmask2, desired_rescale):
        self.img_size = img_size
        self.test_type = test_type
        self.scale_ratio = scale_ratio
        
        # get normalization parameters
        if norm_type == 'imagenet':
            self.norm_mean = [0.485, 0.456, 0.406]
            self.norm_std = [0.229,0.224,0.225]
        else:
            raise NotImplementedError(f'No implementation for normalizing according to {norm_type}')
        
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
        self.transforms4label = v2.Resize(img_size, interpolation=InterpolationMode.NEAREST)
        
        if test_type == 'translation':
            self.crop_params = self.get_crop_parameters(segmask1, segmask2, translation_amt)
        elif test_type == 'scale':
            self.rescale = self.get_scale_param(segmask1, segmask2, 
                                                scale_ratio, desired_rescale) 
        elif test_type.startswith('ROI'):
            self.crop_params = {'reference': self.get_ROI_crop_parameter(segmask1),
                                'sample': self.get_ROI_crop_parameter(segmask2)}
        
    
    def __call__(self, img, img_name):
        
        if img_name == 'label':
            img = self.label2tensor(img)/255
            if self.test_type == 'translation' or self.test_type.startswith('ROI'):
                p = self.crop_params['reference']
                img = v2.functional.crop(img, p[0],p[1],p[2],p[3])
            elif self.test_type == 'scale':
                img = v2.functional.affine(img, angle=0, translate=[0,0], scale=self.rescale['reference'], 
                                            shear=[0,0], interpolation=InterpolationMode.NEAREST)
            if not self.test_type.startswith('ROI'): img = self.centercrop(img)
            img = self.transforms4label(img)
        
        else: # anchor or sample image
            img = self.img2tensor(img)
            if self.test_type=='orientation':
                if img_name == 'sample':
                    img = v2.functional.affine(img, angle=0, translate=[0,0], scale=self.scale_ratio, 
                                                shear=[0,0], interpolation=InterpolationMode.BILINEAR)
            elif self.test_type == 'translation' or self.test_type.startswith('ROI'):
                p = self.crop_params[img_name]
                img = v2.functional.crop(img, p[0],p[1],p[2],p[3])
            elif self.test_type == 'scale':
                img = v2.functional.affine(img, angle=0, translate=[0,0], scale=self.rescale[img_name], 
                                           shear=[0,0], interpolation=InterpolationMode.BILINEAR)
            
            if not self.test_type.startswith('ROI'): img = self.centercrop(img)
            img = self.transforms4imgs(img)
        
        return img

     
    def get_bbox_coordinates(self, segmask):
        segmentation = np.where(segmask == 1)
        if len(segmentation)>0 and len(segmentation[0])>0 and len(segmentation[1])>0:
            x_min = int(np.min(segmentation[1]))
            x_max = int(np.max(segmentation[1]))
            y_min = int(np.min(segmentation[0]))
            y_max = int(np.max(segmentation[0]))
            return x_min, x_max, y_min, y_max
        else: 
            raise ValueError("there is no segmentation present in the provided image")
        
        
    def get_crop_parameters(self, seg1, seg2, translation):
        ''' Returns cropping parameteters based on the desired amount of translation between the two images (measured in pixels)
        Args:
            seg1 (numpy array): segmentation mask of reference image
            seg2 (numpy array): segmentation mask of sample image
            translation: number of pixels to translate the images by
        Returns: 
            crop_params (dict): 
                keys: 'reference' and 'sample', 
                values: cropping parameters - top, left, height, width
        '''
        centercrop_size = min(seg1.shape)
        centercrop_top = seg2.shape[0] // 2 - centercrop_size // 2
        centercrop_left = seg2.shape[1] // 2 - centercrop_size // 2
        
        def get_translation_margins(segmask):
            x_min, x_max, y_min, y_max = self.get_bbox_coordinates(segmask)
            
            margin_left = max(0, x_min - centercrop_left)
            margin_right = max(0, centercrop_left + centercrop_size - x_max)
            margin_top = max(0, y_min - centercrop_top)
            margin_bottom = max(0, centercrop_size - y_max)
            return margin_left, margin_right, margin_top, margin_bottom
        
        def margins_to_crop_params(seg_margins, sideways_direction, up_down_direction, img_name):
            if img_name == 'sample':
                invert_directions = {'up':'down', 'down':'up', 'left':'right', 'right':'left'}
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
        
        # # convert seg2 to the same scale as seg1, only useful of the input images are not aligned (i.e. have orientation difference)
        # seg2 = torch.from_numpy(seg2)
        # seg2 = v2.functional.affine(seg2, angle=0, translate=[0,0], scale=self.scale_ratio, shear=[0,0], interpolation=InterpolationMode.NEAREST)
        # seg2 = seg2.numpy()
        
        # convert pixel translation amounts to scale of input image
        pixel_translation = round(translation * min(seg1.shape)/min(self.img_size))
        
        seg1_margins = get_translation_margins(seg1)
        seg2_margins = get_translation_margins(seg2)
        max_sideways = max(seg1_margins[0]+seg2_margins[1], seg1_margins[1]+seg2_margins[0])
        sideways_direction = 'left' if max_sideways == seg1_margins[0]+seg2_margins[1] else 'right' # direction on seg1 (swapped for seg2)
        max_up_down = max(seg1_margins[2]+seg2_margins[3], seg1_margins[3]+seg2_margins[2])
        up_down_direction = 'up' if max_up_down == seg1_margins[2]+seg2_margins[3] else 'down' # direction on seg1 (swapped for seg2)
        
        crop_params = {}
        
        if (pixel_translation+1)**2 > max_sideways**2+max_up_down**2: # if we want more translation than we can have
            # just put the two images in opposite corners
            crop_params['reference'] = margins_to_crop_params(seg1_margins, sideways_direction, up_down_direction, 'reference')
            crop_params['sample'] = margins_to_crop_params(seg2_margins, sideways_direction, up_down_direction, 'sample')
        
        else:  # randomly sample the amount of horizontal and vertical translation:
            min_sideways_translation = math.ceil(max(0, np.sqrt(pixel_translation**2-max_up_down**2)))
            max_sideways_translation = min(pixel_translation, max_sideways)
            sideways_translation = random.randint(min_sideways_translation, max_sideways_translation)
            up_down_translation = math.floor(max(0, np.sqrt(pixel_translation**2-sideways_translation**2)))
            
            # split sideways translation across both images:
            if sideways_direction == 'left':
                min_ref_sideways = max(0, sideways_translation - seg2_margins[1])
                max_ref_sideways = min(sideways_translation, seg1_margins[0])
                ref_sideways = random.randint(min_ref_sideways, max_ref_sideways)
                ref_left = centercrop_left+ref_sideways
                sample_left = centercrop_left - (sideways_translation-ref_sideways)
            elif sideways_direction == 'right':
                min_ref_sideways = max(0, sideways_translation - seg2_margins[0])
                max_ref_sideways = min(sideways_translation, seg1_margins[1])
                ref_sideways = random.randint(min_ref_sideways, max_ref_sideways)
                ref_left = centercrop_left-ref_sideways
                sample_left = centercrop_left + (sideways_translation-ref_sideways)
            if up_down_direction == 'up':
                min_ref_up_down = max(0, up_down_translation - seg2_margins[3])
                max_ref_up_down = min(up_down_translation, seg1_margins[2])
                ref_up_down = random.randint(min_ref_up_down, max_ref_up_down)
                ref_top = centercrop_top+ref_up_down
                sample_top = centercrop_top - (up_down_translation-ref_up_down)
            elif up_down_direction == 'down':
                min_ref_up_down = max(0, up_down_translation - seg2_margins[2])
                max_ref_up_down = min(up_down_translation, seg1_margins[3])
                ref_up_down = random.randint(min_ref_up_down, max_ref_up_down)
                ref_top = centercrop_top-ref_up_down
                sample_top = centercrop_top + (up_down_translation-ref_up_down)
            
            crop_params['reference'] = [ref_top, ref_left, centercrop_size, centercrop_size]
            crop_params['sample'] = [sample_top, sample_left, centercrop_size, centercrop_size]
        
        return crop_params
    
            
    def get_scale_param(self, seg1, seg2, scale_ratio, desired_rescale):
        ''' Arguments:
                - seg1 (numpy array): segmentation mask of reference image
                - seg2 (numpy array): segmentation mask of sample image
                - scale_ratio (float): the current scale ratio of between reference and sample (size(ref)/size(sample))
                - desired_rescale (float): what the scale difference should be
            Returns:
                rescale (dict{'reference':value,'sample':value}): the ratio by which to rescale both ref and sample
        '''  
        def get_max_upscale(segmask):
            segmentation = np.where(segmask == 1)
            if len(segmentation)>0 and len(segmentation[0])>0 and len(segmentation[1])>0:
                x_min = int(np.min(segmentation[1]))
                x_max = int(np.max(segmentation[1]))
                y_min = int(np.min(segmentation[0]))
                y_max = int(np.max(segmentation[0]))
            else: 
                raise ValueError("there is no segmentation present in the provided image")
            
            center_x = segmask.shape[1]//2
            center_y = segmask.shape[0]//2
            
            dist_from_center_x = max(abs(x_min-center_x), abs(x_max-center_x))
            dist_from_center_y = max(abs(y_min-center_y), abs(y_max-center_y))

            bottleneck_from_center = max(dist_from_center_x, dist_from_center_y)
            
            h = min(segmask.shape)
            max_upscale = (h//2)/bottleneck_from_center
            
            return max_upscale
        
        max_upscale_ref = get_max_upscale(seg1)
        max_upscale_sample = get_max_upscale(seg2)
        
        rescale = {}
        desired_rescale = random.choice([desired_rescale, 1/desired_rescale]) # 50% possibility of up or downsampling
        if (max_upscale_ref >= desired_rescale) and (max_upscale_sample >= desired_rescale): # randomly pick which to upscale
            rescale['reference'] = random.choice([desired_rescale, 1])
            rescale['sample'] = desired_rescale if rescale['reference'] == 1 else 1
        elif (max_upscale_ref >= desired_rescale) and (max_upscale_sample < desired_rescale): # rescale reference
            rescale['reference'] = desired_rescale
            rescale['sample'] = 1
        elif (max_upscale_ref < desired_rescale) and (max_upscale_sample >= desired_rescale): # rescale sample
            rescale['reference'] = 1
            rescale['sample'] = desired_rescale
        elif (max_upscale_ref < desired_rescale) and (max_upscale_sample < desired_rescale): # one of them needs to be scaled down
            if random.random()>0.5:
                rescale['reference'] = max_upscale_ref
                rescale['sample'] = max_upscale_ref/desired_rescale
            else:
                rescale['reference'] = max_upscale_sample/desired_rescale
                rescale['sample'] = max_upscale_sample
        else:
            raise ValueError('There is an error in the implementation')
            
        return rescale
    
    
    def get_ROI_crop_parameter(self, segmask: np.ndarray, center_roi=True):
        ''' Function that takes the bounding box of the object, and adds 10% margins
        Args:
            segmask (numpy array): segmentation mask of reference or sample images
        Returns: 
            crop_params (list): [top, left, height, width] '''
                
        def get_allowed_margins(bbox_coordinates, im_height, im_width):
            x_left = bbox_coordinates[0]
            x_right = im_width-bbox_coordinates[1]
            y_top = bbox_coordinates[2]
            y_bottom = im_height-bbox_coordinates[3]
            return x_left, x_right, y_top, y_bottom     
        
        bbox = self.get_bbox_coordinates(segmask) # bbox has shape left, right, top, bottom
        
        # desired margins: 10% of object height/width
        x_margin = round(0.1*(bbox[1]-bbox[0]))
        y_margin = round(0.1*(bbox[3]-bbox[2]))
        
        # get allowable margins so that we do not crop outside the image boundaries
        H, W = segmask.shape[0:2]
        allowed_margins = get_allowed_margins(bbox, H, W)
        
        # get the horizontal crop parameters
        if x_margin >= allowed_margins[0] + allowed_margins[1]: # if there is insufficient margin
            left = 0; width = W # do not crop
        else:  # there is enough room to create margins on one of the sides
            add_left = random.randint(max(0, x_margin-allowed_margins[1]), min(allowed_margins[0], x_margin)) 
            if center_roi: add_left = x_margin//2 # place bbox in center of crop for real images -> makes ROI crop deterministic
            add_right = x_margin - add_left
            left = bbox[0] - add_left
            width = bbox[1] + add_right - left
        
        # get the vertical crop parameters
        if y_margin >= allowed_margins[2] + allowed_margins[3]: # if there is insufficient margin
            top = 0; height = H # do not crop
        else:  # there is enough room to create margins on one of the sides
            add_top = random.randint(max(0, y_margin-allowed_margins[3]), min(allowed_margins[2], y_margin)) 
            if center_roi: add_top = y_margin//2 # place bbox in center of crop for real images -> makes ROI crop deterministic
            add_bottom = y_margin - add_top
            top = bbox[2] - add_top
            height = bbox[3] + add_bottom - top
            
        crop_params = [top, left, height, width]
        return crop_params
        
        