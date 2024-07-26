import random 
import torch
import math
import numpy as np
from torchvision.transforms import v2


def denormalize(tensor, normalization_type):
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


class Transforms(torch.nn.Module):
    def __init__(self, segmask_ref, segmask_sample, aug_cfg, scale_ratio, real_sample=False, 
                 wheel_jitter_info=(None,None,None,None)):
        super().__init__()
        self.real_sample = real_sample
        self.wheel_jitter_info = wheel_jitter_info
        
        # whether to change appearance of wheels
        self.wheel_jitter = aug_cfg['wheel_jitter']
        
        # how much shear to use
        self.shear= {'reference': (0, 0), 'sample': (0, 0)} # default values
        if aug_cfg['shear'] > 0:
            shearx_ref = random.randint(0, aug_cfg['shear'])
            sheary_ref = random.randint(0, aug_cfg['shear']-shearx_ref)
            shearx_sam = random.randint(0, aug_cfg['shear'])
            sheary_sam = random.randint(0, aug_cfg['shear']-shearx_sam)
            segmask_ref = torch.unsqueeze(torch.from_numpy(segmask_ref), dim=0)
            segmask_sample = torch.unsqueeze(torch.from_numpy(segmask_sample), dim=0)
            segmask_ref = v2.functional.affine(segmask_ref, angle=0, translate=[0,0], scale=1, shear=[shearx_ref,sheary_ref], 
                                           interpolation=v2.InterpolationMode.NEAREST).squeeze().numpy()
            segmask_sample = v2.functional.affine(segmask_sample, angle=0, translate=[0,0], scale=1, shear=[shearx_sam,sheary_sam], 
                                           interpolation=v2.InterpolationMode.NEAREST).squeeze().numpy()
            self.shear['reference'] = (shearx_ref, sheary_ref)
            self.shear['sample'] = (shearx_sam, sheary_sam)
            
        
        # sanity checks, some augmentation are not meant to be used in parallel
        if aug_cfg['random_crop']:
            assert aug_cfg['max_translation']==0 and aug_cfg['rescale'] == 1 and aug_cfg['ROI_crops'] == False
        elif aug_cfg['ROI_crops']:
            assert aug_cfg['max_translation']==0 and aug_cfg['rescale'] == 1
        
        # extract crop size
        size = aug_cfg['img_size']
        # get rotation
        n_rot90_options = [0,1,2,3] 
        self.n_rot90 = random.choice(n_rot90_options) if aug_cfg['rotation'] else 0 # get the same rotation for all images

        # get scale parameters
        def rescale_segmask(segmask, rescale_amount):
            segmask = torch.unsqueeze(torch.from_numpy(segmask), dim=0)
            segmask = v2.functional.affine(segmask, angle=0, translate=[0,0], scale=rescale_amount, shear=[0,0], interpolation=v2.InterpolationMode.NEAREST)
            return segmask.squeeze().numpy()
        
        # first make both images same scale, then randomize scale difference according to config
        if type(scale_ratio) == tuple: # rescale both images to the values given
            segmask_ref = rescale_segmask(segmask_ref, scale_ratio[0])
            segmask_sample = rescale_segmask(segmask_sample, scale_ratio[1])
        elif scale_ratio != 1 and not (aug_cfg['ROI_crops'] or aug_cfg['random_crop']): 
            segmask_sample = rescale_segmask(segmask_sample, scale_ratio) # convert seg2 to the same scale as seg1, only useful of the input images are not aligned (i.e. have orientation difference)

        if aug_cfg['rescale'] != 1 and not (aug_cfg['ROI_crops'] or aug_cfg['random_crop']):
            desired_rescale = random.uniform(1, aug_cfg['rescale'])
            if desired_rescale != 1:
                self.rescale = self.get_scale_param(segmask_ref, segmask_sample, desired_rescale, scale_ratio)
                segmask_ref = rescale_segmask(segmask_ref, self.rescale["reference"]) # rescale segmasks for cropping algorithm
                segmask_sample = rescale_segmask(segmask_sample, self.rescale["sample"]) # rescale segmasks for cropping algorithm
        elif type(scale_ratio) != tuple and not (aug_cfg['ROI_crops'] or aug_cfg['random_crop']):
            self.rescale = {"reference":1,  "sample":scale_ratio} # unity scale difference
        elif type(scale_ratio) == tuple:
            self.rescale = {'reference': scale_ratio[0], 'sample': scale_ratio[1]} # rescale to same defined value
        elif aug_cfg['ROI_crops'] or aug_cfg['random_crop']: # no rescaling
            self.rescale = {'reference': 1, 'sample': 1}

            
        # get crop parameters
        min_HW = min(segmask_ref.shape) # take the smallest between height & width of image (used to make a square crop)
        self.crop_params = {}
        if aug_cfg['random_crop']:
            self.crop_params['reference'] = self.random_allowable_crop_params(segmask_ref)
            self.crop_params['sample'] = self.random_allowable_crop_params(segmask_sample)
        elif aug_cfg['max_translation'] > 0:
            max_translation_pixels = round(min_HW * aug_cfg['max_translation'])
            translation_pixels = random.randint(0, max_translation_pixels)
            self.crop_params = self.get_crop_parameters(segmask_ref, segmask_sample, translation_pixels)
        elif aug_cfg['ROI_crops']:
            if aug_cfg['center_roi'] or real_sample: center_roi = True
            self.crop_params = {'reference': self.get_ROI_crop_parameter(segmask_ref, center_roi),
                                'sample': self.get_ROI_crop_parameter(segmask_sample, center_roi)}
        else:
            top = segmask_ref.shape[0] // 2 - min_HW // 2
            left = segmask_ref.shape[1] // 2 - min_HW // 2
            self.crop_params['reference'] = [top, left, min_HW, min_HW]
            self.crop_params['sample'] = [top, left, min_HW, min_HW]
            # if real_sample: self.crop_params['sample'] 

        # get random flip parameters
        hflip_bool = True if random.random()<aug_cfg['hflip_probability'] else False
        vflip_bool = True if random.random()<aug_cfg['vflip_probability'] else False
        
        # get normalization parameters
        if aug_cfg['normalization'] == 'imagenet':
            self.norm_mean = [0.485, 0.456, 0.406]
            self.norm_std = [0.229,0.224,0.225]
        else:
            raise NotImplementedError(f'No implementation for normalizing according to {aug_cfg['normalization']}')
        
        # define transforms for converting images to tensor
        self.img2tensor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
            ])
        self.label2tensor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=False)
            ])
        
        # define transform sequence using obtained parameters
        self.transforms4imgs = v2.Compose([
            v2.Resize(size, interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
            v2.RandomHorizontalFlip(p = 1 if hflip_bool else 0), # ensures that all 3 images get the same flip 
            v2.RandomVerticalFlip(p=1 if vflip_bool else 0),
            v2.ColorJitter(brightness=aug_cfg['brightness'], 
                            contrast=aug_cfg['contrast'], 
                            saturation=aug_cfg['saturation'], 
                            hue=aug_cfg['hue']),
            v2.GaussianBlur(kernel_size=aug_cfg['g_kernel_size'], 
                            sigma=(aug_cfg['g_sigma_l'], aug_cfg['g_sigma_h'])),
            v2.Normalize(mean=self.norm_mean, std=self.norm_std)
            ])
        self.transforms4segmasks = v2.Compose([
            v2.Resize(size, interpolation=v2.InterpolationMode.NEAREST, antialias=True),
            ])
        self.transforms4label = v2.Compose([
            v2.Resize(size, interpolation=v2.InterpolationMode.NEAREST, antialias=True),
            v2.RandomHorizontalFlip(p = 1 if hflip_bool else 0), # ensures that all 3 images get the same flip 
            v2.RandomVerticalFlip(p=1 if vflip_bool else 0),
            ])

    def affine(self, img, img_name):
        
        if img_name == 'label': # use NEAREST
            img = v2.functional.affine(img, angle=0, translate=[0,0], scale=self.rescale["reference"], 
                                           shear=[0,0], interpolation=v2.InterpolationMode.NEAREST)
        else: # use BILINEAR
            img = v2.functional.affine(img, angle=0, translate=[0,0], scale=self.rescale[img_name], 
                                           shear=self.shear[img_name], interpolation=v2.InterpolationMode.BILINEAR)
        return img
    
    def wheel_color_jitter(self, img, img_name, real_sample):
        wheels = ['wheel_1', 'wheel_2', 'wheel_3', 'wheel_4']
        if img_name == 'reference':
            segmask, instances = self.wheel_jitter_info[:2]
        elif img_name == 'sample':
            segmask, instances = self.wheel_jitter_info[2:]
        
        mask = np.zeros(segmask.shape[:-1],dtype=bool)
        for instance in instances:
            if instance['labelName'] in wheels:
                mask = mask | np.all(segmask==instance['color'][:3], axis=-1)
        
        img_copy = img
        if real_sample: # determinstically apply the same color jitter on wheels to all images
            img_copy = v2.functional.adjust_brightness(img_copy, 0.8)
            img_copy = v2.functional.adjust_contrast(img_copy, 1.4)
        else:
            img_copy = v2.ColorJitter((0.2,1.2), (0.8,2), 0, 0)(img_copy)
        mask = torch.BoolTensor(mask).unsqueeze(0).expand_as(img)
        img[mask] = (img_copy[mask])
        
        return img        
        
    def __call__(self, img, img_name):
        
        if img_name == 'label': # transforms applied to label image
            img = self.label2tensor(img)/255
            img = self.affine(img, img_name)
            top, left, height, width = self.crop_params['reference']
            img = v2.functional.crop(img, top, left, height, width)
            img = torch.rot90(img, self.n_rot90, dims=[-2,-1])
            img = self.transforms4label(img)
            
        elif img_name == 'reference' or img_name == 'sample': # transforms applied to both images
            img = self.img2tensor(img)
            if img_name =='sample' and self.real_sample: 
                img = v2.Resize((320,320), interpolation=v2.InterpolationMode.BILINEAR, antialias=True)(img) #only used for loading real images
            if not self.real_sample or img_name == 'reference': # only apply this to synthetic images
                if self.wheel_jitter: img = self.wheel_color_jitter(img, img_name, self.real_sample)
                img = self.affine(img, img_name)
                top, left, height, width = self.crop_params[img_name]
                img = v2.functional.crop(img, top, left, height, width)
            img = torch.rot90(img, self.n_rot90, dims=[-2,-1])
            img = self.transforms4imgs(img)
        
        elif img_name == 'segmask': # transform for colored segmentation mask, using NEAREST interpolation
            img = self.img2tensor(img)
            top, left, height, width = self.crop_params['reference']
            img = v2.functional.crop(img, top, left, height, width)
            img = self.transforms4segmasks(img)
            
        else:
            raise ValueError("Image name needs to be either 'reference', 'sample' or 'label' ")
            
        return img
    
    def get_ROI_crop_parameter(self, segmask: np.ndarray, center_roi: bool):
        ''' Function that takes the bounding box of the object, and adds 10% margins on both sides to create some
            small semblance of random cropping when possible (i.e. without ever relying on 0 padding)
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
        

    def get_crop_parameters(self, seg1: np.ndarray, seg2: np.ndarray, translation: int) -> dict: 
        ''' Returns cropping parameteters based on the desired amount of translation between the two images (measured in pixels)
        Args:
            seg1 (numpy array): segmentation mask of reference image
            seg2 (numpy array): segmentation mask of sample image
            translation (float): the maximum number of pixels translation there should be between both images
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
        
        seg1_margins = get_translation_margins(seg1)
        seg2_margins = get_translation_margins(seg2)
        max_sideways = max(seg1_margins[0]+seg2_margins[1], seg1_margins[1]+seg2_margins[0])
        sideways_direction = 'left' if max_sideways == seg1_margins[0]+seg2_margins[1] else 'right' # direction on seg1 (swapped for seg2)
        max_up_down = max(seg1_margins[2]+seg2_margins[3], seg1_margins[3]+seg2_margins[2])
        up_down_direction = 'up' if max_up_down == seg1_margins[2]+seg2_margins[3] else 'down' # direction on seg1 (swapped for seg2)
        
        crop_params = {}
        
        if (translation+1)**2 > max_sideways**2+max_up_down**2: # if we want more translation than we can have
            # just put the two images in opposite corners
            crop_params['reference'] = margins_to_crop_params(seg1_margins, sideways_direction, up_down_direction, 'reference')
            crop_params['sample'] = margins_to_crop_params(seg2_margins, sideways_direction, up_down_direction, 'sample')
        
        else:  # randomly sample the amount of horizontal and vertical translation:
            min_sideways_translation = math.ceil(max(0, np.sqrt(translation**2-max_up_down**2)))
            max_sideways_translation = min(translation, max_sideways)
            sideways_translation = random.randint(min_sideways_translation, max_sideways_translation)
            up_down_translation = math.floor(max(0, np.sqrt(translation**2-sideways_translation**2)))
            
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
    
    def get_scale_param(self, seg1, seg2, desired_scale_diff, scale_ratio):
        ''' Arguments:
                - seg1 (numpy array): segmentation mask of reference image
                - seg2 (numpy array): segmentation mask of sample image
                - desired_scale_diff (float): the desired scale difference between reference & sample
                - scale_ratio (float or tuple): the current scale ratio of between reference and sample (size(ref)/size(sample)),
                                                if tuple, it is the value by which to rescale both ref and sam before changing their scale relative to each other
                
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
        # desired_scale_diff = random.choice([desired_scale_diff, 1/desired_scale_diff]) # 50% possibility of up or downsampling
        if (max_upscale_ref >= desired_scale_diff) and (max_upscale_sample >= desired_scale_diff): # randomly pick which to upscale
            rescale['reference'] = random.choice([desired_scale_diff, 1])
            rescale['sample'] = desired_scale_diff if rescale['reference'] == 1 else 1
        elif (max_upscale_ref >= desired_scale_diff) and (max_upscale_sample < desired_scale_diff): # rescale reference
            rescale['reference'] = desired_scale_diff
            rescale['sample'] = 1
        elif (max_upscale_ref < desired_scale_diff) and (max_upscale_sample >= desired_scale_diff): # rescale sample
            rescale['reference'] = 1
            rescale['sample'] = desired_scale_diff
        elif (max_upscale_ref < desired_scale_diff) and (max_upscale_sample < desired_scale_diff): # one of them needs to be scaled down
            if random.random()>0.5:
                rescale['reference'] = max_upscale_ref
                rescale['sample'] = max_upscale_ref/desired_scale_diff
            else:
                rescale['reference'] = max_upscale_sample/desired_scale_diff
                rescale['sample'] = max_upscale_sample
        else:
            raise ValueError('There is an error in the implementation')
        
        if type(scale_ratio) != tuple:
            rescale['sample'] *= scale_ratio
        elif type(scale_ratio) == tuple:
            rescale['reference'] *= scale_ratio[0]
            rescale['sample'] *= scale_ratio[1]
        return rescale
        
    def random_allowable_crop_params(self, segmask):      
        # extract the threshold values of the bounding box surrounding the car
        segmentation = np.where(segmask == 1)
        if len(segmentation)>0 and len(segmentation[0])>0 and len(segmentation[1])>0:
            x_min = int(np.min(segmentation[1]))
            x_max = int(np.max(segmentation[1]))
            y_min = int(np.min(segmentation[0]))
            y_max = int(np.max(segmentation[0]))
            x_delta = x_max-x_min
            y_delta = y_max-y_min
        else: 
            raise ValueError("there is no segmentation present in the provided image")
                    
        # choose random size between the image height & the distance between top & bottom of car
        orig_height, orig_width = segmask.shape
        small_side_size = min(orig_width, orig_height)
        if max(x_delta, y_delta) < small_side_size:
            output_size = random.randint(max(x_delta, y_delta), small_side_size)
            # get actual corner pixels of crop
            top_min = y_max-output_size if y_max > output_size else 0
            top_max = orig_height-output_size if output_size > (orig_height-y_min) else y_min
            top = random.randint(top_min, top_max)
            
            left_min = x_max-output_size if x_max > output_size else 0
            left_max = x_min if (orig_width-x_min) > output_size else orig_width-output_size
            left = random.randint(left_min, left_max)
            self.too_big = False

        else: # if some pixels are out of crop bounds, take a center crop
            self.too_big = True
            print(f'Segmask too big - x_size: {x_delta}, y_size: {y_delta}')
            output_size = max(x_delta, y_delta)
            top_min = y_max-output_size
            top_max = y_min
            top = random.randint(top_min, top_max)
            
            left_min = x_max-output_size 
            left_max = x_min
            left = random.randint(left_min, left_max)
        
        return top, left, output_size, output_size