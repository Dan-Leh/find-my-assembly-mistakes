''' Taken from https://github.com/YanchaoYang/FDA/blob/master/utils/__init__.py'''

import os
import json
import torch
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
import random
from PIL import Image
import numpy as np

class FourierDomainAdaptation():
    def __init__(self, fda_config, img_size):
        self.img_size = img_size
        
        self.beta_min = fda_config['beta_min']
        self.beta_max = fda_config['beta_max']
        self.frac_imgs_w_fda = fda_config['frac_imgs_w_fda']
        
        target_img_root = "/shared/nl011006/res_ds_ml_restricted/TimSchoonbeek"+\
                                                                "/industreal_cont"

        # make a list of all target images and their bounding boxes
        self.trg_img_paths = []
        self.bboxes = []
        for folder_name in ['train_assy', 'train_main']:
            
            path_to_folder = os.path.join(target_img_root, folder_name)
            path_to_imgs = os.path.join(path_to_folder, "images")
            path_to_labels = os.path.join(path_to_folder, 'labels.json')
            
            # open label data about all images in folder
            with open(path_to_labels, 'r') as fp:
                img_info = json.load(fp) 
                # contains keys: ['images','labels','bbox','recs','visibility']
            fp.close()
            
            # filter out images without bounding box & too much occlusion
            for i in range(len(img_info['visibility'])):
                if img_info['visibility'][i]==3 \
                                and img_info['bbox'][i]!=None:
                    path2img = os.path.join(path_to_imgs,img_info['images'][i])
                    self.trg_img_paths.append(path2img)
                    self.bboxes.append(img_info['bbox'][i])
                        
            self.datalength = len(self.trg_img_paths)
            
    def _get_target_image(self) -> torch.Tensor:
        ''' Load a real-world image and return its roi crop. '''
        
        # choose and load a random image
        rnd_idx = random.randrange(0, self.datalength)
        path_to_img = self.trg_img_paths[rnd_idx]
        img = Image.open(path_to_img).convert("RGB")
        
        # make roi crop
        bbox = self.bboxes[rnd_idx]
        img = img.crop((bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]))
                
        self.img2tensor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(self.img_size, interpolation=InterpolationMode.BILINEAR)
            ])
        
        img = self.img2tensor(img)
        
        return img

    def __call__(self, src_img:torch.Tensor):
        ''' Transform source image to target domain. '''
        
        # only use fda on certain percentage of images
        if random.random() < self.frac_imgs_w_fda:  
            # randomly select how much target domain info to use
            beta = random.uniform(self.beta_min, self.beta_max)
            trg_img = self._get_target_image()
            src_in_trg = FDA_source_to_target(src_img, trg_img, L = beta)
            return src_in_trg
        
        else:  # no fda transformation
            return src_img
    
    

def extract_ampl_phase(fft_im):
    fft_amp = torch.abs(fft_im)
    fft_pha = torch.angle(fft_im)
    return fft_amp, fft_pha

def low_freq_mutate( amp_src, amp_trg, L=0.1 ): # 3 x h x w
    _, h, w = amp_src.size()
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)     # get b
    amp_src[:,0:b,0:b]     = amp_trg[:,0:b,0:b]      # top left
    amp_src[:,0:b,w-b:w]   = amp_trg[:,0:b,w-b:w]    # top right
    amp_src[:,h-b:h,0:b]   = amp_trg[:,h-b:h,0:b]    # bottom left
    amp_src[:,h-b:h,w-b:w] = amp_trg[:,h-b:h,w-b:w]  # bottom right
    return amp_src

def FDA_source_to_target(src_img, trg_img, L=0.1):
    ''' Exchange magnitude of low frequencies from trg to src'''
    # input: src_img, trg_img

    # get fft of both source and target
    fft_src = torch.fft.fft2( src_img ) 
    fft_trg = torch.fft.fft2( trg_img )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = extract_ampl_phase( fft_src)
    amp_trg, pha_trg = extract_ampl_phase( fft_trg)

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate( amp_src.clone(), amp_trg, L=L )

    # recompose fft of source
    # fft_src_ = torch.zeros( fft_src.size(), dtype=torch.float )
    fft_src_ = (torch.cos(pha_src) + 1j*torch.sin(pha_src)) * amp_src_

    # get the recomposed image: source content, target style
    src_in_trg = torch.fft.ifft2( fft_src_ )

    # get rid of complex numbers caused by numerican inaccuracies
    src_in_trg = torch.abs(src_in_trg).clamp_(0,1)
    
    return src_in_trg


# def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):
#     a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
#     a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

#     _, h, w = a_src.shape
#     b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
#     c_h = np.floor(h/2.0).astype(int)
#     c_w = np.floor(w/2.0).astype(int)

#     h1 = c_h-b
#     h2 = c_h+b+1
#     w1 = c_w-b
#     w2 = c_w+b+1

#     a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
#     a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
#     return a_src

# def FDA_source_to_target_np( src_img, trg_img, L=0.1 ):
#     # exchange magnitude
#     # input: src_img, trg_img

#     src_img_np = src_img.numpy()
#     trg_img_np = trg_img.numpy()

#     # get fft of both source and target
#     fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
#     fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

#     # extract amplitude and phase of both ffts
#     amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
#     amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

#     # mutate the amplitude part of source with target
#     amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L )

#     # mutated fft of source
#     fft_src_ = amp_src_ * np.exp( 1j * pha_src )

#     # get the mutated image
#     src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
#     src_in_trg = np.real(src_in_trg)

#     return src_in_trg