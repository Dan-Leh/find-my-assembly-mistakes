import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

from utils.eval_transforms import EvalTransforms
from utils.background_randomizer import replace_background

class EvalDataset(Dataset):
    ''' Provides synthetic image pairs & ground truth for testing. '''
    
    def __init__(
        self,
        data_list_filepath: str = "", 
        norm_type: str = 'imagenet',
        img_size: tuple = (256,256), 
        random_background: str = "",
        ignore_0_change: bool = True,
        bg_img_root: str = "./data/COCO_Images"
        ):
        '''
        Arguments:
            data_list_filepath (str): path to json file containing IDs to images
                to make deterministic pairs 
            norm_type (str): which normalization to apply to input images
            img_size (tuple of ints): height and width of images that are outputted
            random_background (str): whether to use random background on test set (only 
                supported on "Main_test_set")
            ignore_0_change (bool): skip testing on image pairs that have no 
                meaningful change as those have 0 IoU (saves time & compute)
            bg_img_root (path): path to where the random background images are 
                                        stored.
        '''
        self.norm_type = norm_type
        self.img_size = img_size
        self.random_background = random_background
        self.bg_img_root = bg_img_root
        
        self.path_to_data, filename = os.path.split(data_list_filepath)
        self.determinstic_set = self._load_json(filename)  # paths to image pairs
        self.make_roi_crops = True if data_list_filepath.endswith('w_crops.json') \
                         else False
        if ignore_0_change:
            if type(self.determinstic_set[0]["n_differences"])==int:
                no_change = 0
            else:
                no_change = [0,0]
            for i in range(len(self.determinstic_set)-1,-1,-1):  # loop backward
                if self.determinstic_set[i]["n_differences"] == no_change:
                    self.determinstic_set.pop(i)
                
        # decode where to load data from
        self.id2indexdict = self._id2index() 
        # list containing the state of each image in set
        self.state_list = self._load_json('state_list.json') 
        # get the number of frames per sequence (used for indexing in state_list)
        seq_dir = os.path.join(self.path_to_data,"sequence0000")
        self.n_frames = len(os.listdir(seq_dir))//3  # 3 files per frame
            
    def _load_json(self, name: str) -> list:
        ''' load json file with specified name from data directory '''
        
        path = os.path.join(self.path_to_data, name)
        with open(path, 'r') as f:
            data = json.load(f)
        f.close()
        return data
        
    def __len__(self):
        return len(self.determinstic_set)
    
    def _id2index(self) -> dict: 
        ''' Make dictionary that converts labelIDs to part index.
        
        Each state is described by a binary sequence, where each bit indicates 
        whether a part of the object is in that state or not. The index of each part 
        of the assembly object in that binary sequence is given by the PartList.json 
        file. This function makes a dictionary that contains the part index 
        corresponding to each part ID contained in the instance segmentation
        annotation of each image in the dataset. It is needed for generating ground
        truth masks by comparing what parts are present in anchor and sample images.
        '''
        id2index = {}
        annotationDefinitions = self._load_json("annotation_definitions.json")
        for annotationDefinition in annotationDefinitions["annotationDefinitions"]:
            if annotationDefinition["id"] == "instance segmentation":
                id_data = annotationDefinition["spec"]
        
        idx_data = self._load_json("PartList.json")["index_parts"]
        for id_item in id_data:
            for idx_item in idx_data:
                if idx_item["part"] == id_item["label_name"]:
                    id2index[id_item["label_id"]] = idx_item["index"]
            
        return id2index

    def _load_image(self, sequence: int, frame: int) -> Image.Image:
        ''' Load an image from the dataset given a sequence and frame index. '''
        
        file_path = os.path.join(self.path_to_data, 
                                 f"sequence{str(sequence).zfill(4)}", 
                                 f"step{str(frame).zfill(4)}.camera.png")
        image = Image.open(file_path).convert("RGB")
        
        return image
    
    def _load_segmentation_masks(self, seq_a: int, seq_b: int, frame_1: int, 
                                frame_2:int) -> tuple[np.ndarray,np.ndarray]:
        ''' Load two binary foreground/background segmentation masks.
        
        Load one segmentation mask per sequence/pose. The purpose of the mask is to 
        make smart transformation functions that eg.translate images without 
        cropping out foreground, or to make eg. ROI crops. Each segmentation mask 
        consists of the union of the segmentation masks for the two frames/states 
        so that areas containing missing parts are not cropped out.
        '''
        segmentations = [None,None]
        for j, sequence in enumerate([seq_a, seq_b]):
            for i, frame in enumerate([frame_1, frame_2]):
                # open image & convert to numpy array
                seq_name = f"sequence{str(sequence).zfill(4)}"
                img_name = fr"step{str(frame).zfill(4)}.camera.instance segmentation.png" 
                label_filepath = os.path.join(self.path_to_data, seq_name, img_name) 
                                              
                image = Image.open(label_filepath).convert('L')  # grayscale
                image = np.array(image)
                # create empty mask on first iteration & accumulate masks
                if i==0: semseg = np.zeros_like(image, dtype=np.uint8) 
                semseg += image

            segmentations[j] = (semseg > 0).astype(np.uint8) # make masks binary
            
        return segmentations[0], segmentations[1]
    
    def _get_instance_info(self, sequence:int, frame:int) -> list:
        ''' Load instance segmentation label definitions from frame_data file. '''
        
        seq_name = f"sequence{str(sequence).zfill(4)}"
        file_name = f"step{str(frame).zfill(4)}.frame_data.json"
        full_path = os.path.join(seq_name, file_name)
        json_data = self._load_json(full_path)
        for annotation in json_data["captures"][0]["annotations"]:
            if annotation["id"] == "instance segmentation":
                instances = annotation["instances"]; break
        
        return instances

    def _load_binary_change_mask(self, sequence: int, frame_1: int, frame_2: int, 
                                 state_1: int, state_2: int) -> Image.Image:
        ''' Load the binary change mask of an image pair from the dataset.
        
        Arguments:
            sequence (int): the sequence index of the anchor image
            frame_1 (int): the index of the frame of the anchor image
            frame_2 (int): the index of the frame of the sample image
        Returns:
            label (PIL Image): the label of the binary change mask 
        '''
        # find the annotation part ID of each part that has incurred change per image
        difference_in_parts = state_1 ^ state_2
        diff_parts_in_state1 = difference_in_parts & state_1
        diff_parts_in_state2 = difference_in_parts & state_2
        labelChangeIds = {}
        labelChangeIds[0] = [key for key in self.id2indexdict.keys() 
                             if diff_parts_in_state1 & (1 << self.id2indexdict[key])]
        labelChangeIds[1] = [key for key in self.id2indexdict.keys() 
                             if diff_parts_in_state2 & (1 << self.id2indexdict[key])]       
        
        for i, frame in enumerate([frame_1, frame_2]):
            seq_name = f"sequence{str(sequence).zfill(4)}"
            img_name = fr"step{str(frame).zfill(4)}.camera.instance segmentation.png"
            label_filepath = os.path.join(self.path_to_data, seq_name, img_name) 
            image = Image.open(label_filepath)
            image = np.array(image)
            
            rgb_list = []  # get the rgb values of all parts that are 'change'
            instances = self._get_instance_info(sequence, frame)
            for instancelabel in instances:
                if instancelabel["labelId"] in labelChangeIds[i]:
                    rgb_list.append(instancelabel["color"])     
            
            if i==0:  # initialize empty mask
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            for rgb in rgb_list: # add change pixels to mask
                mask += np.all(np.equal(image, rgb), axis=-1)     
        
        change_mask = 255*(mask > 0).astype(np.uint8)  # make binary
        change_mask = Image.fromarray(change_mask, mode='L')  # grayscale
        
        return change_mask
    
    def _randomize_image_background(self, anchor:torch.Tensor, 
                        sample:torch.Tensor, img_pair_info:dict, sequences:list[int,int], 
                        frame_ids:list[int,int], transforms:EvalTransforms
                        )-> list[torch.Tensor, torch.Tensor]:
        ''' Add natural image background from COCO to assembly object images
    
        Arguments:
            anchor (tensor): transformed anchor image
            sample (tensor): transformed sample image
            img_pair_info (dict): contains path to background image linked
                to each paired sample and anchor image.
            sequences (list): sequence number of anchor and sample images
            frame_ids (list): background images to use
            transforms (Transform class): the transform applied to image pairs
                prior to becoming eg. ROI crops, used to transform the 
                segmentation masks so they are aligned with the images to cut 
                out the assembly object
                            
        Returns:
            anchor (tensor): anchor image with randomized background
            sample (tensor): sample image with randomized background
        '''
        
        def get_segmentation_mask(sequence, frame):
            ''' Return binary segmentation of assemnbly object '''
            
            label_filepath = os.path.join(self.path_to_data, 
                    f"sequence{str(sequence).zfill(4)}", 
                    fr"step{str(frame).zfill(4)}.camera.instance segmentation.png")
            segmask = Image.open(label_filepath).convert('L')  # grayscale
            segmask = np.array(segmask)
            segmask = Image.fromarray((segmask > 0).astype(np.uint8)*255)
            
            return segmask
            
        # replace background on both images
        bg_img_path = os.path.join(self.bg_img_root, img_pair_info["A1_bg_img"])
        bg_img = Image.open(bg_img_path)
        segmask = get_segmentation_mask(sequences[0], frame_ids[0])
        anchor = replace_background(anchor, segmask, bg_img, 
                                    transforms, self.img_size, "anchor")
        
        bg_img_path = os.path.join(self.bg_img_root, img_pair_info["B2_bg_img"])
        bg_img = Image.open(bg_img_path)
        segmask = get_segmentation_mask(sequences[1], frame_ids[1])
        sample = replace_background(sample, segmask, bg_img, 
                                    transforms, self.img_size, "sample")
        
        return anchor, sample
    
    def _get_max_orientation_diff(self, nQD:float) -> float:
        ''' From the exact norm of quaternion difference value, extract the upper
        limit of the bin to which this orientation belongs. '''
        
        nqd_thresholds = {0:(0,0), 1:(0.005,0.1), 2:(0.1,0.2), 3:(0.2,0.3),
                            4:(0.3,0.4), 5:(0.4,0.5), 6:(0.5,0.6), 7:(0.6,0.7), 
                            8:(0.7,0.8), 9:(0.8,0.9), 10:(0.9,1)}
                    
        for thresholds in nqd_thresholds.values():
            if nQD >= thresholds[0] and nQD <= thresholds[1]:
                return thresholds[1]
        
    def __getitem__(self, idx:int):
        ''' Determinstically go through dataset & return pairs with label. 
                
        Returns:
            anchor (pil Image):
            sample (pil Image):
            change_mask (pil Image):
            variable_of_interest (float): upper bound on the bin of either 
                orientation differences (nQD), translation (pixels) or scale (ratio)
            max_parts_diff (int): upper bound on the bin of the number of parts
                that are changed between anchor and sample images.
        '''
        # get image pair coordinates
        img_pair = self.determinstic_set[idx]
        
        # if orientation difference is the changing variable
        anchor_seq, anchor_frame = img_pair["A1"]
        sample_seq, sample_frame = img_pair["B2"]
        max_parts_diff = img_pair["n_differences"][1]
    
        # get states of both images
        anchor_state = self.state_list[anchor_seq*self.n_frames+anchor_frame]
        sample_state = self.state_list[sample_seq*self.n_frames+sample_frame]
        
        # load images and change mask
        anchor = self._load_image(anchor_seq, anchor_frame)
        sample = self._load_image(sample_seq, sample_frame)
        change_mask = self._load_binary_change_mask(anchor_seq, anchor_frame, 
                                        sample_frame, anchor_state, sample_state)
                    
        variable_of_interest = self._get_max_orientation_diff(
                                img_pair["quaternion_difference"])
        crop_params = {"anchor": img_pair["A1_crop"],
                        "sample": img_pair["B2_crop"]}
        tf = EvalTransforms('unaligned', (0,0), self.img_size, 
                self.norm_type, None, None, crop_params)
    
        # apply transforms
        anchor = tf(anchor, 'anchor')
        sample = tf(sample, 'sample')
        change_mask = tf(change_mask, 'label')
        
        if self.random_background:
            anchor, sample = self._randomize_image_background(anchor, sample, 
                                            img_pair, [anchor_seq, sample_seq], 
                                            [anchor_frame, sample_frame], tf)

        return anchor, sample, change_mask, variable_of_interest, max_parts_diff 