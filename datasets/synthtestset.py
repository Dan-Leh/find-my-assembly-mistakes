import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

from utils.eval_transforms import EvalTransforms

class EvalDataset(Dataset):
    ''' Provides synthetic image pairs & ground truth for testing. '''
    
    def __init__(
        self,
        data_list_filepath: str = "", 
        norm_type: str = 'imagenet',
        img_size: tuple = (256,256), 
        test_type: str = "",
        ignore_0_change: bool = True,
        path_to_clean_imgs: str = "",
        dirty_img: str = "",
        more_nqd_bins: bool=False
        ):
        '''
        Arguments:
            data_list_filepath (str): path to json file containing IDs to images
                to make deterministic pairs 
            norm_type (str): which normalization to apply to input images
            img_size (tuple of ints): height and width of images that are outputted
            test_type (str): whether we are testing for orientation difference, 
                translation, scaling or roi cropping. 
            ignore_0_change (bool): skip testing on image pairs that have no 
                meaningful change as those have 0 IoU (saves time & compute)
            path_to_clean_imgs (str): When evaluating lower quality images, this
                gives the path to clean images of the same states & poses
            dirty_img (str): which image should be dirty, either 'anchor', 'sample'
                (in which case the other one is kept clean) or 'both'. Empty string
                means we are not testing on dirty images
            more_nqd_bins (bool): if true, have plot the results using the 11 nQD ranges
                from 0 to 1 nQD in incremenets of 0.1, else use the previous 4 bins: 
                0, 0-0.1, 0.1-0.2 and 0.2-1
        '''
        self.norm_type = norm_type
        self.img_size = img_size
        self.test_type = test_type
        self.more_nqd_bins = more_nqd_bins
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
        if (test_type in ["orientation", "roi_aligned"] or self.make_roi_crops):
            self.id2indexdict = self._id2index() 
            # list containing the state of each image in set
            self.state_list = self._load_json('state_list.json') 
            # get the number of frames per sequence (used for indexing in state_list)
            seq_dir = os.path.join(self.path_to_data,"sequence0000")
            self.n_frames = len(os.listdir(seq_dir))//3  # 3 files per frame
        
        else:  # images are precropped and saved as pngs
            self._decode_data_paths(dirty_img, path_to_clean_imgs)
        
    def _decode_data_paths(self, dirty_img:str, path_to_clean_imgs:str):
        ''' If we are testing dirty images, take care of whether sample, reference
        or both should contain corruptions '''
        # default values:
        self.path_to_anchors = os.path.join(self.path_to_data, "References")
        self.path_to_samples = os.path.join(self.path_to_data, "Samples")
        self.path_to_masks = os.path.join(self.path_to_data, "ChangeMasks")
        if dirty_img == 'anchor':
            self.path_to_samples = os.path.join(path_to_clean_imgs, "Samples")
        elif dirty_img == 'sample':  # mask needs to correspond to anchor image
            self.path_to_anchors = os.path.join(path_to_clean_imgs, "References")
            self.path_to_masks = os.path.join(path_to_clean_imgs, "ChangeMasks")
            
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
    
    def _get_max_orientation_diff(self, nQD:float) -> float:
        ''' From the exact norm of quaternion difference value, extract the upper
        limit of the bin to which this orientation belongs. '''
        
        if self.more_nqd_bins:
            nqd_thresholds = {0:(0,0), 1:(0.005,0.1), 2:(0.1,0.2), 3:(0.2,0.3),
                              4:(0.3,0.4), 5:(0.4,0.5), 6:(0.5,0.6), 7:(0.6,0.7), 
                              8:(0.7,0.8), 9:(0.8,0.9), 10:(0.9,1)}
        else:
            nqd_thresholds = {0:(0,0), 1:(0.005,0.1), 2:(0.1,0.2), 3:(0.2, 1)}
            
        for thresholds in nqd_thresholds.values():
            if nQD >= thresholds[0] and nQD <= thresholds[1]:
                return thresholds[1]
        
    def __getitem__(self, idx:int):
        ''' Determinstically go through dataset & return pairs with label. 
        
        If self.test_type is eg. translation, scaling, or roi cropping, random 
        augmentations have been applied and image pairs have been saved with them. 
        Else,  data_list_filepath contains the paths to deterministic iamge pairs
        that are loaded from dataset iwithout applying any randomization.
        
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
        if (self.test_type == "orientation" or 
                self.test_type == "roi_aligned" or 
                self.make_roi_crops):
            ref_seq, ref_frame = img_pair["A1"]
            sample_seq, sample_frame = img_pair["B2"]
            max_parts_diff = img_pair["n_differences"][1]
        
            # get states of both images
            ref_state = self.state_list[ref_seq*self.n_frames+ref_frame]
            sample_state = self.state_list[sample_seq*self.n_frames+sample_frame]
            
            # load images and change mask
            anchor = self._load_image(ref_seq, ref_frame)
            sample = self._load_image(sample_seq, sample_frame)
            change_mask = self._load_binary_change_mask(ref_seq, ref_frame, 
                                            sample_frame, ref_state, sample_state)
                        
            if self.test_type == "orientation":
                variable_of_interest = self._get_max_orientation_diff(
                                                    img_pair["quaternion_difference"])
                variable_of_interest = img_pair["quaternion_difference"]
                tf = EvalTransforms(self.test_type, anchor.size, self.img_size, 
                                    self.norm_type, None, None)
            elif self.test_type == "roi_aligned":
                variable_of_interest = self._get_max_orientation_diff(
                                                    img_pair["quaternion_difference"])
                # load segmentation masks to make roi crops
                SegMaskImage1, SegMaskImage2 = self._load_segmentation_masks(ref_seq, 
                                                sample_seq, ref_frame, sample_frame) 
                tf = EvalTransforms(self.test_type, (0,0), self.img_size, 
                        self.norm_type, SegMaskImage1, SegMaskImage2)
            elif self.make_roi_crops:
                variable_of_interest = self._get_max_orientation_diff(
                                        img_pair["quaternion_difference"])
                crop_params = {"anchor": img_pair["A1_crop"],
                               "sample": img_pair["B2_crop"]}
                tf = EvalTransforms(self.test_type, (0,0), self.img_size, 
                        self.norm_type, None, None, crop_params)
            else:
                raise ValueError(f'Invalid test type: {self.test_type}')
        
            # apply transforms
            anchor = tf(anchor, 'anchor')
            sample = tf(sample, 'sample')
            change_mask = tf(change_mask, 'label')
        
        else: # get saved image pairs with fixed translation or scale or ROI crop
            assert self.img_size[0] == self.img_size[1] == 256, \
                "The saved image pairs are of size (256,256)"
            
            img_names = img_pair['img_name']
            variable_of_interest = img_pair['variable_of_interest']
            max_parts_diff = img_pair["n_differences"][1] \
                if type(img_pair["n_differences"])!=int else img_pair["n_differences"] 

            # get path to images
            path_anchor = os.path.join(self.path_to_anchors, img_names)
            path_sample = os.path.join(self.path_to_samples, img_names)
            path_mask = os.path.join(self.path_to_masks, img_names)

            # load images
            anchor = Image.open(path_anchor).convert("RGB")
            sample = Image.open(path_sample).convert("RGB")
            change_mask = Image.open(path_mask).convert("L")
            
            # get & apply transforms
            tf = EvalTransforms('', anchor.size, self.img_size, self.norm_type, None, None)
            anchor = tf(anchor, 'anchor')
            sample = tf(sample, 'sample')
            change_mask = tf(change_mask, 'label')

        # sanity checks
        assert (anchor.shape == sample.shape), "Anchor and sample images do not have "\
                                                    "the same dimensions"
        assert (change_mask.shape[-2:] == anchor.shape[-2:]), "Change mask and anchor "\
                                                "image do not have the same dimensions"
        assert torch.count_nonzero(change_mask) + torch.count_nonzero(1-change_mask) == \
                            self.img_size[0]*self.img_size[1], \
                            "There are values in the change_mask that are neither 0 nor 1"
        
        return anchor, sample, change_mask, variable_of_interest, max_parts_diff 