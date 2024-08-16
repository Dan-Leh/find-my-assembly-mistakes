import os
import json
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
from typing import Union, Optional

from utils.transforms import Transforms
from utils.background_randomizer import replace_background

class SyntheticChangeDataset(Dataset):
    """ Dataset for pairing two synthetic images & generating ground truth change mask """
    
    def __init__(
        self,
        data_path: str = "",
        orientation_thresholds: tuple = (0, 0.1),   
        parts_diff_thresholds: tuple = (0, 5),     
        preprocess: bool = False,                 
        img_transforms: dict = {},
        split: str = "train"
        ):
        '''
        Arguments:
            data_path (string): path to the folder containing the dataset
            orientation_thresholds (tuple): the minimum & maximum nQD (norm of quaternion 
                                            differences) between two images in a pair
            parts_diff_thresholds (tuple): the minimum & maximum number of different
                                           parts between two images in a pair
            preprocess (bool): if True, run preprocessing functions, i.e. save list 
                                of all states and orientation differences in dataset
            img_transforms (dict): a dictionary containing all the image 
                                transforms from the config file
            split (str): train, test or val. Only used for randomizing background images
                as there is a separate list preallocated to each split.
        '''
        assert data_path != "", "Please enter a folder name when instantiating ChangeDataset"
        
        # make input variables class-wide
        self.orientation_thresholds = orientation_thresholds
        self.parts_diff_thresholds = parts_diff_thresholds
        self.img_transforms = img_transforms
        
        # Load metadata from the dataset
        self.path_to_data = data_path
        self.id2indexdict = self._id2index() 
        self.data_list, self.n_sequences, self.n_frames = self._make_data_list()
        self.state_dict, self.state_list = self._state_table(preprocess)
        self.nqd_table = self._orientations_table(preprocess) 
        
        if img_transforms['random_background']:  # initialize vars for randomizing bg
            self.randomize_background = True
            self.bg_img_folder = f"/shared/nl011006/res_ds_ml_restricted/dlehman/COCO_Images"
            self.bg_img_root = os.path.join(self.bg_img_folder, "unlabeled2017")
            img_list_path = os.path.join(self.bg_img_folder, f"{split}_img_list.json")
            with open(img_list_path, 'r') as f:
                self.bg_img_list = json.load(f)
            f.close()
    
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
    
    def _save_json(self, name: str, data: Union[list,dict]) -> None:
        ''' Save a json file with specified name in the data directory
         
        Arguments:
            name (str): the name of the json file that should be saved
            data (list): the variable that is to be saved as a json file 
        '''
        path = os.path.join(self.path_to_data, name)
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        f.close()
        
    def _load_json(self, name: str) -> list:
        ''' load json file with specified name from data directory '''
        
        path = os.path.join(self.path_to_data, name)
        with open(path, 'r') as f:
            data = json.load(f)
        f.close()
        return data
        
    def _orientations_table(self, save: bool = False) -> np.ndarray: 
        ''' Make a table that contains the nQD between all poses in the dataset.
         
        Arguments:
            save (bool): if True, the table is saved as a json file (part of preprocessing),
                         if False, the table is loaded from a prior saved json file
        Returns:
            nqd_table (np.array): a 2D array containing the difference in orientation
                between every sequence in norm of quaternion difference (nQD)
        '''
        
        if save: # construct and save orientation table as json
            print('Creating orientations table')
            
            # find the orientation of each sequence
            orients = []
            for sequence_idx in range(self.n_sequences):
                # load the first frame of each sequence, they all have identical orientation 
                file_name = os.path.join(f"sequence{str(sequence_idx).zfill(4)}", 
                                         "step0000.frame_data.json") 
                data = self._load_json(file_name)
                # extract the 3D bounding box quaternion from the json file
                for annotation in data["captures"][0]["annotations"]:
                    if annotation["id"] == "bounding box 3D":
                        orients.append(annotation["values"][0]["rotation"]); break
            
            orients = np.array(orients)
            n_orients = len(orients)
            nqd_table = np.zeros((n_orients, n_orients))
            # compute the difference in quaternions between each pair of orientations        
            for i in range(n_orients):
                for j in range(n_orients): 
                    nqd_table[i][j] = min(np.linalg.norm(orients[i] - orients[j]), 
                                          np.linalg.norm(orients[i] + orients[j]))
            
            self._save_json('orientation_table.json', nqd_table.tolist())
                          
        else: # load the orientation table from a json file
            assert os.path.exists(os.path.join(self.path_to_data, 'orientation_table.json')), \
            'The file "orientation_table.json" does not exist. Make sure to run preprocess.py '\
            'when using a new dataset.'
            nqd_table = np.array(self._load_json('orientation_table.json'))
                
        return nqd_table
    
    def _state_table(self, save = False) -> tuple[dict, list]:
        ''' Create/load two files containing information on the states in the dataset.
        
        The assembly object state contained in each image is described by a binary
        sequence, where each bit indicates whether a part of the object is visible 
        in the image or not. This function saves all states in dataset and the indices
        of images associated with those states to subsequently make pairing images 
        based on their difference in states.
        
        Arguments:
            save (bool): if True, the table is saved as a json file,
                         if False, the table is loaded from a prior saved json file
        Returns:
            state_dict (dict): keys correspond to states and values are a list of all 
                               the sequence and frame numbers of corresponding images
            state_list (list): a list containing all states to more easily find state
                               of each iamge from image index in dataset
        '''
        
        if save: # construct the state table and save it as a json file
            print("Creating state table")
            state_dict = {}
            state_list = []
            
            for sequence_idx in range(self.n_sequences):
                for frame_idx in range(self.n_frames):
                    instances = self._get_instance_info(sequence_idx, frame_idx)                     
                    # encode visible parts into binary state representation
                    state = 0
                    for instance in instances:
                        # convert the segmentation label ID to part index in binary
                        part_idx = self.id2indexdict[instance["labelId"]] 
                        state += (1 << part_idx)

                    # add state to the dictionary
                    if state not in state_dict.keys():
                        state_dict[state] = [[sequence_idx, frame_idx]]
                    else:
                        state_dict[state].append([sequence_idx, frame_idx])
                    state_list.append(state)
                    
            # count the number of duplicate images
            n_duplicates = 0
            for key in state_dict.keys():
                seq_frame_pairs = np.array(state_dict[key])
                unique_sequences = np.unique(seq_frame_pairs[:,0])
                n_duplicates += len(state_dict[key]) - len(unique_sequences) 
            print(f"Total number of states: {len(state_dict.keys())}. Number of "\
                  f"duplicates (where multiple identical states are in the same sequence): "\
                  f"{n_duplicates}")
                    
            self._save_json('state_table.json', state_dict)
            self._save_json('state_list.json', state_list)
                            
        else: # load the state table from a json file
            assert os.path.exists(os.path.join(self.path_to_data, 'state_table.json')), \
            'The file "state_table.json" does not exist. Make sure to run preprocess.py '\
            'when using a new dataset.'
            state_dict = self._load_json('state_table.json')
            state_dict = {int(k): v for k, v in state_dict.items()} # str to int for state
            state_list = self._load_json('state_list.json')
        state_list = np.array(state_list)

        return state_dict, state_list
        
    def _make_data_list(self):
        '''Get the indeces of all files in the dataset.
        
        Returns:
            data (list): a list of tuples of the form (sequence, frame, state)
            n_sequences (int): the number of sequences (i.e. poses) in the dataset
            n_frames (int): the number of frames (i.e. states) per sequence in the dataset
        '''
        data = []
        sequence_idx = 0
        for sequence in sorted(os.listdir(self.path_to_data)):
            seq_dir_path = os.path.join(self.path_to_data, sequence)
            if os.path.isdir(seq_dir_path): 
                frame_idx=0
                for frame in sorted(os.listdir(seq_dir_path)):
                    if frame.endswith(".frame_data.json"): # count each file only once
                        data.append((sequence_idx, frame_idx))
                        frame_idx += 1
                sequence_idx += 1
        n_frames = frame_idx
        n_sequences = sequence_idx
        
        return data, n_sequences, n_frames 
        
    def _id2index(self) -> dict: 
        ''' Make dictionary that converts labelIDs to part index.
        
        Each state is described by a binary sequence, where each bit indicates 
        whether a part of the object is in that state or not. The index of each
        part of the assembly object in that binary sequence is given by the 
        "PartList.json" file. This function makes a dictionary that contains the 
        part index corresponding to each part ID contained in the instance segmentation
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
        
    def __len__(self):
        return len(self.data_list)

    def _load_image(self, sequence: int, frame: int) -> Image.Image:
        ''' Load an image from the dataset given a sequence and frame index. '''
        
        file_path = os.path.join(self.path_to_data, 
                                 f"sequence{str(sequence).zfill(4)}", 
                                 f"step{str(frame).zfill(4)}.camera.png")
        image = Image.open(file_path).convert("RGB")
        
        return image
    
    def _load_segmentation_masks(self, seq_a: int, seq_b: int, frame_1: int, 
                                frame_2:int) -> tuple[np.ndarray,np.ndarray]:
        ''' load two binary foreground/background segmentation masks.
        
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
    
    def _randomize_image_background(self, anchor:torch.Tensor, 
                        sample:torch.Tensor, sequences:list[int,int], 
                        frame_ids:list[int,int], transforms:Transforms
                        ) -> list[torch.Tensor, torch.Tensor]:
        ''' Return assembly object pasted on random background.
    
        Arguments:
            anchor (tensor): transformed anchor image
            sample (tensor): transformed sample image
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
        # for anchor and sample image 
        for i, (sequence, frame) in enumerate(zip(sequences, frame_ids)):
            # load background image
            bg_img_filenames = random.choice(self.bg_img_list)
            bg_img_path = os.path.join(self.bg_img_root, bg_img_filenames)
            bg_img = Image.open(bg_img_path)
            
            # load binary segmentation mask
            label_filepath = os.path.join(self.path_to_data, f"sequence{str(sequence).zfill(4)}",
                                fr"step{str(frame).zfill(4)}.camera.instance segmentation.png") 
            segmask = Image.open(label_filepath).convert('L') # make the image grayscale
            segmask = np.array(segmask)
            segmask = Image.fromarray((segmask > 0).astype(np.uint8)*255)
            
            # cut out assembly object and add to new background
            if i == 0:  # anchor image
                anchor = replace_background(anchor, segmask, bg_img, transforms, 
                                            self.img_transforms['img_size'], 'anchor')
            elif i == 1:  # anchor image
                sample = replace_background(sample, segmask, bg_img, transforms, 
                                            self.img_transforms['img_size'], 'sample')
        return anchor, sample
    
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
            
    def _filter_allowed_states(self, states: list, state_1: int) -> list:
        ''' get rid of all states that fall outside the specified thresholds '''
        
        n_differences = np.zeros_like(states)
        for i, state_i in enumerate(states):
            n_differences[i] = bin(state_1 ^ state_i).count('1')
            
        allowed_states = states[(n_differences >= self.parts_diff_thresholds[0]) * 
                                (n_differences <= self.parts_diff_thresholds[1])]
        return allowed_states
    
    def _frame_from_state(self, state: int, sequence: int) -> int:
        ''' Get the frame index for an image with given the sequence index and state '''
        img_indices = np.array(self.state_dict[state])
        img_indices_correct_seq = img_indices[img_indices[:,0] == sequence]
        return img_indices_correct_seq[0,1]
       
    def _find_sequenceb_and_state2(self, sequence_a:int, state_1:int) \
                                -> tuple[Optional[int],Optional[int],Optional[float]]: 
        ''' Pair two images based on chosen orientation and part difference thresholds 
        
        Given an anchor image with specified state and sequence and frame index, find 
        a sample image to pair it with, while respecting the thresholds set on 
        orientation difference and part differences.
        
        Arguments:
            sequence_a (int): sequence index of the anchor image
            state_1 (int): state of the anchor image
        Returns:
            sequence_b (int): sequence index of the chosen sample image
            state_2 (int): state of the chosen sample image
            orientation_diff (float): norm of quaternion difference of anchor & sample
        '''
        # Make a list of all possible sequences (orientations) to choose from
        all_sequences = np.arange(self.n_sequences)
        nqd_from_a = self.nqd_table[sequence_a] 
        nqd_tresholds_mask = (nqd_from_a >= self.orientation_thresholds[0]) * \
                             (nqd_from_a <= self.orientation_thresholds[1]) 
        sequences_to_choose_from = all_sequences[nqd_tresholds_mask] 
        
        # get all unique states that have same pose as anchor
        min_idx = sequence_a*self.n_frames
        max_idx = (sequence_a+1)*self.n_frames
        states_a = np.unique(self.state_list[min_idx:max_idx])
        # filter out states that have too many or too few differences in parts
        states_a_filtered = self._filter_allowed_states(states_a, state_1) 
        
        # loop through all sequences/poses until appropriate pair is found
        pair_found = False
        while (not pair_found) and (len(sequences_to_choose_from) > 0):
            sequence_b = random.choice(sequences_to_choose_from)
            # get the possible states of randomly chosen sequence 
            min_idx = sequence_b*self.n_frames
            max_idx = (sequence_b+1)*self.n_frames
            states_b = np.unique(self.state_list[min_idx:max_idx])

            # check if there are any states from states_b that are also in sequence_a
            # to avoid pairing images whose states differ due to occlusion
            shared_states = np.intersect1d(states_a_filtered, states_b, 
                                           assume_unique=True)
            
            if (shared_states.size > 0) and (state_1 in states_b): 
                state_2 = random.choice(shared_states)
                pair_found = True
                orientation_diff = (nqd_from_a[sequence_b])
                return sequence_b, state_2, orientation_diff
            else:  # remove choices that were tried and keep searching
                sequences_to_choose_from = sequences_to_choose_from[
                                sequences_to_choose_from!=sequence_b] 
        
        if not pair_found:  # randomly choose new anchor image and try again
            return None, None, None
                
    def __getitem__(self, idx:int, unpairable:bool=False): 
        ''' Load image pairs
        
        Nomenclature in code: 
                "a" corresponds to the pose (sequence) of the first (anchor) image, 
                "b" corresponds to the pose (sequence) of the second (sample) image.
                "1" corresponds to the state/frame of the first (anchor) image,
                "2" corresponds to the state/frame of the second (sample) image.
                "nqd" refers to norm of quaternion difference
            Thus: a1 is the anchor image, chosen deterministically, & b2 is the '
            sample image, chosen such that there is a corresponding a2.
            We need to load the labels for a1 and a2 to make a change mask
        
        Arguments:             
            idx (int): the index of the anchor image to use
            unpairable (bool): whether the anchor image at specified index was unable to find
                a pair. If so, a new anchor image is picked randomly until a pair is found,
                and the number of images that were 'unpairable' is logged.
        Returns:
            image_a1 (Image.Image): anchor image 
            image_b2 (Image.Image): sample image
            change_mask (Image.Image): ground truth binary change mask
            nqd (float): orientation difference in "norm of quaternion difference" (nQD)
            state_1 (int): binary state representation of anchor, used for tracking statistics
            state_2 (int): binary state representation of sample, used for tracking statistics
            unpairable (int): 0 if the anchor corresponding to idx was pairable, otherwise 1
        '''
        # Extract sequence A and state 1 based on index
        sequence_a, frame_1 = self.data_list[idx]
        state_1 = self.state_list[idx]
        
        # Extract sequence b and state 2
        sequence_b, state_2, nqd = self._find_sequenceb_and_state2(sequence_a, state_1)
        # If no pair can be found for sequence_a, frame_1, take a random image and try again
        if (sequence_b == state_2 == None):  
            return self.__getitem__(idx = random.randint(0,self.__len__()-1), unpairable=True)
            
        # get the frame index corresponding to the pair of images with state 2
        frame_b2 = self._frame_from_state(state_2, sequence_b)
        frame_a2 = self._frame_from_state(state_2, sequence_a)     
        
        # load images and change mask
        image_a1 = self._load_image(sequence_a, frame_1)
        image_b2 = self._load_image(sequence_b, frame_b2)
        change_mask = self._load_binary_change_mask(sequence_a, frame_1, frame_a2, 
                                                                state_1, state_2)
                
        # instantiate transforms so that the same transforms are applied to all images
        seg_mask_img_1, seg_mask_img_2 = self._load_segmentation_masks(sequence_a, 
                                                            sequence_b, frame_1, frame_a2) 
        tf = Transforms(seg_mask_img_1, seg_mask_img_2, self.img_transforms)
        
        # apply transforms
        image_a1 = tf(image_a1, 'anchor')
        image_b2 = tf(image_b2, 'sample')
        change_mask = tf(change_mask, 'label')
        
        if self.randomize_background:
            image_a1, image_b2 = self._randomize_image_background(image_a1, image_b2,
                                    [sequence_a, sequence_b], [frame_1, frame_b2], tf)

        # sanity checks
        assert (image_a1.shape == image_b2.shape), "Anchor and sample images do not have "\
                                                    "the same dimensions"
        assert (change_mask.shape[-2:] == image_a1.shape[-2:]), "Change mask and anchor "\
                                                    "image do not have the same dimensions"
        assert torch.count_nonzero(change_mask) + torch.count_nonzero(1-change_mask) == \
            self.img_transforms['img_size'][0]*self.img_transforms['img_size'][1], \
                "There are values in the change_mask that are neither 0 nor 1"
        
        return image_a1, image_b2, change_mask, nqd, state_1, state_2, int(unpairable) 
    