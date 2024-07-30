import os
import numpy as np
import matplotlib.pyplot as plt
from time import time

class StatTracker():
    ''' Track statistics about states and orientations of training data. '''
    
    def __init__(self, output_dir:str, split:str, batch_size:int) -> None:
        self.split = split
        self.batch_size = batch_size
        
        #create directory for saving stats
        self.save_dir = os.path.join(output_dir, r"Training statistics")
        os.makedirs(os.path.join(self.save_dir,"numpy_stats"), exist_ok=True) 
        
        # initialize vars
        self.orientation_diffs = []
        self.n_part_differences = []
        self.state_combi_list = []
        self.n_unpairable = 0  # total number up to current moment
        self.n_unpairable_in_dataset = 0  # number after one run through dataset
        self.prev_epochs_unpairable = 0
        self.iter_count = 0
        self.n_images_in_dataset = 0
        
        # time tracker
        self.time_stats = {}
        self.curr_time = time()
        
    def update_stats(self, batch_stat_info:list) -> None:
        ''' Add stats from current iteration to logged stats. '''
        orientation_diffs, states1, states2, unpairables = batch_stat_info
        self._update_orientations(orientation_diffs.numpy())
        self._update_parts_difference(states1.numpy(), states2.numpy())
        self._update_unpairable(unpairables.numpy())
        self.iter_count += 1
        
    def _update_combis(self, state1, state2):
        combined_state_id = state1 + (state2 << 34)
        if combined_state_id not in self.state_combi_list:
            self.state_combi_list.append(combined_state_id)
        
    def _update_orientations(self, quaternion_diffs):
        self.orientation_diffs += quaternion_diffs.tolist()
        
    def _update_parts_difference(self, states1, states2):
        for i in range(states1.shape[0]):
            difference_in_parts = states1[i] ^ states2[i]
            parts_diff = difference_in_parts.bit_count() 
            self.n_part_differences.append(parts_diff) 
            self._update_combis(states1[i], states2[i])

    def _update_unpairable(self, unpairables:int):
        self.n_unpairable += np.sum(unpairables)
            
    def end_of_dataloader(self) -> None:
        if self.n_unpairable_in_dataset == 0:  # first time function is called
            self.n_images_in_dataset = self.iter_count
        self.n_unpairable_in_dataset = self.n_unpairable - \
                                        self.prev_epochs_unpairable
        self.prev_epochs_unpairable += self.n_unpairable_in_dataset
        
    def save_stats(self) -> None:
        ''' Save histograms of orientation difference and number part difference,
        and save statistics about the number of unique state combinations and
        nunmber of images that could not find a pair given the thresholds on 
        orientation and part difference. '''
        
        # save numpy of orientation differences
        orientation_diffs = np.array(self.orientation_diffs)
        np.save(os.path.join(self.save_dir, "numpy_stats", 
                self.split+"_orientations.npy"), orientation_diffs)
        # make and save histogram of orientation differences
        plt.figure()
        plt.hist(orientation_diffs)
        plt.xlabel('Norm of Quaternion Difference')
        plt.ylabel('Frequency')
        plt.title("Orientation differences between image pairs")
        plt.savefig(os.path.join(self.save_dir, self.split+"_orientations.png"))
        plt.close()
        
        # save numpy of number of part differences
        n_part_differences = np.array(self.n_part_differences)
        np.save(os.path.join(self.save_dir, "numpy_stats", self.split+
                             "_state_diff.npy"), n_part_differences)
        # make and save histogram of number of part differences
        plt.figure()
        plt.hist(n_part_differences)
        plt.xlabel('Number of parts')
        plt.ylabel('Frequency')
        plt.title("Difference in parts between image pairs")
        plt.savefig(os.path.join(self.save_dir, self.split+"_state_diff.png"))
        plt.close()
        
        # stats on number of unpairable images and unique pairing
        total_n_images = self.iter_count*self.batch_size
        unpairable_perc = 100 * self.n_unpairable/(total_n_images) 
        with open(os.path.join(self.save_dir, 
                               self.split+"_unpairable.txt"), 'w') as f:
            f.truncate(0) # clear file
            # number of unpairable images
            if self.n_unpairable_in_dataset != 0:
                f.write(f"Total number of images that are unpairable in dataset:"
                        f"{self.n_unpairable_in_dataset} out of "
                        f"{self.n_images_in_dataset}\n")
            f.write(f"Percentage of images that are not pairable: "
                    f"{unpairable_perc:.2f}%\n\n")
            
            # unique state combinations
            n_combis = len(self.state_combi_list)
            perc_duplicates = 100* (total_n_images-n_combis)/total_n_images
            f.write(f"Number of unique state combinations: {n_combis} out of "
                    f"{total_n_images} total training pairs.\n")
            f.write(f"Percentage of state pairs that are duplicates: "
                    f"{perc_duplicates}.\n")
            f.write(f"Percentage of state pairs that are original: "
                    f"{100-perc_duplicates}\n")
        f.close()