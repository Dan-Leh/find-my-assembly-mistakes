import numpy as np
from torchvision import utils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def make_numpy_grid(tensor_data, pad_value=0,padding=0):
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data[0:8], nrow=8, pad_value=pad_value,
                          padding=padding)
    vis = np.array(vis.cpu()).transpose((1,2,0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis


class LossPlotter():
    ''' Take care of plotting learning rate, loss, f1 and iou every epoch. '''
    
    def __init__(self, output_path:str, prev_results_dir:str="") -> None:
        """ Initialize variables.
        
        Arguments:
            output_path (str): the path where all results are saved (from config)
            prev_results_dir (str): if resuming training from a previous run, this
                variable indicates the output_path of that run, so that new losses &
                scores are appended to old ones for plotting.
        """
        self.input_path = "" if prev_results_dir == "" \
            else os.path.join(prev_results_dir, "metrics") 
        self.metrics = {}
        self.metrics['train'] = {}
        self.metrics['val'] = {}
        self.plot_dir = os.path.join(output_path, 'metrics')
        os.mkdir(self.plot_dir) 
        
    def initialize_metric(self, split:str, metric_keys:list) -> None:
        if self.input_path == "": # if training from scratch
            for key in metric_keys:
                self.metrics[split][key] =[]
        else: # if we are resuming training
            prev_path = os.path.join(self.input_path, f"{split}_metrics.csv")
            df = pd.read_csv(prev_path)
            prev_metrics = df.to_dict(orient='list')
            self.metrics[split] = prev_metrics
        
    def update_score_log(self, split:str, scores:dict, loss:float, epoch_id:int, 
                                                                lr:float) -> None:
        ''' Get new loss/lr/score values at the end of an epoch. '''
        # add variables to scores
        scores['loss'] = loss
        scores['epoch'] = epoch_id
        scores['lr'] = lr
        
        if len(self.metrics[split]) == 0: 
            self.initialize_metric(split, scores.keys())
        
        # append values of current epoch to previous epochs
        for key in scores.keys():
            self.metrics[split][key].append(scores[key])
        
    def plot_losses(self, split:str):
        ''' Save a csv file of all the metrics and plot loss, f1, iou and lr. '''
        
        # make and save csv file
        csv_path = os.path.join(self.plot_dir,f'{split}_metrics.csv')
        pd.DataFrame(self.metrics[split]).to_csv(csv_path, index=False)

        _, ax = plt.subplots(4,1, sharex=True, figsize=(6,9))
        if (self.metrics['val'] != {}) and (self.metrics['train'] != {}):
            for i, metric in enumerate(['f1_1', 'iou_1', 'loss', 'lr']):  
                # plot training curves             
                ax[i].plot(self.metrics['train']['epoch'],
                           self.metrics['train'][metric], 
                           linestyle='-', color='blue', label='train')
                if metric != 'lr':  # plot validation curves
                    ax[i].plot(self.metrics['val']['epoch'], 
                               self.metrics['val'][metric], 
                               linestyle='-', color='orange', label='val')
                    ax[i].set_title(metric + ' score')
                else:
                    ax[i].set_title('learning rate')
                ax[i].legend()
                ax[i].grid()
        ax[3].set_xlabel('Training epoch')
        
        plot_path = os.path.join(self.plot_dir,f'loss_plots.png')
        plt.savefig(plot_path)
        plt.close()
            

class Score_tracker_test_set():
    def __init__(self, output_path, var_of_interest, avg_type):
        self.metrics = {}
        self.output_path = output_path
        self.var_of_interest = var_of_interest
        self.avg_type = avg_type
        # os.mkdir(self.output_path)
        
        self.metrics_initialized=False
        
    def initialize_metric(self, metric_keys):
        for key in metric_keys:
            self.metrics[key] = []
        self.metrics_initialized = True
        
    def update_score_log(self, scores, current_config):
        change_of_interest, parts_diff = current_config
        scores[self.var_of_interest] = change_of_interest
        scores['parts_diff'] = parts_diff
        
        if not self.metrics_initialized: 
            self.initialize_metric(scores.keys())
            
        for key in scores.keys():
            self.metrics[key].append(scores[key])
            
    
    def take_average(self, end=False):
        assert self.var_of_interest == 'pose' , "this function should only be called if trying to track the error for each pose"
        df = pd.DataFrame(self.metrics)
        if not end: # reinitialize metrics so they don't grow huge
            last_row = df.iloc[-1] # last contains new pose
            df = df[:-1] # remove last row from table
            for key in df.keys():
                self.metrics[key] = [last_row[key]]
        self.initialize_metric(self.metrics.keys()) # make metrics 0 again
        
        means = pd.DataFrame(df.mean()).T
        try: # append to df if df exists, else initialize df
            self.df_averages = self.df_averages._append(means)
        except AttributeError:
            self.df_averages = means
        
        csv_path = os.path.join(self.output_path, f'metrics_avg_per_pose_{self.avg_type}.csv')
        self.df_averages.to_csv(csv_path, index=False)
            
            
    def save_scores(self):
        
        if self.var_of_interest == "pose":
            self.take_average(end=True)
        else: 
            csv_path = os.path.join(self.output_path, f'eval_metrics_{self.avg_type}.csv')   
            df = pd.DataFrame(self.metrics)
            df.to_csv(csv_path, index=False)
            print(f'Table containing changing {self.var_of_interest} has been saved.')
        
            v_min = {'iou_1': 0.2, 'f1_1': 0.4}
            v_max = {'iou_1': 1, 'f1_1': 1}
            swap_classes = {'iou_1': 'iou_0', 'f1_1': 'F1_0'}
            for metric_of_interest in ['f1_1', 'iou_1']:
                all_score = df[metric_of_interest].to_numpy()
                all_voi = df[self.var_of_interest].to_numpy() # voi = variable of interest
                all_parts_diffs = df['parts_diff'].to_numpy()
                all_score_swapped_classes = df[swap_classes[metric_of_interest]].to_numpy()
                
                voi_bins = np.sort(df[self.var_of_interest].unique())
                parts_diffs_bins = np.sort(df.parts_diff.unique())

                scores = np.zeros((len(voi_bins), len(parts_diffs_bins)))
                for i,voi in enumerate(voi_bins):
                    for j,parts_diff in enumerate(parts_diffs_bins):
                        if j>0:
                            score_bin = all_score[(all_voi==voi) & (all_parts_diffs==parts_diff)]
                            scores[i,j] = np.mean(score_bin)  # should just be a single entry, but was multiple in the past, so using mean for backward compatibility
                        elif j==0:
                            score_bin = all_score_swapped_classes[(all_voi==voi) & (all_parts_diffs==parts_diff)]
                            scores[i,j] = (np.mean(score_bin)-0.99) * 100
                
                # plotting the heatmap 
                plt.figure()
                hm = sns.heatmap(data=scores, annot=True, xticklabels=parts_diffs_bins, 
                        yticklabels=voi_bins, vmin=v_min[metric_of_interest], vmax=v_max[metric_of_interest]) 
                hm.set(xlabel="Maximum parts difference", 
                    ylabel=f"Maximum {self.var_of_interest} difference",
                    title=f"{metric_of_interest} wrt {self.var_of_interest} and parts difference")

                plt_path = os.path.join(self.output_path,f'{self.var_of_interest}_{metric_of_interest}_{self.avg_type}.png')
                plt.savefig(plt_path)
                plt.close()
                    
                print(f'Plots containing {metric_of_interest} for {self.var_of_interest} change have been saved.')
            
        return