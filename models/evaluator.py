''' The code in this file has been built off the same-named file published in
"Remote Sensing Image Change Detection with Transformers": https://github.com/justchenhao/BIT_CD.git '''


import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import v2
from PIL import Image

from models.build_functions import build_model
from utils.metric_tool import ConfuseMatrixMeter
from utils.logger_tool import Logger
from utils.utils import denormalize
from utils.plotter import make_numpy_grid, TestScorePlotter


class CDEvaluator():
    ''' Evaluator of change detection on synthetic assembly images '''
    
    def __init__(self, args, dataloader, test_type, output_dir):
        ''' Initialize score trackers and variables for evaluation script.
         
        We define separate score trackers, one for micro- and one for macro-
        averaging. The former computes IoU by summing the confusion matrix
        across batches in the same 'category', and then computing one IoU score.
        The macro-averaged IoU is computed by taking the IoU score of each batch
        (each frame in case batchsize=1), and averaging that score across all 
        batches/images, giving access to additional information such as the 
        distribution of scores across frames.
        
        A 'category' refers to a range of controlled variables among test image 
        pairs, eg. orientation difference, number of part differences, scale 
        difference or translation difference between anchor and sample images.
        '''
        self.dataloader = dataloader
        
        self.n_samples_per_category = args.save_fig_iter
        # initialize the amount of change (in nQD and number of parts difference)
        self.curr_config = (-1,-1)

        # instantiate network
        self.net, self.device = build_model(args=args, train=False)
        
        self.changing_var = self._decode_test_type(test_type)
        
        # Instantiate classes to make & record the evaluation scores
        self.running_metric_micro = ConfuseMatrixMeter(n_class=2)
        self.running_metric_macro = ConfuseMatrixMeter(n_class=2)
        self.score_tracker_micro = TestScorePlotter(output_dir, 
                                                self.changing_var, 'micro')
        self.score_tracker_macro = TestScorePlotter(output_dir, 
                                                self.changing_var, 'macro')
        
        self.batch_size = 1  # required for macro-averaging

        # initialize logger
        logger_path = os.path.join(output_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)
        self.log_iter = args.log_iter
        # initialize variables
        self.net_pred = None
        self.batch = None
        self.batch_id = 0
        self.n_batches = len(self.dataloader)
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir
        self.norm_type = args.img_transforms['normalization']
        
    def _decode_test_type(self, test_type):
        ''' Turn test type into the variable that is being varied to make test categories.'''

        if test_type in ['scale', 'orientation', 'translation']:
            changing_var = test_type 
        elif test_type in ['present', 'roi_present', 'missing', 'roi_missing',
                           'roi', 'roi_aligned']:
            changing_var = 'orientation'
        else:
            raise ValueError(f"Unrecognized test type {test_type}")
        return changing_var

    def _load_checkpoint(self):

        if os.path.exists(os.path.join(self.checkpoint_dir, 'best_ckpt.pt')):
            ckpt_path = os.path.join(self.checkpoint_dir, 'best_ckpt.pt')
        elif os.path.exists(os.path.join(self.checkpoint_dir, 'last_ckpt.pt')):
            ckpt_path = os.path.join(self.checkpoint_dir, 'last_ckpt.pt')
        else:
            raise FileNotFoundError(f'no such checkpoint in {self.checkpoint_dir}')

        self.logger.write(f'loading checkpoint from {ckpt_path} \n')
        # load the entire checkpoint
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.to(self.device)

        # take note of checkpoint characteristics
        best_val_iou = checkpoint['best_val_iou']
        best_epoch_id = checkpoint['best_epoch_id']
        self.logger.write(f'Historical_best_iou = {best_val_iou:.4f}'
                          f'(at epoch {best_epoch_id})\n\n')

    def _visualize_label(self, gt=True):
        ''' Make a blend of anchor and change mask for visualization
        
        Arguments:
            gt (bool):  if true, return the blended ground truth change mask,
                        else return the blended predicted change mask. 
        ''' 
        if gt == True:
            label_batch = self.batch[2].to(torch.uint8) * 255
        else: 
            label_batch = torch.argmax(self.net_pred, dim=1, keepdim=False)
            label_batch = label_batch.to(torch.uint8) * 255
        
        ref_batch = denormalize(self.batch[0], self.norm_type)
        # restric number of images to visualize to max 8:
        n_images = self.batch_size if self.batch_size<8 else 8
        pred_vis = torch.zeros(n_images, 3, label_batch.shape[-2], 
                               label_batch.shape[-1]) # N x C x H x W
        # make the prediction overlap with the reference image
        for i in range(n_images):
            label = v2.ToPILImage()(label_batch[i])
            ref = v2.ToPILImage()(ref_batch[i])
            blend = Image.blend(ref.convert('RGBA'), label.convert('RGBA'), 0.8)
            pred_vis[i] = v2.functional.pil_to_tensor(blend.convert('RGB'))/255
        
        return pred_vis

    def _update_metric(self) -> None:
        """ Update running average of scores (every iteration). """

        target = self.batch[2].to(self.device).detach().cpu().numpy()
        net_pred = self.net_pred.detach()
        net_pred = torch.argmax(net_pred, dim=1).cpu().numpy()
        # update confusion matrices
        self.running_metric_macro.update_cm(net_pred, target, return_iou=False)
        self.running_metric_micro.update_cm(net_pred, target, return_iou=False)

        # for macro-averaging: convert confusion matrix to IoU & other scores 
        # each iteration and save/log for averaging later
        scores_macro = self.running_metric_macro.get_scores()  # get IoU & others
        self.score_tracker_macro.update_score_log(scores_macro, self.curr_config)
        self._reset_running_scores('macro')  # reset average meter
    
    def _get_batch_configuration(self):
        ''' Return the batch configuration
        
        The batch configuration is defined as a tuple containing the value
        of the variable of interest and the maximum amount of part differences
        between two images in a batch. The variable of interest can be the 
        upper bound on orientation difference (nQD), translation (pixel amount)
        or the amount of scale difference (ratio) between anchor and sample 
        images. 
        '''
        
        var_of_interest = self.batch[3]
        max_parts_diffs = self.batch[4]
        for i in range(self.batch_size-1):  # ensure batch configuration is contant
           assert var_of_interest[i] == var_of_interest[i+1],\
               f"In the same batch, we have different {self.changing_var} values"
           assert max_parts_diffs[i] == max_parts_diffs[i+1], \
               "In the same batch, we have two different max part differences"
        return var_of_interest[0].item(), max_parts_diffs[0].item()

    def _collect_running_batch_states(self):
        ''' Update states/scores and log text & images depending on epoch. '''
        
        var_of_interest, max_parts_diff = self._get_batch_configuration() 
        # check if we are in the same 'category' & update category if not
        if (var_of_interest, max_parts_diff) != self.curr_config:
            self._new_category_states(var_of_interest, max_parts_diff)
            
        self._update_metric()  # update confusion matrices / IoU scores
        
        if np.mod(self.batch_id, self.log_iter) == 0:
            message = f'Testing iteration [{self.batch_id},{self.n_batches}]\n'
            self.logger.write(message)

        ########## visualize results #########
        if self.n_printed < self.n_samples_per_category:
            self.n_printed += 1  # resets every time we enter new category
            
            anchor = denormalize(self.batch[0], self.norm_type)
            sample = denormalize(self.batch[1], self.norm_type)
            vis_anchor = make_numpy_grid(anchor)
            vis_sample = make_numpy_grid(sample)
            vis_pred = make_numpy_grid(self._visualize_label(gt=False))
            vis_gt = make_numpy_grid(self._visualize_label(gt=True))
            
            vis = np.concatenate([vis_anchor,vis_sample,vis_pred,vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            
            pathname = f'max_parts_diff_{max_parts_diff}({self.n_printed}).png'
            file_name = os.path.join(self.curr_vis_dir, pathname)
            plt.imsave(file_name, vis)
            
    def _new_category_states(self, var_of_interest, max_parts_diff:int) -> None:
        ''' Update states given that we are in new test category.
        
        A new test category means testing for a different combination of eg.
        orientation difference and range of part differences between anchor
        and sample images.
        '''
        # update score for specific category/configuration
        if self.running_metric_micro.initialized == True:  
            # if this is not the first iteration, compute scores based on the 
            # sum of all confusion matrices computed for the previous category
            scores = self.running_metric_micro.get_scores()  
            self.score_tracker_micro.update_score_log(scores, self.curr_config)
            self._reset_running_scores('micro')  # restart average meter for new category
        
        self.curr_config = (var_of_interest, max_parts_diff)  # update batch config
        
        # for visualization purposes, because we only save the first 
        # self.n_samples_per_category' images, update n_printed and folder to
        # in which to save images of new category
        self.n_printed = 0
        if self.changing_var == 'orientation' or self.changing_var.startswith('roi'):
            self.curr_vis_dir = os.path.join(self.vis_dir, 
                                             f"o_diff_up_to_{var_of_interest}")
        elif self.changing_var == 'translation':
            self.curr_vis_dir = os.path.join(self.vis_dir, 
                                             f"translation_of_{var_of_interest}")
        elif self.changing_var == 'scale':
            self.curr_vis_dir = os.path.join(self.vis_dir, 
                                             f"scale_ratio_of_{var_of_interest}")
        if not os.path.exists(self.curr_vis_dir):
            os.mkdir(self.curr_vis_dir)
            
    def _collect_final_states(self):
        ''' Call functions that take average and plot scores for each category. '''
        
        # save the micro-averaged scores from the last category & save for plotting
        scores = self.running_metric_micro.get_scores() 
        self.score_tracker_micro.update_score_log(scores, self.curr_config)
        
        # create heatmaps of scores per category and save csv with table of scores
        self.score_tracker_micro.save_scores() 
        self.score_tracker_macro.save_scores() 
        
    def _reset_running_scores(self, type=''):
        if type=='micro':
            self.running_metric_micro.reset()
        elif type=='macro':
            self.running_metric_macro.reset()
        elif type=='':
            self.running_metric_micro.reset()
            self.running_metric_macro.reset()

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch[0].to(self.device)
        img_in2 = batch[1].to(self.device)
        self.net_pred = self.net(img_in1, img_in2)

    def eval_models(self):

        self._load_checkpoint()

        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self._reset_running_scores('')
        self.net.eval()

        # Iterate over data.
        for self.batch_id, batch in enumerate(self.dataloader, 0):
            with torch.no_grad():
                self._forward_pass(batch)
            self._collect_running_batch_states()
        self._collect_final_states() 
