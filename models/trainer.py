''' The code in this file has been built off the same-named file published in
"Remote Sensing Image Change Detection with Transformers": https://github.com/justchenhao/BIT_CD.git '''

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import types

import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader

from datasets.synthdataset import SyntheticChangeDataset
from utils.metric_tool import ConfuseMatrixMeter
from utils.loss_funcs import get_loss_func
from utils.plotter import make_numpy_grid, LossPlotter
from utils.logger_tool import Logger, Timer
from utils.data_stats import StatTracker
from utils.loss_funcs import get_loss_func
from utils.utils import remove_train_augmentations, denormalize
from models.build_functions import get_optimizer, get_scheduler, build_model


class CDTrainer():
    ''' Trainer of change detection on synthetic assembly images '''

    def __init__(self, args:types.SimpleNamespace) -> None:
        ''' Initialize models, optimizer, etc... with config variables. '''

        self.datasets = {
            'train': SyntheticChangeDataset(
                data_path=args.train_dir,
                orientation_thresholds=args.orientation_thresholds, 
                parts_diff_thresholds=args.parts_diff_thresholds, 
                img_transforms=args.img_transforms
                ),
            'val': SyntheticChangeDataset(
                data_path=args.val_dir,
                orientation_thresholds=args.orientation_thresholds,
                parts_diff_thresholds=args.parts_diff_thresholds, 
                img_transforms=remove_train_augmentations(args.img_transforms)
                )
        }

        self.dataloaders = {x: DataLoader(self.datasets[x], 
                                batch_size=args.batch_size, shuffle=True, 
                                num_workers=args.num_workers, drop_last=True)
                        for x in ['train', 'val']}
        
        # initialize functions and network       
        self.net, self.device = build_model(args=args, gpu=args.gpu)
        self.optimizer = get_optimizer(self.net, args)
        self.lr_scheduler = get_scheduler(self.optimizer, args)
        self.loss_func = get_loss_func(args.loss)
        self.running_metric = ConfuseMatrixMeter(n_class=2)
        self.loss_plotter = LossPlotter(args.output_dir, args.resume_results_dir)
        # gives insight on the number of unique pairs, etc...
        self.stat_tracker = StatTracker(args.output_dir, "train", 
                                         args.batch_size)

        self._initialize_variables(args)
        
    def _initialize_variables(self, args:types.SimpleNamespace) -> None:
        ''' Initialize variables with config values & define logger'''
       
        # define logger file
        logger_path = os.path.join(args.output_dir, 'log.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)
        self.log_iter = args.log_iter
        self.img_save_freq = args.save_fig_iter        
        # define timer
        self.timer = Timer()
        self.batch_size = args.batch_size

        #  training log
        self.epoch_iou = 0
        self.best_val_iou = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 1
        self.max_num_epochs = args.max_epochs
     
        self.global_step = {'train': 0, 'val': 0}
        # Arbitrarily decided that 1 epoch = 20k images:
        self.steps_per_epoch = 20000//self.batch_size 
        self.total_steps = (self.max_num_epochs - 
                            self.epoch_to_start)*self.steps_per_epoch
        self.net_pred = None
        self.batch = None
        self.loss = None
        self.running_loss = None
        self.split = ""
        self.batch_id = 0
        self.val_steps_per_epoch = 100
        self.epoch_id = 0
        self.vis_dir = args.vis_dir
        self.save_ckpt = args.save_ckpt
        self.resume_ckpt = args.resume_ckpt_path
        self.norm_type = args.img_transforms['normalization']
        self.gradually_augment = args.img_transforms['gradually_augment']
        if self.gradually_augment:  
            self.augmentations = args.img_transforms
            self.num_workers = args.num_workers
        if self.save_ckpt: self.checkpoint_dir = args.checkpoint_dir

    def _load_checkpoint(self) -> None:
        ''' load the last checkpoint from previous training and update states. '''
        
        self.logger.write(f'loading checkpoint from {self.resume_ckpt}...\n')
        # load the entire checkpoint
        checkpoint = torch.load(os.path.join(self.resume_ckpt),
                                map_location=self.device)
        # update net states
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.to(self.device)
        
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.lr_scheduler.load_state_dict(
            # checkpoint['exp_lr_scheduler_state_dict'])

        # update some states
        self.epoch_to_start = checkpoint['epoch_id'] + 1
        self.best_val_iou = checkpoint['best_val_iou']
        self.best_epoch_id = checkpoint['best_epoch_id']
        
        # update max epochs to include pretrained epochs
        self.max_num_epochs = self.max_num_epochs + self.epoch_to_start - 1
        self.total_steps = (self.max_num_epochs - self.epoch_to_start)\
                                                * self.steps_per_epoch

        self.logger.write(f'Epoch_to_start = {self.epoch_to_start}, '
                          f'Historical_best_iou = {self.best_val_iou:.4f}'
                          f'(at epoch {self.best_epoch_id})\n\n')

    def _timer_update(self) -> None:
                
        self.global_step[self.split] += 1
        self.timer.update_progress((self.global_step['train'] + 1) 
                                   / self.total_steps)
        est = self.timer.estimated_remaining()
        imps = (self.global_step['train'] + 1) * self.batch_size / \
                                        self.timer.get_stage_elapsed()
        return imps, est

    def _visualize_label(self, gt:bool=True) -> Image.Image:
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
        vis_array = torch.zeros(n_images, 3, label_batch.shape[-2], 
                               label_batch.shape[-1]) # N x C x H x W
        # make the prediction overlap with the anchor image
        for i in range(n_images):
            label = v2.ToPILImage()(label_batch[i])
            ref = v2.ToPILImage()(ref_batch[i])
            blend = Image.blend(ref.convert('RGBA'), label.convert('RGBA'), 0.8)
            vis_array[i] = v2.functional.pil_to_tensor(blend.convert('RGB'))/255
        
        return vis_array

    def _save_checkpoint(self, ckpt_name:str):
        ''' Save latest (every epoch) or best checkpoint (when IoU is max) '''
        
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_iou': self.best_val_iou,
            'best_epoch_id': self.best_epoch_id,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'exp_lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
        }, os.path.join(self.checkpoint_dir, ckpt_name))        

    def _update_metric(self) -> np.ndarray:
        """ Update running average of scores (every iteration). """
        
        target = self.batch[2].to(self.device).detach().cpu().numpy().squeeze(1)
        net_pred = self.net_pred.detach()
        net_pred = torch.argmax(net_pred, dim=1).cpu().numpy()

        current_iou = self.running_metric.update_cm(net_pred, target) 
        return current_iou

    def _collect_running_batch_states(self) -> None:
        ''' Update states/scores and log text & images depending on iteration. '''
        
        running_iou = self._update_metric()
        imps, est = self._timer_update()
        if self.split == 'train': self.stat_tracker.update_stats(self.batch[3:])
        
        # Log text at right interval
        if np.mod(self.global_step[self.split], self.log_iter) == 0: 
            message = (f'[{self.timer.time_since_start_str()}]:\t {self.split} '
                f'- Epoch {self.epoch_id}/{self.max_num_epochs},\t Batch '
                f'{self.batch_id}/{self.steps_per_epoch},\t img/s: {imps * \
                self.batch_size:.2f},\t est: {est:.2f}h,\t loss: '
                f'{self.loss.item():.5f},\t running_iou: {running_iou:.5f}\n')
            self.logger.write(message)
        
        # save images at right interval
        if np.mod(self.global_step[self.split], self.img_save_freq) == 0: 
            
            anchor = denormalize(self.batch[0], self.norm_type)
            sample = denormalize(self.batch[1], self.norm_type)
            vis_anchor = make_numpy_grid(anchor)
            vis_sample = make_numpy_grid(sample)
            vis_pred = make_numpy_grid(self._visualize_label(gt=False))
            vis_gt = make_numpy_grid(self._visualize_label(gt=True))
            
            vis = np.concatenate([vis_anchor,vis_sample,vis_pred,vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            
            vis_split_dir = os.path.join(self.vis_dir, self.split)
            file_name = os.path.join(vis_split_dir, f'epoch{str(self.epoch_id)}'\
                                            f'_batch{str(self.batch_id)}.jpg')
            plt.imsave(file_name, vis)

    def _collect_epoch_states(self) -> None:
        ''' Save running scores at the end of each epoch '''
        
        scores = self.running_metric.get_scores()
        self.loss_plotter.plot_losses(self.split)  # update loss curve
        self.stat_tracker.save_stats()
        
        # take average of loss
        if self.split == 'train': 
            epoch_loss = self.running_loss / self.steps_per_epoch  
        else: epoch_loss = self.running_loss / self.val_steps_per_epoch
        
        self.loss_plotter.update_score_log(self.split, scores, epoch_loss, 
                            self.epoch_id, self.lr_scheduler.get_last_lr()[0])
        self.epoch_iou = scores['iou_1']
        self.logger.write(f'{self.split}: Epoch {self.epoch_id} / '
                f'{self.max_num_epochs-1}, epoch_IoU= {self.epoch_iou:.4f}\n')

        message = ''
        for k, v in scores.items():
            message += f'{k}: {v:.3e}  '
        self.logger.write(message+'\n')
        self.logger.write('\n')

    def _update_checkpoints(self) -> None:
        ''' Save the current model checkpoint. '''
        
        self._save_checkpoint(ckpt_name='last_ckpt.pt')
        self.logger.write(f'Lastest model updated. '
                f'Epoch_iou={self.epoch_iou:.4f}, Historical_best_iou='
                f'{self.best_val_iou:.4f} (at epoch {self.best_epoch_id})\n')
        self.logger.write('\n')

        # update the best model (based on eval iou)
        if self.epoch_iou > self.best_val_iou:
            self.best_val_iou = self.epoch_iou
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('*' * 10 + 'Best model updated!\n')
            self.logger.write('\n')
            
    def _increase_augmentation(self):
        ''' Gradually increase image augmentation each epoch. '''
        
        scaling_factor = self.epoch_id / self.max_num_epochs
        # scale augmentations according to epoch
        scaled_augs = self.augmentations.copy()
        for key in ['brightness', 'contrast', 'saturation', 'hue', 
                                            'g_sigma_h', 'g_sigma_l']:
            scaled_augs[key] = scaling_factor * self.augmentations[key]
        scaled_augs['shear'] = round(scaling_factor * self.augmentations['shear'])
        
        self.datasets['train'].img_transforms = scaled_augs  # update set
        # update data loader so that the changes come through
        self.dataloaders['train'] =  DataLoader(self.datasets['train'], 
                                batch_size=self.batch_size, shuffle=True, 
                                num_workers=self.num_workers, drop_last=True)
        
        return iter(self.dataloaders['train'])

    def _reset_running_scores(self):
        self.running_metric.reset()
        self.running_loss = 0

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch[0].to(self.device)
        img_in2 = batch[1].to(self.device)
        self.net_pred = self.net(img_in1, img_in2)

    def _get_loss(self):
        gt = self.batch[2].to(self.device).long()
        self.loss = self.loss_func(self.net_pred, gt)
        self.running_loss += self.loss.cpu().detach().numpy()
        
    def _backward_pass(self):
        self._get_loss()
        self.loss.backward()

    def train_model(self):

        if self.resume_ckpt != '':
            self._load_checkpoint()
        else:
            self.logger.write('training from scratch...')
            
        train_data_iter = iter(self.dataloaders['train'])  # initialize iter

        # loop over a set number of epochs
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs+1):
            
            ################## train #################
            ##########################################
            self._reset_running_scores()
            self.split = 'train'
            self.net.train()  # Set model to training mode
            if self.gradually_augment: 
                train_data_iter = self._increase_augmentation()
            self.logger.write('lr: %0.7f\n' % self.optimizer.param_groups[0]['lr'])
            self.logger.write(f'lr: {self.lr_scheduler.get_last_lr()[0]}\n')
            
            # Iterate over data
            for self.batch_id in range(0, self.steps_per_epoch):
                try:  # endlessly cycle through dataloader
                    batch = next(train_data_iter) 
                except StopIteration:  # StopIteration is thrown if dataset ends
                    self.stat_tracker.end_of_dataloader()  # reinitialize dataloader
                    train_data_iter = iter(self.dataloaders['train'])
                    batch = next(train_data_iter) 

                self._forward_pass(batch)
                self.optimizer.zero_grad()
                self._backward_pass()
                self.optimizer.step()
                self._collect_running_batch_states()
            
            self._collect_epoch_states()
            self.lr_scheduler.step()

            ################## Eval ##################
            ##########################################
            self.logger.write('Begin evaluation...\n')
            self._reset_running_scores()
            self.split = 'val'
            self.net.eval()

            # Iterate over data.
            for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):
                with torch.no_grad():
                    self._forward_pass(batch)
                    self._get_loss()
                self._collect_running_batch_states()
                if self.batch_id > self.val_steps_per_epoch: 
                    break  # to avoid looping for too long
            self._collect_epoch_states()
            
            ########### Update_Checkpoints ###########
            ##########################################
            if self.save_ckpt:
                self._update_checkpoints()