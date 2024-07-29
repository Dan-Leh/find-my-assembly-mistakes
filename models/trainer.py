import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

from models.networks import *

import torch
from torchvision.transforms import v2

from utils.metric_tool import ConfuseMatrixMeter
from utils.losses import cross_entropy, focal_loss
from misc.utils import make_numpy_grid, Loss_tracker
from misc.logger_tool import Logger, Timer
from utils.transforms import denormalize
from utils.data_stats import Stat_tracker
from models.tmp_functions import get_optimizer, get_scheduler, build_model, get_loss_func


class CDTrainer():

    def __init__(self, args, dataloaders):
        self.args = args
        self.dataloaders = dataloaders
        
        self.stat_tracker = Stat_tracker(args.output_dir, "train", 
                                         args.batch_size)
        self.model, self.device = build_model(args=args, gpu=args.gpu)
        self.optimizer = get_optimizer(self.model, args)
        self.lr_scheduler = get_scheduler(self.optimizer, args)
        self.loss_func = get_loss_func(args.loss)
        self.running_metric = ConfuseMatrixMeter(n_class=2)
        self.loss_tracker = Loss_tracker(args.output_dir, args.resume_results_dir)

        # define logger file
        logger_path = os.path.join(args.output_dir, 'log.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)
        self.log_iter = args.log_iter
        self.img_save_freq = args.save_fig_iter
        self.eval_freq = args.eval_freq
        # define timer
        self.timer = Timer()
        self.batch_size = args.batch_size

        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 1
        self.max_num_epochs = args.max_epochs
     
        self.global_step = 0
        self.steps_per_epoch = 20000//self.batch_size # I arbitrarily decided to make the number of steps per epoch such that 1 epoch = 20k images
        self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch
        
        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.loss = None
        self.running_loss = None
        self.is_training = False
        self.batch_id = 0
        self.val_global_step = 0 # for the saving of figures, to use the same save frequency in validation
        self.val_steps_per_epoch = 100
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.output_dir = args.output_dir
        self.vis_dir = args.vis_dir
        self.save_ckpt = args.save_ckpt
        self.resume_ckpt = args.resume_ckpt_path

        self.norm_type = args.data_augmentations['normalization']
        
        

        self.VAL_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.output_dir, 'val_acc.npy')):
            self.VAL_ACC = np.load(os.path.join(self.output_dir, 'val_acc.npy'))
        self.TRAIN_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.output_dir, 'train_acc.npy')):
            self.TRAIN_ACC = np.load(os.path.join(self.output_dir, 'train_acc.npy'))


    def _load_checkpoint(self, ckpt_name='last_ckpt.pt'):

        self.logger.write(f'loading checkpoint from {self.resume_ckpt}...\n')
        # load the entire checkpoint
        checkpoint = torch.load(os.path.join(self.resume_ckpt),
                                map_location=self.device)
        # update net_G states
        self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

        # self.optimizer.load_state_dict(checkpoint['optimizer_G_state_dict'])
        # self.lr_scheduler.load_state_dict(
            # checkpoint['exp_lr_scheduler_G_state_dict'])

        self.net_G.to(self.device)

        # update some other states
        self.epoch_to_start = checkpoint['epoch_id'] + 1
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_epoch_id = checkpoint['best_epoch_id']
        
        self.max_num_epochs = self.max_num_epochs + self.epoch_to_start - 1 # to include pretrained epochs
        self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

        self.logger.write('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)\n' %
                (self.epoch_to_start, self.best_val_acc, self.best_epoch_id))
        self.logger.write('\n')

    
    



    def _timer_update(self):
        self.global_step = (self.epoch_id-self.epoch_to_start) * self.steps_per_epoch + self.batch_id

        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        est = self.timer.estimated_remaining()
        imps = (self.global_step + 1) * self.batch_size / self.timer.get_stage_elapsed()
        return imps, est


    def _visualize_label(self, gt=True): # if gt, load the ground truth label, else the predicted

        if gt == True:
            label_batch = self.batch[2].to(torch.uint8) * 255
        else: 
            label_batch = torch.argmax(self.G_pred, dim=1, keepdim=False)
            label_batch = label_batch.to(torch.uint8) * 255
        
        ref_batch = denormalize(self.batch[0], self.norm_type)
        
        n_images = self.batch_size if self.batch_size<8 else 8
        pred_vis = torch.zeros(n_images, 3, label_batch.shape[-2], label_batch.shape[-1]) # initialize empty tensor
        # make the prediction overlap with the reference image
        for i in range(n_images):
            label = v2.ToPILImage()(label_batch[i])
            ref = v2.ToPILImage()(ref_batch[i])
            blend = Image.blend(ref.convert('RGBA'), label.convert('RGBA'), 0.8)
            pred_vis[i] = v2.functional.pil_to_tensor(blend.convert('RGB'))/255
        
        return pred_vis

    def _save_checkpoint(self, ckpt_name):
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_acc': self.best_val_acc,
            'best_epoch_id': self.best_epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.lr_scheduler.state_dict(),
        }, os.path.join(self.checkpoint_dir, ckpt_name))

    def _update_lr_schedulers(self):
        self.lr_scheduler.step()

    def _update_metric(self):
        """
        updates running average
        """
        target = self.batch[2].to(self.device).detach().cpu().numpy().squeeze(1)
        G_pred = self.G_pred.detach()

        G_pred = torch.argmax(G_pred, dim=1).cpu().numpy()

        current_score = self.running_metric.update_cm(pr=G_pred, gt=target) 
        return current_score

    def _collect_running_batch_states(self):
        
        running_acc = self._update_metric()
        if self.is_training:
            self.split = 'train' 
            log_txt = True if np.mod(self.global_step, self.log_iter) == 0 else False
            log_imgs = True if np.mod(self.global_step, self.img_save_freq) == 0 else False
            
        else:
            self.split = 'val'
            log_txt = True if np.mod(self.val_global_step, self.log_iter) == 0 else False
            log_imgs = True if np.mod(self.val_global_step, self.img_save_freq) == 0 else False

        imps, est = self._timer_update()
        if log_txt:
            message = f'[{self.timer.time_since_start_str()}]: \
                {self.split} - Epoch {self.epoch_id}/{self.max_num_epochs}, \
                Batch {self.batch_id}/{self.steps_per_epoch}, img/s: {imps*self.batch_size:.2f}, est: {est:.2f}h, \
                G_loss: {self.loss.item():.5f}, running_mf1: {running_acc:.5f}\n'
            self.logger.write(message)

        if log_imgs:
            vis_input = make_numpy_grid(denormalize(self.batch[0], self.norm_type))
            vis_input2 = make_numpy_grid(denormalize(self.batch[1], self.norm_type))

            vis_pred = make_numpy_grid(self._visualize_label(gt=False))

            vis_gt = make_numpy_grid(self._visualize_label(gt=True))
            vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            
            vis_split_dir = os.path.join(self.vis_dir, self.split)
            if not os.path.exists(vis_split_dir): os.mkdir(vis_split_dir)
            file_name = os.path.join(
                vis_split_dir, f'epoch{str(self.epoch_id)}_batch{str(self.batch_id)}.jpg')
            plt.imsave(file_name, vis)

    def _collect_epoch_states(self):
        scores = self.running_metric.get_scores()
        epoch_loss = self.running_loss / self.steps_per_epoch if self.is_training else self.running_loss / self.val_steps_per_epoch
        self.loss_tracker.update_score_log(self.split, scores, epoch_loss, self.epoch_id, self.lr_scheduler.get_last_lr()[0])
        self.epoch_acc = scores['mf1']
        self.logger.write('%s: Epoch %d / %d, epoch_mF1= %.5f\n' %
              (self.split, self.epoch_id, self.max_num_epochs-1, self.epoch_acc))
        message = ''
        for k, v in scores.items():
            message += f'{k}: {v:.3e} '
        self.logger.write(message+'\n')
        self.logger.write('\n')

    def _update_checkpoints(self):

        # save current model
        self._save_checkpoint(ckpt_name='last_ckpt.pt')
        self.logger.write('Lastest model updated. Epoch_acc=%.4f, Historical_best_acc=%.4f (at epoch %d)\n'
              % (self.epoch_acc, self.best_val_acc, self.best_epoch_id))
        self.logger.write('\n')

        # update the best model (based on eval acc)
        if self.epoch_acc > self.best_val_acc:
            self.best_val_acc = self.epoch_acc
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('*' * 10 + 'Best model updated!\n')
            self.logger.write('\n')

    def _update_training_acc_curve(self):
        # update train acc curve
        self.TRAIN_ACC = np.append(self.TRAIN_ACC, [self.epoch_acc])
        # np.save(os.path.join(self.output_dir, 'train_acc.npy'), self.TRAIN_ACC)
        self.loss_tracker.save_losses('train')
        
    def _update_val_acc_curve(self):
        # update val acc curve
        self.VAL_ACC = np.append(self.VAL_ACC, [self.epoch_acc])
        # np.save(os.path.join(self.output_dir, 'val_acc.npy'), self.VAL_ACC)
        self.loss_tracker.save_losses('val')

    def _clear_cache(self):
        self.running_metric.clear()
        self.running_loss = 0


    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch[0].to(self.device)
        img_in2 = batch[1].to(self.device)
        self.G_pred = self.net_G(img_in1, img_in2)

    def get_loss(self):
        gt = self.batch[2].to(self.device).long()
        self.loss = self.loss_func(self.G_pred, gt)
        self.running_loss += self.loss.cpu().detach().numpy()
        
        
    def _backward_G(self):
        self.get_loss()
        self.loss.backward()


    def train_models(self):

        if self.resume_ckpt != '':
            self._load_checkpoint()
        else:
            print('training from scratch...')
            
        train_data_iter = iter(self.dataloaders['train'])

        # loop over the dataset multiple times
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs+1):

            ################## train #################
            ##########################################
            self._clear_cache()
            self.is_training = True
            self.net_G.train()  # Set model to training mode
            # Iterate over data
            self.logger.write('lr: %0.7f\n' % self.optimizer.param_groups[0]['lr'])
            self.logger.write(f'lr: {self.lr_scheduler.get_last_lr()[0]}\n')
            
            for self.batch_id in range(0, self.steps_per_epoch):
                try:
                    batch = next(train_data_iter) 
                except StopIteration: # StopIteration is thrown if dataset ends
                    # reinitialize data loader 
                    self.stat_tracker.end_of_dataloader()
                    train_data_iter = iter(self.dataloaders['train'])
                    batch = next(train_data_iter) 

                self._forward_pass(batch)
                # update G
                self.optimizer.zero_grad()
                self._backward_G()
                self.optimizer.step()
                self._collect_running_batch_states()
                self.stat_tracker.update_stats(self.batch[3:])
            
            self._collect_epoch_states()
            self._update_training_acc_curve()
            self._update_lr_schedulers()
            self.stat_tracker.save_stats()
                
            ################## Eval ##################
            ##########################################
            self.logger.write('Begin evaluation...\n')
            self._clear_cache()
            self.is_training = False
            self.net_G.eval()

            # Iterate over data.
            for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):
                self.val_global_step += 1 
                with torch.no_grad():
                    self._forward_pass(batch)
                    self.get_loss()
                self._collect_running_batch_states()
                if self.batch_id > self.val_steps_per_epoch: # temporary fix for large validation set, just run first 100 batches
                    break
            self._collect_epoch_states()
            
            ########### Update_Checkpoints ###########
            ##########################################
            self._update_val_acc_curve()
            if self.save_ckpt:
                self._update_checkpoints()