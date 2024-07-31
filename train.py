import torch; torch.manual_seed(0)
from torch.utils.data import DataLoader

from datasets.synthdataset import SyntheticChangeDataset
from config import read_config 
from models.trainer import CDTrainer


cfg = read_config(train=True) # get all variables from config file

def remove_train_augmentations(tf_cfg: dict) -> dict:
    ''' Remove train-only augmentations from validation set. '''
    val_transforms = tf_cfg.copy() 
    for aug in ['hflip_probability', 'vflip_probability', 'brightness', 
                'contrast', 'saturation', 'hue', 'shear']:
        val_transforms[aug] = 0
    for aug in ['g_kernel_size', 'g_sigma_l', 'g_sigma_h']:
        val_transforms[aug] = 1
    for aug in ['rotation']:
        val_transforms[aug] = False
        
    return val_transforms

datasets = {
    'train': SyntheticChangeDataset(
        data_path=cfg.train_dir,
        orientation_thresholds=cfg.orientation_thresholds, 
        parts_diff_thresholds=cfg.parts_diff_thresholds, 
        img_transforms=cfg.img_transforms
        ),
    'val': SyntheticChangeDataset(
        data_path=cfg.val_dir,
        orientation_thresholds=cfg.orientation_thresholds,
        parts_diff_thresholds=cfg.parts_diff_thresholds, 
        img_transforms=remove_train_augmentations(cfg.img_transforms)
        )
}

dataloaders = {x: DataLoader(datasets[x], batch_size=cfg.batch_size, 
                             shuffle=True, num_workers=cfg.num_workers, 
                             drop_last=True)
                for x in ['train', 'val']}

Trainer = CDTrainer(args=cfg, dataloaders=dataloaders)
Trainer.train_model()