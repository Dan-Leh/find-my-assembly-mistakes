import os
from torch.utils.data import DataLoader

from datasets.synthtestset import EvalDataset
from config import read_config
from models.evaluator import CDEvaluator

cfg = read_config(train=False)

# which test sets the model should use: Anytime cfg.test_sets contains any or 
# all names, they are all 'tested'
eval_set_list = []

    
data_root = "/shared/nl011006/res_ds_ml_restricted/dlehman/SyntheticData"
    
# instantiate dataset
CDD_test = EvalDataset(data_list_filepath=cfg.test_dir,
                        img_size=cfg.img_transforms['img_size'],
                        norm_type = cfg.img_transforms['normalization'],
                        test_type = 'roi')
dataloader = DataLoader(CDD_test, batch_size=1, shuffle=False, 
                        num_workers=cfg.num_workers, drop_last=True)

# create output directory
output_dir = os.path.join(cfg.output_dir)
os.makedirs(output_dir)
cfg.vis_dir = os.path.join(output_dir, "visualize")
os.makedirs(cfg.vis_dir)

# run evaluation
Evaluator = CDEvaluator(args=cfg, dataloader=dataloader, 
                test_type='roi', output_dir = output_dir)
Evaluator.eval_models()
