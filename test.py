import os
from torch.utils.data import DataLoader

from datasets.synthtestset import EvalDataset
from models.evaluator import CDEvaluator
from utils.manage_config import read_config

cfg = read_config(train=False)

def make_datalist_path(setname):
    if setname == "Random_background":
        setname = "Main_test_set"
    path = os.path.join(cfg.data_root, setname, 'test_img_pairs.json')
    return path
    
datalist_path = make_datalist_path(cfg.test_set_name)
random_background = True if cfg.test_set_name == "Random_background" else False
        
# instantiate dataset
CDD_test = EvalDataset(data_list_filepath=datalist_path,
                        img_size=cfg.img_transforms['img_size'],
                        norm_type = cfg.img_transforms['normalization'],
                        random_background = random_background,
                        bg_img_root=cfg.bg_img_root)
dataloader = DataLoader(CDD_test, batch_size=1, shuffle=False, 
                        num_workers=cfg.num_workers, drop_last=True)

# create output directory
os.mkdir(cfg.vis_dir)

# run evaluation
Evaluator = CDEvaluator(args=cfg, dataloader=dataloader, 
                test_type='orientation', output_dir=cfg.output_dir)
Evaluator.eval_models()