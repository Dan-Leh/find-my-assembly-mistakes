''' Evaluate model on sets that are less 'clean' than regular test sets '''

import os
from torch.utils.data import DataLoader
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

from datasets.synthtestset import EvalDataset
from config import read_config
from models.evaluator import CDEvaluator

cfg = read_config(train=False)

# which test sets the model should use: Anytime cfg.test_sets contains any or 
# all names, they are all 'tested'
eval_set_list = []
if 'rand_background' in cfg.test_sets:
    eval_set_list.append('rand_background')

    
data_root = "/shared/nl011006/res_ds_ml_restricted/dlehman/SyntheticData"
set2dir_name = {"rand_background": "Val_states_v2_ROI_rand_bg"}

dirty_img_runs = ['anchor', 'sample', 'both']

clean_img_path = ("/shared/nl011006/res_ds_ml_restricted/dlehman/"
                  "SyntheticData/Test_pairs/Val_states_v2_ROI")


# Run evaluation script for all the specified test sets
for eval_set in eval_set_list:
    for dirty_img in dirty_img_runs:
        # Run evaluation script for every aspect/type that is tested for (eg. nQD)
        
        test_name = 'Orientation_' + dirty_img + '_dirty'
        if eval_set == 'rand_background':
            data_list_filepath = os.path.join(data_root, "Test_pairs", 
                                set2dir_name[eval_set], "pair_info.json")
 
        print(f'Output directory name: {test_name}')
        
        # instantiate dataset
        CDD_test = EvalDataset(data_list_filepath=data_list_filepath,
                                img_size=cfg.img_transforms['img_size'],
                                norm_type=cfg.img_transforms['normalization'],
                                test_type='roi',
                                path_to_clean_imgs=clean_img_path,
                                dirty_img=dirty_img)
        dataloader = DataLoader(CDD_test, batch_size=1, shuffle=False, 
                                num_workers=cfg.num_workers, drop_last=True)

        # create output directory
        output_dir = os.path.join(cfg.output_dir, eval_set, test_name)
        os.makedirs(output_dir)
        cfg.vis_dir = os.path.join(output_dir, "visualize")
        os.makedirs(cfg.vis_dir)

        # run evaluation
        Evaluator = CDEvaluator(args=cfg, dataloader=dataloader, 
                        test_type='roi', output_dir = output_dir)
        Evaluator.eval_models()
