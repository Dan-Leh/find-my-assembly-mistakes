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
if 'v1' in cfg.test_sets:
    eval_set_list.append('Val_states')
if 'v2' in cfg.test_sets:
    eval_set_list.append('Val_states_v2')
if 'pp' in cfg.test_sets:
    eval_set_list.append('PseudoPlane')
if 'extra' in cfg.test_sets:
    eval_set_list.append('Val_states_extra')
if 'inter' in cfg.test_sets:
    eval_set_list.append('Val_states_inter')
    
data_root = "/shared/nl011006/res_ds_ml_restricted/dlehman/SyntheticData"
set2dir_name = {"Val_states": "Val_states_set_200x1000", 
                "PseudoPlane": "PseudoPlane_200x1000",
                "Val_states_v2": "Val_states_v2",
                "Val_states_extra": "Val_states_extra",
                "Val_states_inter": "Val_states_inter"}

# whether to test on ROI crops or full images
if cfg.img_transforms['ROI_crops']:
    test_type_list = ['ROI', 'ROI_missing', 'ROI_present']
    convert_test_type = {'ROI': 'Orientation', 'ROI_missing': 'Missing', 
                         'ROI_present': 'Present'}
else:
    test_type_list = ['Orientation', 'Scale', 'Translation', 
                      'Missing', 'Present']

# Run evaluation script for all the specified test sets
for eval_set in eval_set_list:
    # Run evaluation script for every aspect/type that is tested for (eg. nQD)
    for test_type in test_type_list: 
        vis_dir_test_type = test_type  # name of directory to save images in
        ds_test_type = test_type.lower()
        
        if test_type == 'Orientation':
            data_list_filepath = os.path.join(data_root, set2dir_name[eval_set], 
                                              "eval_rand_20k.json")
        elif test_type == 'Translation' or test_type == 'Scale':
            data_list_filepath = os.path.join(data_root, "Test_pairs", 
                                    eval_set+"_"+test_type, "pair_info.json")
        elif test_type == 'Missing':
            data_list_filepath = os.path.join(data_root, set2dir_name[eval_set], 
                                              "eval_missing_parts_2k.json")
        elif test_type =='Present':
            data_list_filepath = os.path.join(data_root, set2dir_name[eval_set], 
                                              "eval_present_parts_2k.json")
        elif test_type.startswith('ROI'):
            vis_dir_test_type = convert_test_type[test_type]
            if cfg.img_transforms['center_roi']:  # if the roi crops be aligned
                vis_dir_test_type += "_roi_aligned"
                test_type = 'ROI_aligned'
                if test_type == 'ROI_missing':
                    data_list_filepath = os.path.join(data_root, 
                            set2dir_name[eval_set], "eval_missing_parts_4k.json")
                elif test_type == 'ROI_present':
                    data_list_filepath = os.path.join(data_root, 
                            set2dir_name[eval_set], "eval_present_parts_4k.json")
                else:
                    data_list_filepath = os.path.join(data_root, 
                            set2dir_name[eval_set], "eval_rand_20k.json")
            else: # if the crops can have some small translations
                data_list_filepath = os.path.join(data_root, "Test_pairs", 
                                    eval_set+"_"+test_type, "pair_info.json")
                
        print(f'vis dir name: {vis_dir_test_type}')
       
        # instantiate dataset
        CDD_test = EvalDataset(data_list_filepath=data_list_filepath,
                               img_size=cfg.img_transforms['img_size'],
                               norm_type = cfg.img_transforms['normalization'],
                               test_type = ds_test_type)
        dataloader = DataLoader(CDD_test, batch_size=1, shuffle=False, 
                                num_workers=cfg.num_workers, drop_last=True)

        # create output directory
        output_dir = os.path.join(cfg.output_dir, eval_set, vis_dir_test_type)
        os.makedirs(output_dir)
        cfg.vis_dir = os.path.join(output_dir, "visualize")
        os.makedirs(cfg.vis_dir)

        # run evaluation
        Evaluator = CDEvaluator(args=cfg, dataloader=dataloader, 
                        test_type=ds_test_type, output_dir = output_dir)
        Evaluator.eval_models()
