import os
from torch.utils.data import DataLoader
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

from datasets.decode_test_folder import decode_test_sets
from datasets.synthtestset import EvalDataset
from config import read_config
from models.evaluator import CDEvaluator

cfg = read_config(train=False)

# which test sets the model should use: Anytime cfg.test_sets contains any or 
# all names, they are all 'tested'
test_data_configuration = decode_test_sets(cfg.test_sets, 
                                           cfg.img_transforms["ROI_crops"],
                                           cfg.img_transforms["center_roi"])

set_name = test_data_configuration['set_name']
test_type = test_data_configuration['test_type']
data_path = test_data_configuration['data_path']
more_nqd_bins = test_data_configuration['more_nqd_bins']

for i in range(len(set_name)):
    
    # instantiate dataset
    CDD_test = EvalDataset(data_list_filepath=data_path[i],
                            img_size=cfg.img_transforms['img_size'],
                            norm_type = cfg.img_transforms['normalization'],
                            test_type = test_type[i].lower(),
                            more_nqd_bins = more_nqd_bins[i])
    dataloader = DataLoader(CDD_test, batch_size=1, shuffle=False, 
                            num_workers=cfg.num_workers, drop_last=True)

    # create output directory
    output_dir = os.path.join(cfg.output_dir, set_name[i], test_type[i])
    os.makedirs(output_dir)
    cfg.vis_dir = os.path.join(output_dir, "visualize")
    os.mkdir(cfg.vis_dir)

    # run evaluation
    Evaluator = CDEvaluator(args=cfg, dataloader=dataloader, 
                    test_type=test_type[i].lower(), output_dir = output_dir)
    Evaluator.eval_models()
