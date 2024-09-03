import json
import os
import argparse
import yaml
import types
import warnings

def read_config(train:bool = True) -> types.SimpleNamespace:
    ''' Read config from a yaml file.
    
    This function overwrites any arguments from the config that are passed
    as flags in the command line or through a bash script and else uses the
    default values for each parameter from the yaml config file.
    Arguments:
        train (bool): if true, config is used for training, else for testing
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
            default='/shared/nl011006/res_ds_ml_restricted/dlehman'\
                '/state-diff-net/config_files/much_aug.yaml')
        
    # gather all the values that were given through command line   
    overwrite_parser = get_overwrite_arguments(parser, train)
    cmd_args = overwrite_parser.parse_args()
    
    # open config file 
    with open(cmd_args.config, 'r') as configfile:
        config = yaml.load(configfile, Loader=yaml.FullLoader)
        configfile.close()
    
    delattr(cmd_args, 'config')  # delete config path so that it is not saved as arg
    config = update_config(config, cmd_args)  # overwrite args based on flags
    config = filter_config(config, cmd_args, 'train' if train else 'test')

    # create directories and save config
    if train:
        config = make_train_dirs(config)      
    else:  # test
        config = make_test_dirs(config)
    save_config(config)  
    
    # for ease of use, turn directory to
    config_ns = types.SimpleNamespace(**config) 
    return config_ns


def update_config(config:dict, new_args:argparse.Namespace) -> dict:
    ''' Function that replaces the arguments from the config file with the ones 
    passed by command line.
    
    Args:
        config (dict): a dictionary of all the config arguments from yaml file
        new_args (argparse.Namespace): All of the arguments passed through argparse
    Returns: 
        config (dict): dictionary where values of specified config arguments 
            are updated
    '''
    for arg, value in new_args.__dict__.items():
        if value!=None: # if a value was passed through the command line
            check_exists(config, arg)  # to avoid accidentally adding argument
            # determine whether the parameter is child of another parameter
            parts = arg.split('/') 
            if len(parts) > 1: # if it is a nested value in the config file
                config[parts[0]][parts[1]] = value
            else:  # update value
                config[arg] = value
        else:
            check_exists(config, arg)
    return config


def check_exists(config_dict:dict, arg:str) -> None:
    ''' Checks that the args passed by command line exist in the config file'''
    
    if '/' not in arg: # if it is a singe argument
        error = arg not in config_dict.keys() 
    else: # if argument is composed of parent and child, check that child exists
        error = arg.split('/')[1] not in config_dict[arg.split('/')[0]].keys()
    if error:
        warnings.warn(f"The given argument \"{arg}\" is not in the config file")
    
    
def filter_config(config_dict:dict, cmd_args:argparse.Namespace, split:str) -> dict:
    ''' Get rid of arguments that are not relevant for train or test split. 
    
    Delete from config the arguments that are only relevant for training when
    testing and delete arguments that are only relevant for testing while training,
    by removing any argument that does not have a corresponding argparse. 
    '''    
    for key in list(config_dict.keys()):  # check all keys in the config file
        if type(config_dict[key]) == dict:   # if this it is a subconfig
            for child_key in list(config_dict[key].keys()):
                arg = key+'/'+child_key
                if arg not in vars(cmd_args).keys():
                    del config_dict[key][child_key]
                    print(f"Delete '{arg}' from config as it is not used for "
                                                                    f"{split}ing")
        elif key not in vars(cmd_args).keys():
            del config_dict[key]
            print(f"Delete '{key}' from config as it is not used for {split}ing")
    
    return config_dict


def save_config(config_dict:dict) -> None: 
    ''' Save yaml file with config parameters in output directory. '''
    
    config_path = os.path.join(config_dict['output_dir'], "config.yaml")

    config_copy = config_dict.copy()
    # delete arguments that do not need saving
    del config_copy['vis_dir']
    del config_copy['output_dir']
    
    # save config
    with open(config_path, "w") as configfile:
        yaml.dump(config_dict, configfile)
        
        
def make_train_dirs(config_dict:dict) -> dict:
    ''' Make directories for saving checkpoints, training log and examples. 
    
    If the experiment name already exists in the output directory, the name is
    changed by adding a number to the end. 
    ''' 
    # output directory
    output_path = os.path.join(config_dict['output_root'], 
                               config_dict['experiment_name'])
    if os.path.isdir(output_path):  # if the directory already exists
        orig_path = output_path; i=0
        while os.path.isdir(output_path):  # as long as name exists
            i+=1
            output_path = orig_path + '('+str(i)+')'  # add number to name
        config_dict['experiment_name'] = config_dict['experiment_name'] + \
                                    '('+str(i)+')' # rename experiment_name
    os.makedirs(output_path)
    config_dict['output_dir'] = os.path.join(config_dict['output_root'], 
                                             config_dict['experiment_name'])
    
    # checkpoint directory    
    if config_dict['save_ckpt']:
        config_dict['checkpoint_dir'] = os.path.join(
                config_dict['checkpoint_root'], config_dict['experiment_name'])
        os.makedirs(config_dict['checkpoint_dir'], exist_ok=True)
    
    # visualization directory
    config_dict['vis_dir'] = os.path.join(config_dict['output_dir'], 'visualize')
    os.mkdir(config_dict['vis_dir'])
    os.mkdir(os.path.join(config_dict['vis_dir'], 'train'))
    os.mkdir(os.path.join(config_dict['vis_dir'], 'val'))

    return config_dict

    
def make_test_dirs(config_dict: dict): 
    ''' Add arguments for testing and build directories '''
    
    config_dict['output_dir'] = os.path.join(config_dict['output_root'], 
                                config_dict['experiment_name'], f"Test")
    # add number to output dir if it already exists
    if os.path.isdir(config_dict['output_dir']):
        orig_path = config_dict['output_dir']
        i=1
        output_path = orig_path + '('+str(i)+')'
        while os.path.isdir(output_path):
            i+=1
            output_path = orig_path + '('+str(i)+')'
        config_dict['output_dir'] = output_path
    os.makedirs(config_dict['output_dir'])
    
    if 'checkpoint_dir' not in config_dict.keys():
        config_dict['checkpoint_dir'] = os.path.join(
            config_dict['checkpoint_root'], config_dict['experiment_name'])
    
    config_dict['vis_dir'] = os.path.join(config_dict['output_dir'], 'visualize')
    
    return config_dict


def get_overwrite_arguments(parser, train:bool): 

    parser.add_argument('--experiment_name', type=str, help="Name used for creating checkpoint and 'results' directory")

    if train:  # these config arguments are only relevant during training
        parser.add_argument('--train_dir', type=str, help="The directory containing all the training data")
        parser.add_argument('--val_dir', type=str, help="The directory containing all the validation data")
        parser.add_argument('--checkpoint_root', type=str, help="Checkpoint dir will be created by adding folder with experiment name")

        parser.add_argument('--resume_ckpt_path', type=str, help="If not an empty string, training is resumed with weights loaded from the given checkpoint path")
        parser.add_argument('--resume_results_dir', type=str, help="The directory of the results of the model from which we are resuming training, used so that the resumed training plots its loss curve in the same image")

        # data details
        parser.add_argument('--orientation_thresholds', nargs=2, type=float, help="The minimum and maximum norm of quaternion difference between anchor and sample images")
        parser.add_argument('--parts_diff_thresholds', nargs=2, type=int, help="The minimum and maximum amount of parts that should differ between anchor and sample images (amount of change)")

        # training hyperparams
        parser.add_argument('--batch_size', type=int)
        parser.add_argument('--loss', type=str, help="Can be 'ce' (cross-entropy loss) or 'focal' (focal loss)")
        parser.add_argument('--optimizer', type=str, help="Options are 'sgd' or 'adam'")
        parser.add_argument('--lr_policy', type=str, help="Options include 'cosine', 'linear', 'constant' & 'step'")
        parser.add_argument('--lr', type=float, help="Maximum learning rate")
        parser.add_argument('--T_0', type=int, help='Number of epochs for first period of cosine lr scheduler')
        parser.add_argument('--T_mult', type=int, help='Multiplication factor for subsequent periods of cosine lr scheduler')
        parser.add_argument('--warmup_epochs', type=int, help='Number of epochs during which learning rate linearly increases from 0 to maximum lr')
        parser.add_argument('--max_epochs', type=int, help='Number of epochs to train for')
        parser.add_argument('--save_ckpt', type=str2bool, help='Whether to save a checkpoint. Useful for testing code without creating checkpoint files.')    
        parser.add_argument('--save_ckpt_interval', type=int, help='At what interval of epochs to save a checkpoint?')
        parser.add_argument('--init_type', type=str, help="What type of weight initialization to use for parameters that are not loaded from pretrained weights. Options include 'normal', 'xavier', 'kaiming' and 'orthogonal'")
        # geometric transformations
        parser.add_argument('--img_transforms/rotation', type=str2bool, help="Whether to rotate the image randomly, either 0, 90, 180 or 270 degrees.")
        parser.add_argument('--img_transforms/shear', type=int, help='Max amount of random shearing to apply independently to each image.')
        parser.add_argument('--img_transforms/hflip_probability', type=float, help="Probability of random vertical flipping, same flipping applied to all images in the same pair")
        parser.add_argument('--img_transforms/vflip_probability', type=float, help="Probability of random horizontal flipping, same flipping applied to all images in the same pair")
        # photometric augmentations
        parser.add_argument('--img_transforms/brightness', type=float)
        parser.add_argument('--img_transforms/contrast', type=float)
        parser.add_argument('--img_transforms/saturation', type=float)
        parser.add_argument('--img_transforms/hue', type=float)
        parser.add_argument('--img_transforms/g_kernel_size', type=float, help="Kernel size of gaussian blur")
        parser.add_argument('--img_transforms/g_sigma_h', type=float, help="Maximum standard deviation that can be chosen for blurring kernel.")
        parser.add_argument('--img_transforms/g_sigma_l', type=float, help="Minimum standard deviation that can be chosen for blurring kernel.")
        parser.add_argument('--img_transforms/gradually_augment', type=str2bool, help="If true, start with little shear and color-based augmentation (brightness, contrast, saturation, hue, g_sigma_h) & gradually increase values proportionally to epoch till max epoch is reached. If false, keep values constant.")

        # Forier Domain Adaptation
        parser.add_argument('--fda/beta_min', type=float, help="The minimum 'width' of low frequency information to copy from the fft of a target domain image onto the source domain image, expressed as a fraction of the total image height/width.")
        parser.add_argument('--fda/beta_max', type=float, help="The maximum 'width' of low frequency information to copy from the fft of a target domain image onto the source domain image, expressed as a fraction of the total image height/width.")
        parser.add_argument('--fda/frac_imgs_w_fda', type=float, help="The fraction of images that should have any kind of Fourier Domain Adaptation (to allow for training on mix of source and target domains). 0 means no fda, 1 means fda on every image.")
        
    else: # test
        parser.add_argument('--checkpoint_dir', type=str, help="Only used during testing, to point to directory containing the model checkpoints")
        parser.add_argument('--test_sets', type=str, help="Names of the test sets to evaluate model on.")

    # misc
    parser.add_argument('--output_root', type=str, help="The directory where all experiment results are stored. A folder with the experiment name will be created in this directory")
    parser.add_argument('--gpu', type=str2bool, help='Uses GPU if true, else CPU')
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--save_fig_iter', type=int, help='After how many iterations should current training/validation examples be saves')
    parser.add_argument('--log_iter', type=int, help='How frequently to write the training/validation progress status into the log file.')
    
    # geometric transformations
    parser.add_argument('--img_transforms/img_size', nargs=2, type=int, help="Size of the input images to the network, regardless of prior cropping, etc...")
    parser.add_argument('--img_transforms/ROI_crops', type=str2bool, help="Whether to create region of interest crops of the assembly object")
    parser.add_argument('--img_transforms/center_roi', type=str2bool, help='Whether or not to have roi crops perfectly centered i.e. to create perfectly aligned image pairs.')
    parser.add_argument('--img_transforms/random_crop', type=str2bool, help="Whether to create random crops of anchor and sample images wherein the whole assembly object is still visible.")
    parser.add_argument('--img_transforms/max_translation', type=float, help="Maximum amount of random translation between the anchor and sample images, indicated as a fraction of the image size, i.e. 10% translation should be 0.1.")
    parser.add_argument('--img_transforms/rescale', type=float, help="The maximum scale ratio between anchor and sample image. Default should be 1 if images should be the same scale")
    parser.add_argument('--img_transforms/normalization', type=str, help="According to which weights to normalize input images. Currently only supports 'imagenet'.")
    parser.add_argument('--img_transforms/frac_random_background', type=float, help="The percentage of images that should contain a randomized background consisting of randomly chosen COCO images")
    
    # model architecture parameters
    parser.add_argument('--cyws/encoder', type=str, help="Which encoder to use. Options are resnet18, resnet34 and resnet50")
    parser.add_argument('--cyws/attention', type=str, help="Either 'gca' for global cross-attention, 'lca' for local cross-attention, 'msa' for multi-headed self-attention in addition to gca, or 'noam' for no attention module, i.e. simple concatenation")
    parser.add_argument('--cyws/coam_layer_data', type=json.loads, help="List with the following: [number of layers at which to use cross-attention module, [list of the channel dimensions at each of the attention modules (starting at bottleneck)],[list of channel dimensions to use in attention layers]]")
    parser.add_argument('--cyws/decoder_attn_type', type=str, help="Should be 'scse' for squeeze and excitation block, else None")
    parser.add_argument('--cyws/self_attention', type=str, help="Only relevant if attention='msa': Type of self-attention to use if attention type is 'msa', options are: 'linear' or 'full'")
    parser.add_argument('--cyws/n_MSA_layers', type=int, help="Only relevant if attention='msa': Number of self-attention layers to use before cross-attention")
    parser.add_argument('--cyws/n_SA_heads', type=int, help="Only relevant if attention='msa': Number of heads in the self-attention")
    parser.add_argument('--cyws/kernel_sizes', type=int, nargs=3, help="Only relevant if attention='lca': Receptive field on the sample image across which to take local self-attention, at each of the resolutions indicated in 'coam_layer_data'.")
    parser.add_argument('--cyws/pretrained_encoder', type=str2bool, help="Whether to load imagenet pretrained weights")

    return parser


def str2bool(v):
    ''' Convert string to boolean (alternative to store_true in argparse).'''
    
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')