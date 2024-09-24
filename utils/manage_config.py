import os
import argparse
import yaml
import types
import warnings

from config_args import get_overwrite_arguments

def read_config(train:bool = True) -> types.SimpleNamespace:
    ''' Read config from a yaml file.
    
    This function overwrites any arguments from the config that are passed
    as flags in the command line or through a bash script and else uses the
    default values for each parameter from the yaml config file.
    Arguments:
        train (bool): if true, config is used for training, else for testing
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
        
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
        yaml.dump(config_copy, configfile)
        
        
def make_train_dirs(config_dict:dict) -> dict:
    ''' Make directories for saving checkpoints, training log and examples. 
    
    If the experiment name already exists in the output directory, the name is
    changed by adding a number to the end. 
    '''
    # checkpoint directory    
    if config_dict['save_ckpt']:
        config_dict['checkpoint_dir'] = os.path.join(
                config_dict['checkpoint_root'], config_dict['experiment_name'])
        os.makedirs(config_dict['checkpoint_dir'], exist_ok=True)
        
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
    
    # visualization directory
    config_dict['vis_dir'] = os.path.join(config_dict['output_dir'], 'visualize')
    os.mkdir(config_dict['vis_dir'])
    os.mkdir(os.path.join(config_dict['vis_dir'], 'train'))
    os.mkdir(os.path.join(config_dict['vis_dir'], 'val'))
    
    return config_dict

    
def make_test_dirs(config_dict: dict): 
    ''' Add arguments for testing and build directories '''
    
    config_dict['output_dir'] = os.path.join(config_dict['output_root'], 
                                config_dict['experiment_name'], f"Test", 
                                config_dict['test_set_name'])
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
    
    config_dict['vis_dir'] = os.path.join(config_dict['output_dir'], "visualize")

    return config_dict