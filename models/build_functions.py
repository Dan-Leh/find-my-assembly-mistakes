''' The code in this file has been adapted from "Remote Sensing Image Change Detection with Transformers": 
https://github.com/justchenhao/BIT_CD.git '''


import torch.optim as optim
from torch.optim import lr_scheduler
from models.network.cyws import CYWS
import torch
from torch.nn import init


def get_optimizer(model, args):
    ''' Return optimizer according to config argument. '''
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    else:
        raise NotImplementedError(f'Optimizer [{args.optimizer}] is not implemented')

    return optimizer


def get_scheduler(optimizer, args):
    """ Return a learning rate scheduler

    Parameters:
        optimizer: the optimizer of the network
        args (namespace): stores all the config arguments, args.lr_policy 
                          specifies the type of scheduler to use.

    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs//3
        scheduler = lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=0.1)
    elif args.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                            T_0=args.T_0, T_mult=args.T_mult)
    elif args.lr_policy == 'constant':  # do not change learning rate
        scheduler = lr_scheduler.ConstantLR(optimizer, factor=1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', 
                                   args.lr_policy)
    
    if args.warmup_epochs > 0:
        warmup_scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.1, 
                                                 total_iters=args.warmup_epochs)
        scheduler = lr_scheduler.SequentialLR(optimizer, 
                [warmup_scheduler, scheduler], milestones=[args.warmup_epochs])
        
    return scheduler


def build_model(args, init_gain=0.02, gpu=False, train=True):
    ''' Return model with initialized and/or pretrained weights. '''
    
    # instantiate model
    model = CYWS(cyws_params = args.cyws, classes = 2)
    device = torch.device("cuda" if torch.cuda.is_available() 
                                   and args.gpu else "cpu")
    model.to(device)
    if train == True:
        model = initialize_weights(model, args)

    return model, device


def initialize_weights(net, args):
    """Initialize the network weights
    
    Arguments:
        net (network): the network to be initialized
        args (namespace): the config arguments, args.init_type is the name of
            an initialization method: normal | xavier | kaiming | orthogonal
    Return an initialized network.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        init_gain = 0.02  # scaling factor for normal, xavier and orthogonal.
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or
                                        classname.find('Linear') != -1):
            if args.init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif args.init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif args.init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif args.init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'initialization method '\
                                f'{args.init_type} is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        # BatchNorm Layer is not a matrix; only normal distribution applies.
        elif classname.find('BatchNorm2d') != -1:  
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # only initialize layers without pretrained weights
    print(f'initialize network with {args.init_type}')
    if args.cyws['pretrained_encoder'] == True:
        net.unet_model.decoder.apply(init_func)
        net.unet_model.segmentation_head.apply(init_func)
        net.coattention_modules.apply(init_func)
    else:
        net.apply(init_func)  # apply the initialization function <init_func>
        
    return net


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

