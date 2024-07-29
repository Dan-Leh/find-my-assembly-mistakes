

# functions that will later be moved to where they belong
import torch.optim as optim

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


from torch.optim import lr_scheduler

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


from cyws_files.cyws import CYWS
import sys; sys.path.insert(1, './../assembly-error-localization'); from cyws_files.cyws import CYWS
import torch
from torch.nn import init

def build_model(args, init_gain=0.02, gpu=False):
    ''' Return model with initialized and/or pretrained weights. '''
    
    # instantiate model
    model = CYWS(cyws_params = args.cyws, classes = args.output_dim)
    device = torch.device("cuda" if torch.cuda.is_available() 
                                   and args.gpu else "cpu")
    model.to(device)
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


import torch.nn.functional as F

def get_loss_func(loss_type):
    
    def cross_entropy(input, target, weight=None, reduction='mean'):
        
        target = target.long()
        if target.dim() == 4:
            target = torch.squeeze(target, dim=1)
        if input.shape[-1] != target.shape[-1]:
            raise ValueError('prediction and target shape do not match')

        return F.cross_entropy(input=input, target=target, weight=weight,
                            reduction=reduction)

    def focal_loss(input, target, gamma=1, alpha=0.05, reduction='mean'):
        ''' We assume that about 5% of the pixels are change, so in addition to
        focal loss, we have a weighting factor on the cross-entropy'''

        # turn target into one-hot
        target_onehot = torch.cat([1-target, target], dim=1)
        
        pt = alpha*torch.mul(input[:,0,:,:], target_onehot[:,0,:,:]) \
            + (1-alpha)*torch.mul(input[:,1,:,:], target_onehot[:,1,:,:])
        loss = -torch.pow(1 - pt, gamma) * torch.log(pt) 

        if reduction == 'mean':
            loss = torch.mean(loss)
            
        return loss

    if loss_type == 'ce':
        loss_function = cross_entropy
    elif loss_type == 'focal':
        loss_function = focal_loss
    else:
        raise NotImplementedError(loss_type)

    return loss_function
