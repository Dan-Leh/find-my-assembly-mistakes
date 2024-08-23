import torch.nn.functional as F
import torch

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