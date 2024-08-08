import numpy as np

###################       metrics      ###################
class AverageMeter(object):
    """ Compute and store the average and current values. """
    
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict

    def reset(self):
        self.initialized = False


###################      cm metrics      ###################
class ConfuseMatrixMeter(AverageMeter):
    """Computes and stores the average and current value"""
    
    def __init__(self, n_class):
        super(ConfuseMatrixMeter, self).__init__()
        self.n_class = n_class

    def update_cm(self, pr:np.ndarray, gt:np.ndarray, return_iou:bool=True) -> float:
        """ Update running confusion matrix and return current IoU score"""
        
        confuse_matrix = get_confuse_matrix(num_classes=self.n_class, 
                                            label_gts=gt, label_preds=pr) 
        self.update(confuse_matrix, weight=1)  # updates values of the running cm
        current_scores = cm2score(confuse_matrix) 
        if return_iou:
            return current_scores['iou_1']

    def get_scores(self):
        ''' Convert the sum of the running cofusion matrix into scores like IoU '''
        
        scores_dict = cm2score(self.sum)
        return scores_dict


def cm2score(confusion_matrix):
    ''' Compute all scores of interest from confusion matrix. 
    
    Returns:
        score_dict (dict):  - miou: mean intersection over union
                            - iou_0: iou for 'no change' class
                            - iou_1: iou for 'change' class
                            - mf1: mean F1 scores
                            - f1_0: F1 score for 'no change' class
                            - f1_1: F1 score for 'change' class
                            - accuracy
                            - precision
                            - recall   
    '''
    hist = confusion_matrix
    n_class = hist.shape[0]
    true_pred = np.diag(hist)       # TP & TN
    actual_pos = hist.sum(axis=1)   # TP + FN
    pred_pos = hist.sum(axis=0)     # TP + FP
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & f1
    # ---------------------------------------------------------------------- #
    accuracy = true_pred.sum() / (hist.sum() + np.finfo(np.float32).eps)
    recall = true_pred / (actual_pos + np.finfo(np.float32).eps)
    precision = true_pred / (pred_pos + np.finfo(np.float32).eps)

    f1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_f1 = np.nanmean(f1)
    # ---------------------------------------------------------------------- #
    # 2. IoU
    # ---------------------------------------------------------------------- #
    iou = true_pred / (actual_pos + pred_pos - 
                       true_pred + np.finfo(np.float32).eps)
    mean_iu = np.nanmean(iou)

    ious = dict(zip(['iou_'+str(i) for i in range(n_class)], iou))
    precisions = dict(zip(['precision_'+str(i) for i in range(n_class)], precision))
    recalls = dict(zip(['recall_'+str(i) for i in range(n_class)], recall))
    f1s = dict(zip(['f1_'+str(i) for i in range(n_class)], f1))

    score_dict = {'accuracy': accuracy, 'miou': mean_iu, 'mf1':mean_f1}
    score_dict.update(ious)
    score_dict.update(f1s)
    score_dict.update(precisions)
    score_dict.update(recalls)
    
    return score_dict


def get_confuse_matrix(num_classes, label_gts, label_preds):
    """"""
    def __fast_hist(label_gt, label_pred):
        """ Collect values for Confusion Matrix.
        
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :param label_gt: <np.array> ground-truth
        :param label_pred: <np.array> prediction
        :return: <np.ndarray> values for confusion matrix
        """
        mask = (label_gt >= 0) & (label_gt < num_classes)
        hist = np.bincount(num_classes * label_gt[mask].astype(int) + 
                           label_pred[mask], minlength=num_classes**2).reshape(
                                                        num_classes, num_classes)
        return hist
    confusion_matrix = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_gts, label_preds):
        confusion_matrix += __fast_hist(lt.flatten(), lp.flatten())
    return confusion_matrix
