import numpy as np
import torch
import time
from tqdm import tqdm
from utils.losses import calc_loss

np.seterr(divide='ignore',invalid='ignore')


def get_result_dict(evaluator, epoch, start_time, end_time, total_loss, total_num, total_bce=0, total_dice=0, total_focal=0):
    """Create result dictionary from evaluator metrics"""
    return {
        'epoch': epoch,
        'loss': total_loss / total_num,
        'bce_loss': total_bce / total_num,
        'dice_loss': total_dice / total_num,
        'focal_loss': total_focal / total_num,
        'accuracy': evaluator.OverallAccuracy(),
        'mIou': evaluator.MeanIntersectionOverUnion(),
        'IoU': evaluator.IntersectionOverUnion(),
        'precision': evaluator.Precision(),
        'recall': evaluator.Recall(),
        'f1': evaluator.F1Score(),
        'duration': end_time - start_time
    }

class evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def OverallAccuracy(self):  
        OA = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()  
        return OA
    
    def Precision(self):  
        # Handle division by zero case
        denominator = self.confusion_matrix.sum(axis=0)
        precision = np.where(denominator == 0, 0.0, np.diag(self.confusion_matrix) / denominator)
        return precision  

    def Recall(self):
        # Handle division by zero case
        denominator = self.confusion_matrix.sum(axis=1)
        recall = np.where(denominator == 0, 0.0, np.diag(self.confusion_matrix) / denominator)
        return recall
    
    def F1Score(self):
        precision = self.Precision()
        recall = self.Recall()
        # Handle division by zero case
        denominator = precision + recall
        f1score = np.where(denominator == 0, 0.0, 2 * precision * recall / denominator)
        return f1score

    def IntersectionOverUnion(self):  
        intersection = np.diag(self.confusion_matrix)  
        union = np.sum(self.confusion_matrix, axis = 1) + np.sum(self.confusion_matrix, axis = 0) - np.diag(self.confusion_matrix)  
        IoU = intersection / union
        return IoU

    def MeanIntersectionOverUnion(self):  
        intersection = np.diag(self.confusion_matrix)  
        union = np.sum(self.confusion_matrix, axis = 1) + np.sum(self.confusion_matrix, axis = 0) - np.diag(self.confusion_matrix)  
        IoU = intersection / union
        mIoU = np.nanmean(IoU)  
        return mIoU
    
    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU
    
    def _generate_matrix(self, gt_image, pre_image):
        if  'torch' in str(gt_image.dtype):
            gt_image = gt_image.cpu()
            gt_image = gt_image.numpy()
        if 'torch' in str(pre_image.dtype):
            pre_image = pre_image.cpu()
            pre_image = pre_image.numpy()
        gt_image = gt_image.astype('int') 
        pre_image = pre_image.astype('int')
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask]+ pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

def validationing(val_loader, model, epoch, evaluator, args, dataset_name=None):
    """Validation function for one epoch - compatible with SAM-CFFNet style"""
    val_loss = 0.0
    val_bce_loss = 0.0
    val_dice_loss = 0.0
    val_focal_loss = 0.0
    val_num = 0.0
    start_since = time.time()
    device = args.device
    model.eval()
    evaluator.reset()
    
    # Use provided dataset_name or fall back to args.dataset
    display_name = dataset_name if dataset_name is not None else args.dataset
    
    # Use EMA weights for validation if available
    use_ema = hasattr(args, 'ema') and args.ema is not None
    if use_ema:
        args.ema.apply_shadow()
    
    try:
        tbar = tqdm(val_loader, desc=f'Validating {display_name}', leave=True)
        for i, batch in enumerate(tbar):
            x = batch[0]
            y = batch[1]

            rgb, target = x.to(device), y.to(device)
            with torch.no_grad():
                output = model(rgb.float())

            # Get loss parameters from args, with defaults
            bce_weight = getattr(args, 'bce_weight', 0.5)
            dice_weight = getattr(args, 'dice_weight', 0.5)
            focal_weight = getattr(args, 'focal_weight', 0.0)
            bcewl_weight = getattr(args, 'bcewl_weight', None)
            focal_alpha = getattr(args, 'focal_alpha', 1.0)
            focal_gamma = getattr(args, 'focal_gamma', 2.0)        

            loss, bce_loss, dice_loss, focal_loss = calc_loss(output, target.float(), 
                            bce_weight=bce_weight,
                            dice_weight=dice_weight,
                            focal_weight=focal_weight,
                            alpha=focal_alpha,
                            gamma=focal_gamma,
                            bcewl_weight=bcewl_weight)
            
            batch_size = x.size(0)
            val_num += batch_size
            val_loss += loss.data.cpu().numpy() * batch_size
            val_bce_loss += bce_loss.data.cpu().numpy() * batch_size
            val_dice_loss += dice_loss.data.cpu().numpy() * batch_size
            if isinstance(focal_loss, torch.Tensor):
                val_focal_loss += focal_loss.data.cpu().numpy() * batch_size
            else:
                val_focal_loss += focal_loss * batch_size

            # Handle tuple output for CALandDet model (output contains segmentation + classification)
            if isinstance(output, tuple):
                # Use only the segmentation output for prediction
                seg_output = output[0]
            else:
                seg_output = output
                
            pred = torch.where(seg_output > 0.5, 1, 0)
        
            evaluator.add_batch(target, pred)
            tbar.set_description(f'Validating {display_name} - Epoch: [{epoch+1:3d}]/[{args.epochs:3d}] '
                               f'Loss: {val_loss / val_num:.4f} '
                               f'(BCE: {val_bce_loss / val_num:.4f}, '
                               f'Dice: {val_dice_loss / val_num:.4f}, '
                               f'Focal: {val_focal_loss / val_num:.4f})')
    finally:
        # Always restore original weights after validation
        if use_ema:
            args.ema.restore()
    
    # Import get_result_dict from train module
    result_dict = get_result_dict(evaluator, epoch, start_since, time.time(), 
                                 val_loss, val_num, val_bce_loss, val_dice_loss, val_focal_loss) 
    return result_dict
