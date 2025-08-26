import time
import torch
from tqdm import tqdm
from utils.losses import calc_loss
from utils.evaluator import evaluator
from utils.evaluator import get_result_dict


def training(train_loader, model, optimizer, epoch, evaluator, args):
    """Training function for one epoch - compatible with SAM-CFFNet style"""
    train_loss = 0.0
    train_bce_loss = 0.0
    train_dice_loss = 0.0
    train_focal_loss = 0.0
    total_num = 0.0
    start_since = time.time()
    device = args.device
    model.train()
    evaluator.reset()
    
    tbar = tqdm(train_loader, desc=f'Training {args.dataset}', leave=True)
    for i, batch in enumerate(tbar):
        x = batch[0]
        y = batch[1]

        rgb, target = x.to(device), y.to(device)
        output = model(rgb.float())

        # Get loss parameters from args, with defaults
        bce_weight = getattr(args, 'bce_weight', 0.5)
        dice_weight = getattr(args, 'dice_weight', None)
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
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update EMA if enabled
        if hasattr(args, 'ema') and args.ema is not None:
            args.ema.update()
        
        batch_size = x.size(0)
        total_num += batch_size
        train_loss += loss.data.cpu().numpy() * batch_size
        train_bce_loss += bce_loss.data.cpu().numpy() * batch_size
        train_dice_loss += dice_loss.data.cpu().numpy() * batch_size
        if isinstance(focal_loss, torch.Tensor):
            train_focal_loss += focal_loss.data.cpu().numpy() * batch_size
        else:
            train_focal_loss += focal_loss * batch_size

        pred = torch.where(output > 0.5, 1, 0)
    
        evaluator.add_batch(target, pred)
        tbar.set_description(f'Training {args.dataset} - Epoch: [{epoch+1:3d}]/[{args.epochs:3d}] '
                           f'Loss: {train_loss / total_num:.4f} '
                           f'(BCE: {train_bce_loss / total_num:.4f}, '
                           f'Dice: {train_dice_loss / total_num:.4f}, '
                           f'Focal: {train_focal_loss / total_num:.4f})')
    
    result_dict = get_result_dict(evaluator, epoch, start_since, time.time(), 
                                 train_loss, total_num, train_bce_loss, train_dice_loss, train_focal_loss)
    return result_dict

# Alias for backward compatibility
def train(train_loader, model, optimizer, epoch, evaluator, args):
    return training(train_loader, model, optimizer, epoch, evaluator, args)