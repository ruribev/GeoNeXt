import torch
import numpy as np
import torch.nn.functional as F


def dice_loss(
        prediction: torch.Tensor,
        target:     torch.Tensor,
        lam:        float = 0.1,   # boundary boost   (λ = 0 ⇒ classic Dice)
        gamma:      float = 1.0,   # shape of the bump
        smooth:     float = 1.0,
        eps:        float = 1e-7,
) -> torch.Tensor:
    """
    Boundary-Boosted Dice loss  (lightweight, NaN-safe).

    * Accepts **logits** or **probabilities**.
    * Works for binary or multi-class masks.

    Weight map  w = 1 + λ · (p·(1−p))^γ
    ─────────────────────────────────────
    • w≥1 everywhere  → interior pixels keep a full voice.  
    • Peak at p=0.5   → boundaries get   (1+λ·0.25^γ) × more gradient.

    Good starting values:  λ=1, γ=1.  Tune λ∈[0.5,2].
    """

    # ---------- 1. probabilities in (0,1) ------------------------------
    if prediction.min() < 0.0 or prediction.max() > 1.0:
        p = torch.sigmoid(prediction)
    else:
        p = prediction
    p = p.clamp(min=eps, max=1.0 - eps)          # avoid exact 0 / 1

    # ---------- 2. one-hot target if needed ----------------------------
    g = target.float()

    # ---------- 3. weight map ------------------------------------------
    w = 1.0 + lam * (p * (1.0 - p)).pow(gamma)

    # ---------- 4. flatten and compute Dice ----------------------------
    p_flat = (w * p).reshape(p.shape[0], p.shape[1], -1)
    g_flat = (w * g).reshape(g.shape[0], g.shape[1], -1)

    intersect = (p_flat * g_flat).sum(-1)
    denom     = p_flat.sum(-1) + g_flat.sum(-1)

    dice      = (2. * intersect + smooth) / (denom + smooth)
    return 1.0 - dice.mean()            # average over batch & classes


def focal_loss(
        prediction, target,
        alpha=0.3,  # FP penalty
        beta =0.7,  # FN penalty   (β>α favours recall)
        gamma=1.5,  # focal strength
        smooth=1.0):
    """
    logits: raw network output, target: binary mask
    """
    probs = torch.sigmoid(prediction)
    TP    = (probs * target).sum(dim=[1,2,3])
    FP    = (probs * (1-target)).sum(dim=[1,2,3])
    FN    = ((1-probs) * target).sum(dim=[1,2,3])

    tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
    loss    = (1 - tversky).pow(gamma)          # focal version
    return loss.mean()


def calc_loss(prediction, target, dice_weight=0.5, bce_weight=None, focal_weight=0.0, alpha=1.0, gamma=2.0, bcewl_weight=None):
    """Calculating the loss and metrics
    Args:
        prediction = predicted image (logits) or tuple for CALandDet
        target = Targeted image or tuple for CALandDet  
        bce_weight = weight for binary cross entropy loss (default: 0.5)
        dice_weight = weight for dice loss (default: 0.5)
        focal_weight = weight for focal loss (default: 0.0)
        alpha = weighting factor for focal loss rare class (default: 1.0)
        gamma = focusing parameter for focal loss (default: 2.0)
        bcewl_weight = class weights for binary cross entropy (default: None)
        model_type = type of model to determine loss calculation method
    Output:
        loss : combined loss of the epoch """
    
    # Handle CALandDet model output (tuple of segmentation and classification)
    if isinstance(prediction, tuple):
        # For CALandDet: prediction = (segmentation_output, classification_output)
        # For now, we only use the segmentation output for loss calculation
        prediction = prediction[0]  # Use only segmentation output
    
    # Handle target if it's also a tuple (for CALandDet with classification targets)
    if isinstance(target, tuple):
        target = target[0]  # Use only segmentation target
    
    # Handle None values and set defaults for standard models
    if dice_weight is None:
        dice_weight = 0.5
    if bce_weight is None:
        bce_weight = 0.5
    if focal_weight is None:
        focal_weight = 0.0
    
    # Normalize weights to sum to 1 if they don't already
    total_weight = bce_weight + dice_weight + focal_weight
    if total_weight > 0:
        bce_weight = bce_weight / total_weight
        dice_weight = dice_weight / total_weight
        focal_weight = focal_weight / total_weight
    
    # Calculate individual losses
    bce = F.binary_cross_entropy_with_logits(prediction, target, weight=bcewl_weight)
    
    # For dice loss, we need probabilities
    prediction_sigmoid = torch.sigmoid(prediction)
    dice = dice_loss(prediction_sigmoid, target)
    
    # Calculate focal loss if weight > 0
    if focal_weight > 0:
        focal = focal_loss(prediction, target, alpha=alpha, gamma=gamma)
    else:
        focal = 0.0
    
    # Combine losses
    loss = bce * bce_weight + dice * dice_weight + focal * focal_weight

    return loss, bce, dice, focal