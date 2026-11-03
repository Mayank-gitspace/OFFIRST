"""
U-Net for DR lesion segmentation (segmentation-models-pytorch), ConvNeXt-Tiny encoder by default.
ConvNeXt has no stride-2 feature level (stem is stride 4); smp handles this with a
0-channel stage, and the U-Net output still matches the input size.
4-class MULTI-LABEL output: one logit channel per lesion (MA, HE, EX, SE),
each passed through its own sigmoid — lesions may overlap, so no softmax/one-hot.
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


def is_valid_encoder(name):
    """Check a name against the installed smp encoder registry (+ timm for 'tu-').

    'tu-' names may carry a timm pretrained tag, e.g. 'tu-convnext_tiny.fb_in22k_ft_in1k'.
    timm.list_models() does not match tagged names unless pretrained=True, so the
    architecture and the tag are checked separately.
    """
    if name in smp.encoders.get_encoder_names():
        return True
    if name.startswith("tu-"):
        try:
            import timm
            arch, _, tag = name[3:].partition(".")
            if not timm.is_model(arch):
                return False
            return (not tag) or (name[3:] in timm.list_models(arch + "*", pretrained=True))
        except Exception:
            return False
    return False


def resolve_encoder(name, fallbacks=("tu-efficientnet_b3", "tu-mobilenetv3_large_100")):
    for cand in [name, *fallbacks]:
        if cand and is_valid_encoder(cand):
            if cand != name:
                warnings.warn(f"Encoder '{name}' not available in this smp; using '{cand}'.")
            return cand
    raise ValueError(f"No valid encoder among {[name, *fallbacks]}")


def build_model(encoder_name="tu-convnext_tiny", num_classes=4, pretrained=True,
                fallbacks=("tu-efficientnet_b3", "tu-mobilenetv3_large_100")):
    """Returns (model, resolved_encoder_name)."""
    enc = resolve_encoder(encoder_name, fallbacks)
    kwargs = dict(encoder_name=enc, in_channels=3, classes=num_classes, activation=None)
    try:
        # 'tu-' encoders take encoder_weights=True (any non-None works in older smp too)
        weights = (True if enc.startswith("tu-") else "imagenet") if pretrained else None
        model = smp.Unet(encoder_weights=weights, **kwargs)
    except Exception as e:  # e.g. no internet on Kaggle -> weights download fails
        if not pretrained:
            raise
        warnings.warn(f"Could not load ImageNet weights for '{enc}' ({e}); using random init.")
        model = smp.Unet(encoder_weights=None, **kwargs)
    return model, enc


def set_encoder_trainable(model, trainable):
    for p in model.encoder.parameters():
        p.requires_grad = trainable


def build_optimizer(model, lr, encoder_lr, weight_decay):
    """AdamW with two parameter groups: encoder (lower lr) and decoder + head (lr).

    Both groups are always present, so freezing/unfreezing the encoder only toggles
    requires_grad: frozen params get no .grad and AdamW skips them (no state, no decay).
    Per-group lrs keep their ratio under the cosine schedule (it scales each group's
    own initial_lr).
    """
    enc_ids = {id(p) for p in model.encoder.parameters()}
    enc = [p for p in model.parameters() if id(p) in enc_ids]
    dec = [p for p in model.parameters() if id(p) not in enc_ids]
    groups = [{"params": enc, "lr": encoder_lr, "name": "encoder"},
              {"params": dec, "lr": lr, "name": "decoder"}]
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


def _soft_dice_loss(logits, targets, smooth=1.0):
    """Per-class soft Dice over the whole batch (stable for sparse lesions), averaged."""
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    inter = (probs * targets).sum(dims)
    union = probs.sum(dims) + targets.sum(dims)
    dice = (2.0 * inter + smooth) / (union + smooth)
    return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits, targets: (B, 4, H, W); targets multi-hot float
        logits, targets = logits.float(), targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        return bce + _soft_dice_loss(logits, targets, self.smooth)


class FocalDiceLoss(nn.Module):
    """Sigmoid focal loss + Dice — better for tiny lesions (MA)."""
    def __init__(self, alpha=0.25, gamma=2.0, smooth=1.0):
        super().__init__()
        self.alpha, self.gamma, self.smooth = alpha, gamma, smooth

    def forward(self, logits, targets):
        logits, targets = logits.float(), targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = (alpha_t * (1 - p_t) ** self.gamma * bce).mean()
        return focal + _soft_dice_loss(logits, targets, self.smooth)


def build_loss(name):
    return FocalDiceLoss() if name == "focal_dice" else DiceBCELoss()
