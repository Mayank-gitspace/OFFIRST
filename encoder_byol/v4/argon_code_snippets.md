# Argon (Gemini) reply — BYOL v4 snippets

```python
# ==============================================================================
# 1) team_offirst_encoder_byol.py :: vicreg_var_cov (AMP-safe)
# ==============================================================================
import torch
import torch.nn.functional as F

def vicreg_var_cov(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4):
    """
    z: (B, D) online projector output. Casts to fp32 for AMP stability.
    Returns: (L_var, L_cov) scalar tensors.
    """
    z = z.float()
    B, D = z.shape
    if B <= 1:
        zero = z.new_zeros(())
        return zero, zero

    z_centered = z - z.mean(dim=0, keepdim=True)

    # Variance loss: hinge on per-dimension std across batch
    std = torch.sqrt(z_centered.var(dim=0, unbiased=False) + eps)
    l_var = F.relu(gamma - std).mean()

    # Covariance loss: sum of squared off-diagonal covariance elements / D
    cov = (z_centered.T @ z_centered) / (B - 1)
    cov_off_diag = cov - torch.diag(torch.diagonal(cov))
    l_cov = cov_off_diag.pow(2).sum() / D

    return l_var, l_cov
```

```python
# ==============================================================================
# 2) team_offirst_encoder_byol.py :: train_step integration (minimal diff)
# ==============================================================================
# Inside train_step(model, batch, cfg, ...), after obtaining online projections:
# Assume `z_online` is the (B, D) online projector tensor (e.g., stacked or global views)
# or a list of online projections `z_list = [z_g1, z_g2, ...]`:

    # --- Existing BYOL loss ---
    loss_byol = compute_byol_loss(p_online, z_target)

    # --- v4 VICReg regularizer on online projector z ---
    vc_cfg = cfg["train"]["vicreg"]
    if vc_cfg["enabled"]:
        # If z_online is a list of view projections, average var/cov across global views (or concat/stack)
        z_views = z_online if isinstance(z_online, (list, tuple)) else [z_online]
        var_terms, cov_terms = zip(*(vicreg_var_cov(z, gamma=vc_cfg["gamma"]) for z in z_views))
        l_var = torch.stack(var_terms).mean()
        l_cov = torch.stack(cov_terms).mean()
        loss_vicreg = vc_cfg["loss_weight"] * (vc_cfg["lambda_var"] * l_var + vc_cfg["lambda_cov"] * l_cov)
        loss = loss_byol + loss_vicreg
    else:
        l_var = l_cov = loss_vicreg = loss_byol.new_zeros(())
        loss = loss_byol

    # Log scalars for monitor
    metrics.update({
        "loss_total": loss.detach(),
        "loss_byol": loss_byol.detach(),
        "vicreg_var": l_var.detach(),
        "vicreg_cov": l_cov.detach(),
    })
```

```python
# ==============================================================================
# 3) team_offirst_encoder_byol.py :: make_views (mode == "anatomy_v4")
# ==============================================================================
import numpy as np

def apply_v4_pipeline(crop, rng, v_cfg):
    """Uniform v4 appearance pipeline for ALL global and local crops (including disc/mac)."""
    crop = apply_cam_bright(crop, rng, v_cfg["cam_bright"])
    if rng.random() < v_cfg["color_jitter"]["p"]:
        cj = v_cfg["color_jitter"]
        crop = apply_color_jitter(
            crop, rng,
            brightness=cj["brightness"],
            contrast=cj["contrast"],
            saturation=cj["saturation"],
            hue=cj["hue"],
        )
    if rng.random() < v_cfg["blur_p"]:
        crop = apply_gaussian_blur(crop, rng)
    crop = apply_jpeg_compression(crop, rng, v_cfg["jpeg"])
    return crop


def make_views(img, rng: np.random.Generator, row: dict, mode: str = "anatomy_v4", cfg: dict = None):
    cfg = cfg or CFG
    v_cfg = cfg["views"]

    # 1. Global views (2 x 384)
    globals_list = []
    for _ in range(v_cfg["n_globals"]):
        g_crop = crop_random_global(img, rng, size=v_cfg["global_size"], scale=v_cfg["global_scale"])
        g_crop = apply_v4_pipeline(g_crop, rng, v_cfg)
        globals_list.append(g_crop)

    # 2. Local views (4 x 160)
    locals_list = []
    anchor_stats = {"disc": 0, "mac": 0, "random": 0, "fallback_random": 0}

    if mode == "anatomy_v4":
        a_cfg = cfg["anatomy_v4"]
        local_sz = v_cfg["local_size"]

        # Fixed quota: 1 disc, 1 macula, 2 random -> shuffled per image
        slots = ["disc", "mac", "random", "random"]
        rng.shuffle(slots)

        for slot in slots:
            actual_slot = slot
            if slot == "disc":
                if row.get("disc_conf", 0.0) >= a_cfg["conf_thresh_disc"]:
                    dd = rng.uniform(*a_cfg["disc_dd"])  # strictly [2.5, 3.2]
                    l_crop = crop_disc(img, row, dd=dd, jitter_frac=a_cfg["jitter_frac"], size=local_sz, rng=rng)
                else:
                    actual_slot = "random"
                    anchor_stats["fallback_random"] += 1

            elif slot == "mac":
                if row.get("mac_conf", 0.0) >= a_cfg["conf_thresh_mac"]:
                    dd = rng.uniform(*a_cfg["mac_dd"])   # strictly [2.5, 3.2]
                    l_crop = crop_macula(img, row, dd=dd, jitter_frac=a_cfg["jitter_frac"], size=local_sz, rng=rng)
                else:
                    actual_slot = "random"
                    anchor_stats["fallback_random"] += 1

            if actual_slot == "random":
                l_crop = crop_random_local(img, rng, size=local_sz, scale=v_cfg["local_scale"])

            # Uniform colour jitter + blur + jpeg + cam_bright on ALL local views (no disc exemption)
            l_crop = apply_v4_pipeline(l_crop, rng, v_cfg)
            locals_list.append(l_crop)
            anchor_stats[actual_slot] += 1

        return globals_list, locals_list, anchor_stats

    raise ValueError(f"Unsupported view mode: {mode}")
```

```python
# ==============================================================================
# 4) config_v4.py / YAML-equivalent dict fragment :: CFG["anatomy_v4"] & train
# ==============================================================================
CFG_V4_PATCH = {
    "views": {
        "mode": "anatomy_v4",
        "n_globals": 2,
        "global_size": 384,
        "global_scale": (0.4, 1.0),
        "n_locals": 4,
        "local_size": 160,
        "local_scale": (0.08, 0.35),
        "color_jitter": {
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.15,
            "hue": 0.03,
            "p": 0.8,
            "disc_exempt": False,  # Deleted v2 disc-only mild BC exemption
        },
        "blur_p": 0.3,
        "jpeg": {"p": 0.3, "quality": (40, 90)},
        "cam_bright": {"p": 0.5, "range": (-0.1, 0.1)},
    },
    "anatomy_v4": {
        "slots": ("disc", "mac", "random", "random"),
        "shuffle_slots": True,
        "disc_dd": (2.5, 3.2),     # Locked medium-wide context; no tight 1.2-1.5
        "mac_dd": (2.5, 3.2),
        "jitter_frac": 0.15,
        "conf_thresh_disc": 0.5,
        "conf_thresh_mac": 0.5,
    },
    "train": {
        "epochs": 120,
        "warmup_epochs": 8,
        "vicreg": {
            "enabled": True,
            "loss_weight": 0.1,
            "lambda_var": 10.0,
            "lambda_cov": 1.0,
            "gamma": 1.0,
        },
    },
    "monitor": {
        "eval_epochs": [70, 120],
        "rare_headline": ["ODC", "ODP", "ARMD"],
    },
}
```

```python
# ==============================================================================
# 5) test_vicreg.py :: 5-line unit-test assert for vicreg_var_cov
# ==============================================================================
z_const = torch.ones(64, 128)
z_corr = torch.randn(64, 1) * torch.ones(1, 128)
z_orth = torch.nn.init.orthogonal_(torch.empty(64, 128)) * (64 ** 0.5)
v_const, c_const = vicreg_var_cov(z_const, gamma=1.0)
v_corr, c_corr = vicreg_var_cov(z_corr, gamma=1.0)
v_orth, c_orth = vicreg_var_cov(z_orth, gamma=1.0)
assert v_const > 0.99 and v_orth < 0.05 and c_corr > c_orth + 50.0, f"VICReg test failed: {(v_const, v_orth, c_corr, c_orth)}"
```
