"""Unit checks of model code paths not exercised by the default smoke: neck_dim=768, grad checkpointing,
feature='timm_head', checkpoint round-trip (load_ckpt), EMA update. CPU, random data, tiny sizes."""
import os, json, torch, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
src = (HERE / "kernels/smoke/team_offirst_encoder_byol_mnv3.py").read_text()
src = src[:src.index("# ----------------------------------------------------------------------------- main")]
g = {"__name__": "t"}; exec(compile(src, "kernel", "exec"), g)
CFG, res = g["CFG"], {}
def step_case(name, neck, feature, ckpt):
    CFG["model"]["neck_dim"] = neck; CFG["model"]["feature"] = feature; CFG["model"]["pretrained"] = False
    online, target, opt = g["build"](4, ckpt); scaler = torch.amp.GradScaler("cuda", enabled=False)
    G = torch.randint(0, 255, (4, 2, 3, 256, 256), dtype=torch.uint8); L = torch.randint(0, 255, (4, 4, 3, 128, 128), dtype=torch.uint8)
    t0 = [p.clone() for p in target.parameters()][:3]
    losses = [float(g["train_step"](online, target, opt, scaler, G, L, 0.9, False)[0]) for _ in range(3)]
    moved = any(not torch.equal(a, b) for a, b in zip(t0, list(target.parameters())[:3]))
    enc = g["encoder"](target); enc.eval()
    with torch.no_grad(): e = enc(torch.randn(2, 3, 256, 256))
    p = Path("/tmp/mnv3_t.pth"); torch.save(g["ckpt_dict"](online, target, 1, "anatomy_v4"), p)
    net2, tg2, _ = g["load_ckpt"](p); tg2.eval(); enc.eval()
    with torch.no_grad(): same = torch.allclose(g["encoder"](tg2)(torch.ones(1, 3, 256, 256)), enc(torch.ones(1, 3, 256, 256)), atol=1e-5)
    res[name] = {"losses": [round(x, 4) for x in losses], "finite": all(np.isfinite(losses)), "emb_shape": list(e.shape),
                 "target_ema_moved": moved, "ckpt_roundtrip_equal": bool(same)}
step_case("default_gap960", 0, "gap", False)
step_case("neck768", 768, "gap", False)
step_case("gap960_gradckpt", 0, "gap", True)
step_case("timm_head1280", 0, "timm_head", False)
print(json.dumps(res, indent=1)); json.dump(res, open(HERE / "localtest/test_model_paths.json", "w"), indent=1)
