"""Tiny VICReg unit check: extracts vicreg_var_cov from the built kernel and asserts collapse/redundancy is penalised."""
import re, torch, torch.nn.functional as F
src = open(__file__.replace("test_vicreg.py", "kernels/full/team_offirst_encoder_byol_anatomy_v4.py")).read()
fn = src[src.index("def vicreg_var_cov"):src.index("def byol_loss")]
ns = {"torch": torch, "F": F}; exec(fn, ns); vc = ns["vicreg_var_cov"]
torch.manual_seed(0)
z_const = torch.ones(512, 64)                                    # collapsed
z_corr = torch.randn(512, 1) * torch.ones(1, 64)                 # all dims redundant
z_orth = torch.randn(512, 64)                                    # decorrelated, unit std (B > D)
r = {k: tuple(float(x) for x in vc(z)) for k, z in [("const", z_const), ("corr", z_corr), ("orth", z_orth)]}
print(r)
# note: Argon's original test (64x128 orthogonal, const > 0.99) had wrong thresholds: per-dim std there is ~0.7 and const L_var = 1 - sqrt(eps)
assert r["const"][0] > 0.98 and r["orth"][0] < 0.05 and r["corr"][1] > 20 * r["orth"][1], r
zh = z_orth.half().requires_grad_(); l = sum(vc(zh)); l.backward(); assert torch.isfinite(zh.grad.float()).all()
print("VICReg unit check PASSED (fp16 input -> finite grads)")
