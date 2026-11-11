"""Classical (non-ML) clean-tabletop -> portable-like fundus degrader for asymmetric KD.
Order: approx CRF tone -> radial vignetting -> defocus blur -> color/green shift -> light leak -> noise -> JPEG.
Defaults are fit to official RET-Amrita Visuscout 100 samples (2026-11-06). Not a recovered camera CRF.
Usage: python3 degrade.py in.png out.png [--seed 0]"""
import cv2, numpy as np, argparse
from dataclasses import dataclass, asdict

@dataclass
class DegradeParams:
    # Defaults fit 2026-11-06 to 24 official RET-Amrita Visuscout 100 frames vs RFMiD
    # (see visuscout_params.json). Exposure nudged 1.04 -> 1.12 to close leftover L gap.
    crf_gamma: float = 1.05        # mild mid-tone roll; almost matches Visuscout brightness
    crf_toe: float = 0.02
    crf_shoulder: float = 0.92
    exposure: float = 1.12         # do NOT crush brightness (Visuscout L ≈ RFMiD)
    vignette_strength: float = 0.16  # mild rim darkening (edge/centre 1.02 -> ~0.93)
    vignette_power: float = 2.0
    blur_sigma: float = 1.15       # matches Sobel-mean drop ~18.6 -> ~14.2
    gain_r: float = 0.85           # cooler / less red than tabletop RFMiD
    gain_g: float = 1.05
    gain_b: float = 1.15           # more blue in the mix
    leak_strength: float = 0.0     # 0 = off; rare green crescent seen on some Visuscout frames
    leak_angle_deg: float = -1.0
    leak_width: float = 0.25
    noise_sigma: float = 1.4
    jpeg_quality: int = 70
    seed: int = 0

def _fov(img):
    m = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 15).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)); return m.astype(np.float32)

def crf(x, p):
    x = np.clip(x * p.exposure, 0, None)
    y = np.power(np.clip(x, 0, 1), p.crf_gamma)
    s = p.crf_shoulder  # soft shoulder above s
    hi = y > s; y[hi] = s + (1 - s) * (1 - np.exp(-(y[hi] - s) / (1 - s)))
    return p.crf_toe + (1 - p.crf_toe) * y

def degrade(img, p: DegradeParams = DegradeParams()):
    rng = np.random.default_rng(p.seed)
    h, w = img.shape[:2]; m = _fov(img); scale = w / 1024
    x = crf(img.astype(np.float32) / 255, p)
    ys, xs = np.nonzero(m > 0); cy, cx = ys.mean(), xs.mean(); R = np.sqrt(m.sum() / np.pi)
    Y, X = np.indices((h, w)); rad = np.clip(np.sqrt((Y - cy) ** 2 + (X - cx) ** 2) / R, 0, 1.2)
    x *= (1 - p.vignette_strength * np.power(np.clip(rad, 0, 1), p.vignette_power))[..., None]
    if p.blur_sigma > 0: x = cv2.GaussianBlur(x, (0, 0), p.blur_sigma * scale)
    x *= np.array([p.gain_b, p.gain_g, p.gain_r], np.float32)  # BGR
    if p.leak_strength > 0:
        ang = np.deg2rad(p.leak_angle_deg if p.leak_angle_deg >= 0 else rng.uniform(0, 360))
        d = ((X - cx) * np.cos(ang) + (Y - cy) * np.sin(ang)) / R
        leak = np.clip((d - (1 - p.leak_width)) / p.leak_width, 0, 1) ** 2
        x += (p.leak_strength * leak)[..., None] * np.array([0.45, 1.0, 0.35], np.float32)  # greenish leak (seen on some Visuscout frames)
    if p.noise_sigma > 0: x += rng.normal(0, p.noise_sigma / 255, x.shape).astype(np.float32)
    x *= m[..., None]
    out = (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
    if p.jpeg_quality < 100:
        out = cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(p.jpeg_quality)])[1], 1)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--leak", type=float, default=0.0); a = ap.parse_args()
    cv2.imwrite(a.out, degrade(cv2.imread(a.inp), DegradeParams(seed=a.seed, leak_strength=a.leak))); print(asdict(DegradeParams()))
