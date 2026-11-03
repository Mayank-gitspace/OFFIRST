"""Cheap, explainable fundus-photo quality checks (no deep learning).
Gives the 'reason' half of the Upload / Retake verdict.
Thresholds are starting guesses: tune them on EyeQ Good vs Reject images.
"""
import cv2
import numpy as np

BLUR_MIN = 60.0        # variance of Laplacian inside the fundus mask
DARK_MEAN_MIN = 45.0   # mean green-channel brightness inside mask (0-255)
BRIGHT_P99_MAX = 252.0 # glare / over-exposure guard
DISC_EDGE_MARGIN = 0.08
ANISO_MIN = 0.45       # structure-tensor eigen ratio; motion blur smears edges in one direction
CENTRE_OFFSET_MAX = 0.12  # fundus field centroid offset from image centre (fraction of image size)  # optic disc centre must sit this far (fraction of radius) inside the field


def fundus_mask(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    return m > 0


def check(img_bgr, size=512):
    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    mask = fundus_mask(img)
    reasons, metrics = [], {}
    if mask.mean() < 0.2:
        return {"verdict": "Retake", "reasons": ["no retina in frame (or far too dark)"], "metrics": {"mask_frac": float(mask.mean())}}

    green = img[:, :, 1].astype(np.float32)
    # erode mask so the bright field edge doesn't count as "sharp detail"
    inner = cv2.erode(mask.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    lap = cv2.Laplacian(green, cv2.CV_32F, ksize=3)
    metrics["blur_var"] = float(lap[inner].var())
    metrics["mean_bright"] = float(green[mask].mean())
    metrics["p99_bright"] = float(np.percentile(green[mask], 99))

    gx = cv2.Sobel(green, cv2.CV_32F, 1, 0); gy = cv2.Sobel(green, cv2.CV_32F, 0, 1)
    jxx, jxy, jyy = (gx * gx)[inner].mean(), (gx * gy)[inner].mean(), (gy * gy)[inner].mean()
    ev = np.linalg.eigvalsh(np.array([[jxx, jxy], [jxy, jyy]]))
    metrics["aniso"] = float(ev[0] / max(ev[1], 1e-6))

    if metrics["blur_var"] < BLUR_MIN:
        reasons.append("too blurry")
    elif metrics["aniso"] < ANISO_MIN:
        reasons.append("motion blur (patient or camera moved)")
    if metrics["mean_bright"] < DARK_MEAN_MIN:
        reasons.append("too dark")
    if metrics["p99_bright"] > BRIGHT_P99_MAX:
        reasons.append("glare / over-exposed")

    # optic disc ~ brightest blob in a heavily smoothed image
    smooth = cv2.GaussianBlur(green, (0, 0), 15)
    smooth[~mask] = 0
    _, _, _, (dx, dy) = cv2.minMaxLoc(smooth)
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    r = np.sqrt(mask.sum() / np.pi)
    dist = np.hypot(dx - cx, dy - cy) / r
    metrics["field_offset"] = float(np.hypot(cx - size / 2, cy - size / 2) / size)
    if metrics["field_offset"] > CENTRE_OFFSET_MAX:
        reasons.append("retina not centred / cut off")
    metrics["disc_xy"] = (int(dx), int(dy))
    metrics["disc_dist_frac"] = float(dist)
    if dist > 1 - DISC_EDGE_MARGIN:
        reasons.append("optic disc not in frame")

    return {"verdict": "Retake" if reasons else "Upload", "reasons": reasons, "metrics": metrics}


if __name__ == "__main__":
    import sys, json
    for p in sys.argv[1:]:
        res = check(cv2.imread(p))
        print(p, json.dumps(res))
