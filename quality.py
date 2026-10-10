"""Photo-quality check and reliability estimate (mirrors the web app)."""
import numpy as np
from PIL import Image


def image_quality(img: Image.Image) -> dict:
    a = np.asarray(img.convert("RGB").resize((96, 96)), dtype=np.float32)
    lum = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]) / 255.0
    mean = float(lum.mean())
    grad = float((np.abs(np.diff(lum, axis=1)).mean() + np.abs(np.diff(lum, axis=0)).mean()) / 2)
    mx, mn = a.max(axis=2), a.min(axis=2)
    background = (mx > 204) & ((mx - mn) / np.maximum(mx, 1) < 0.18)
    coverage = float(1 - background.mean())

    b = 1 - min(1.0, abs(mean - 0.5) / 0.4)
    s = min(1.0, grad / 0.03)
    c = min(1.0, coverage / 0.4)
    tips = []
    if mean < 0.25: tips.append("too_dark")
    if mean > 0.80: tips.append("too_bright")
    if grad < 0.012: tips.append("blurry")
    if coverage < 0.15: tips.append("leaf_too_small")
    return {"score": (b + s + c) / 3, "brightness": mean, "sharpness": grad,
            "coverage": coverage, "tips": tips}


def reliability(confidence: float, quality_score: float) -> dict:
    """Estimate (NOT a validated accuracy): 60% model confidence + 40% photo quality."""
    r = 0.6 * min(1.0, confidence / 0.8) + 0.4 * quality_score
    level = "high" if r >= 0.7 else "medium" if r >= 0.45 else "low"
    return {"score": round(r, 3), "level": level}
