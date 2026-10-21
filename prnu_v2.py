"""
prnu_v2.py - modern PRNU camera identification (works on resized / cropped / rotated copies)
=========================================================================================
  1) noise residual   : undecimated wavelet (starlet) denoiser, Mihcak-type local-variance Wiener shrinkage
  2) fingerprint K    : maximum-likelihood estimate  K = sum(W_i * I_i) / sum(I_i^2)   (Chen et al.),
                        row/column means removed, spectral peaks flattened
  3) detection        : cross-correlation of the test residual with I*K over ALL shifts (FFT),
                        statistic = PCE (peak-to-correlation energy), PCE > 60 is the classic decision level
  4) geometry search  : rotation (0/90/180/270) x scale (resize factor) x shift (crop position)
  5) tamper map       : sliding-window correlation at the winning alignment, weak blocks = edited area

Needs: numpy, scipy, pillow, opencv-python (cv2)
"""
import os
import numpy as np
import cv2
from PIL import Image
from scipy import fft as sfft

SIGMA0 = 3.0          # noise level (gray levels) assumed by the wavelet filter
LEVELS = 4
PCE_T = 60.0          # classic Goljan/Fridrich decision level
WORKERS = 2

# ----------------------------------------------------------------------------------------------- loading
def load_gray(path):
    """Gray float32 in the file's own pixel frame (EXIF orientation is NOT applied: PRNU lives in sensor pixels)."""
    im = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    return im @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


# ----------------------------------------------------------------------------------------------- noise residual
_H = np.array([1, 4, 6, 4, 1], np.float32) / 16


def _kern(j):
    k = np.zeros(4 * 2 ** j + 1, np.float32)
    k[::2 ** j] = _H
    return k


def _gains(levels):
    x = np.random.default_rng(0).normal(size=(512, 512)).astype(np.float32)
    g, c = [], x
    for j in range(levels):
        k = _kern(j)
        cn = cv2.sepFilter2D(c, -1, k, k, borderType=cv2.BORDER_REFLECT)
        g.append(float((c - cn).std()))
        c = cn
    return g


GAIN = _gains(LEVELS)


def extract_noise(gray, sigma0=SIGMA0, levels=LEVELS):
    """Residual W = image - denoise(image); zero-meaned per row and column."""
    g = gray.astype(np.float32)
    c, ws = g, []
    for j in range(levels):
        k = _kern(j)
        cn = cv2.sepFilter2D(c, -1, k, k, borderType=cv2.BORDER_REFLECT)
        ws.append(c - cn)
        c = cn
    rec = c.copy()
    for j, w in enumerate(ws):
        s0 = (sigma0 * GAIN[j]) ** 2
        w2 = w * w
        best = None
        for win in (3, 5, 7, 9):                                   # local signal variance, min over window sizes
            v = np.maximum(cv2.boxFilter(w2, -1, (win, win), borderType=cv2.BORDER_REFLECT) - s0, 0)
            best = v if best is None else np.minimum(best, v)
        rec += w * best / (best + s0)
    n = g - rec
    n -= n.mean(1, keepdims=True)
    n -= n.mean(0, keepdims=True)
    return n


# ----------------------------------------------------------------------------------------------- fingerprint
def estimate_fingerprint(grays, noises):
    """Maximum-likelihood PRNU estimate from n photos of the same size (clipped pixels are ignored)."""
    num = np.zeros_like(noises[0], dtype=np.float32)
    den = np.zeros_like(num)
    for g, n in zip(grays, noises):
        w = ((g > 4) & (g < 250)).astype(np.float32)
        num += n * g * w
        den += g * g * w
    K = num / (den + 1.0)
    return clean_fingerprint(K)


def clean_fingerprint(K):
    K = K - K.mean(1, keepdims=True)
    K = K - K.mean(0, keepdims=True)
    F = sfft.rfft2(K, workers=WORKERS)
    mag = np.abs(F).astype(np.float32)
    sm = cv2.boxFilter(mag, -1, (9, 9), borderType=cv2.BORDER_REFLECT)
    F *= np.minimum(1.0, 3.0 * sm / (mag + 1e-9))                  # flatten isolated spectral peaks
    return sfft.irfft2(F, s=K.shape, workers=WORKERS).astype(np.float32)


# ----------------------------------------------------------------------------------------------- correlation
def resample(gray, scale):
    """Undo a resize: a test image that is `scale` times the original is enlarged by 1/scale."""
    if abs(scale - 1.0) < 1e-9:
        return gray
    return cv2.resize(gray, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_CUBIC)


def center_patch(a, n):
    h, w = a.shape
    nh, nw = min(n, h), min(n, w)
    y, x = (h - nh) // 2, (w - nw) // 2
    return a[y:y + nh, x:x + nw], (y, x)


class Reference:
    """A camera fingerprint with its FFT cached."""

    def __init__(self, K, name=""):
        self.K, self.name = K.astype(np.float32), name
        self.shape = K.shape
        self.Kf = sfft.rfft2(self.K, workers=WORKERS)


def xcorr(prod, ref):
    """c[s] = sum_x prod(x) * K(x+s) for every top-left shift s (circular, prod zero padded)."""
    H, W = ref.shape
    a = np.zeros((H, W), np.float32)
    a[:prod.shape[0], :prod.shape[1]] = prod
    return sfft.irfft2(np.conj(sfft.rfft2(a, workers=WORKERS)) * ref.Kf, s=(H, W), workers=WORKERS)


def pce_of(c, valid):
    """PCE of the best peak inside the valid shift area; null from valid shifts (or all if too few)."""
    vh, vw = valid
    cv = c[:vh, :vw]
    idx = np.unravel_index(np.argmax(cv), cv.shape)
    peak = float(cv[idx])
    mask = np.ones(c.shape, bool) if vh * vw < 5000 else np.zeros(c.shape, bool)
    if vh * vw >= 5000:
        mask[:vh, :vw] = True
    y0, x0 = idx
    ys = [(y0 + d) % c.shape[0] for d in range(-5, 6)]
    xs = [(x0 + d) % c.shape[1] for d in range(-5, 6)]
    mask[np.ix_(ys, xs)] = False
    null = float(np.mean(c[mask] ** 2))
    return peak * abs(peak) / (null + 1e-30), idx, peak, np.sqrt(null)


def test_prod(gray, scale=1.0, rot=0, patch=None):
    """Prepare I*W of the test image at the camera's pixel scale."""
    g = np.ascontiguousarray(np.rot90(gray, rot))
    g = resample(g, scale)
    origin = (0, 0)
    if patch:
        g, origin = center_patch(g, patch)
        g = np.ascontiguousarray(g)
    W = extract_noise(g)
    return W * g, g, W, origin


def match(gray, ref, scale=1.0, rot=0, patch=None):
    """PCE of one geometry hypothesis. Returns dict(pce, shift (y,x) of the test's top-left inside K, ...)."""
    prod, g, W, origin = test_prod(gray, scale, rot, patch)
    H, Wd = ref.shape
    if g.shape[0] > H or g.shape[1] > Wd:
        return dict(pce=0.0, shift=(0, 0), peak=0.0, null=1.0, scale=scale, rot=rot, ok=False)
    c = xcorr(prod, ref)
    pce, idx, peak, null = pce_of(c, (H - g.shape[0] + 1, Wd - g.shape[1] + 1))
    sh = (idx[0] - origin[0], idx[1] - origin[1])                  # position of the full test frame in K
    return dict(pce=pce, shift=sh, peak=peak, null=null, scale=scale, rot=rot, ok=True, c=c, patch_shape=g.shape)


def search(gray, ref, scales, rots=(0, 1, 2, 3), patch=768, keep_curve=False):
    """Scan rotations x scales; returns best hypothesis (and the PCE-vs-scale curves)."""
    best, curves = None, {}
    for r in rots:
        cur = []
        for s in scales:
            m = match(gray, ref, s, r, patch)
            m.pop("c", None)
            cur.append((s, m["pce"]))
            if best is None or m["pce"] > best["pce"]:
                best = m
        curves[r] = cur
    best["curves"] = curves
    return best


def refine(gray, ref, best, half=0.004, step=0.0004, patch=1400):
    """Fine scale scan around a coarse winner, then full-frame PCE."""
    rot = best["rot"]
    top = dict(best)
    for s in np.arange(best["scale"] - half, best["scale"] + half + 1e-9, step):
        m = match(gray, ref, float(s), rot, patch)
        m.pop("c", None)
        if m["pce"] > top["pce"]:
            top = m
    return top


# ----------------------------------------------------------------------------------------------- tamper map
def local_map(gray, ref, hyp, win=96, stride=24):
    """Sliding-window correlation r between the test noise and I*K at the winning geometry.
    Returns (r_map, grid_step, test_gray_at_camera_scale)."""
    g = np.ascontiguousarray(np.rot90(gray, hyp["rot"]))
    g = resample(g, hyp["scale"])
    y0, x0 = hyp["shift"]
    h, w = g.shape
    H, W = ref.shape
    ys, xs = max(y0, 0), max(x0, 0)                                # K window that overlaps the test frame
    ye, xe = min(y0 + h, H), min(x0 + w, W)
    g = g[ys - y0:ye - y0, xs - x0:xe - x0]
    Kc = ref.K[ys:ye, xs:xe]
    Wn = extract_noise(g)
    P = g * Kc
    box = lambda a: cv2.boxFilter(a, -1, (win, win), normalize=False, borderType=cv2.BORDER_REFLECT)
    num, a2, b2 = box(Wn * P), box(Wn * Wn), box(P * P)
    r = num / (np.sqrt(a2 * b2) + 1e-9)
    return r[win // 2::stride, win // 2::stride], r, g, (ys, xs)


def load_cached_noise(path, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    f = os.path.join(cache_dir, os.path.basename(path) + ".npy")
    if os.path.exists(f):
        return np.load(f).astype(np.float32)
    n = extract_noise(load_gray(path))
    np.save(f, n.astype(np.float16))
    return n
