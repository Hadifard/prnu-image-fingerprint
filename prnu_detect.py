"""
Which images share a camera?
============================
Every camera sensor has tiny, fixed pixel flaws (PRNU) that leave a
noise "fingerprint" on every photo. AI images have no sensor, so they
have no fingerprint to match.

  1) noise residual = image - blurred image   (scene removed)
  2) correlate the noise of every pair of images
  3) 10+ sigmas above chance -> same camera
  4) an image that matches nothing is "unmatched": no camera could be linked to it.
     (Unmatched does NOT prove "AI": resized, edited or weak-sensor photos are unmatched too.)

The spectral peak score (periodic_peak_score) is printed for information only: on real
AI-generated portraits it does not separate them from camera photos.

Usage:  python prnu_detect.py --folder photos/
Needs:  numpy, scipy, pillow, matplotlib
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter
from scipy.sparse.csgraph import connected_components

MAX = 2048  # analyse at most a MAX x MAX center crop


def center(img, n):
    """Center crop of n x n pixels."""
    y, x = (img.shape[0] - n) // 2, (img.shape[1] - n) // 2
    return img[y:y + n, x:x + n]


def load_gray(path):
    """Grayscale, landscape, center crop (never resize!)."""
    img = np.asarray(Image.open(path).convert("L"), dtype=float)
    if img.shape[0] > img.shape[1]:
        img = np.rot90(img)  # portrait files -> landscape
    return center(img, min(MAX, *img.shape))


def load_gray_alt(path):
    """Portrait files only: the same photo rotated the OTHER way (cameras store portrait shots in both directions)."""
    img = np.asarray(Image.open(path).convert("L"), dtype=float)
    if img.shape[0] <= img.shape[1]:
        return None
    return center(np.rot90(img, 3), min(MAX, *img.shape))


def noise_residual(img):
    """Image minus its blurred copy = noise only."""
    res = img - gaussian_filter(img, sigma=1.5)
    res -= res.mean(axis=1, keepdims=True)  # row pattern
    res -= res.mean(axis=0, keepdims=True)  # column pattern
    local = np.sqrt(gaussian_filter(res ** 2, 3))
    return res / (local + 1.0)  # mute strong edges


def correlate(a, b):
    """Correlation of two noise patterns (common center area)."""
    n = min(len(a), len(b))
    a, b = center(a, n).ravel(), center(b, n).ravel()
    a, b = a - a.mean(), b - b.mean()
    r = a @ b / np.sqrt((a @ a) * (b @ b))
    return r, r * n  # z = sigmas above chance (chance ~ 1/n)


def correlation_matrix(res, alt=None):
    """R: correlation of noise.  Z: sigmas above chance.
    alt[i] = residual of portrait photo i rotated the other way (or None): the better of both orientations counts."""
    k = len(res)
    alt = alt if alt is not None else [None] * k
    R, Z = np.zeros((k, k)), np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            best = correlate(res[i], res[j])
            if alt[j] is not None:
                c = correlate(res[i], alt[j])
                best = c if c[1] > best[1] else best
            if alt[i] is not None:
                c = correlate(alt[i], res[j])
                best = c if c[1] > best[1] else best
            R[i, j], Z[i, j] = best
    return R + R.T, Z + Z.T


def group_by_camera(Z, z=10.0):
    """Pairs 10 sigmas above chance share one camera."""
    _, labels = connected_components(Z > z, directed=False)
    return labels


def periodic_peak_score(res):
    """High score = isolated sharp peaks in the spectrum."""
    F = np.abs(np.fft.fftshift(np.fft.fft2(center(res, 256))))
    c = F.shape[0] // 2
    F[c - 16:c + 17, c - 16:c + 17] = 0  # skip scene textures
    background = median_filter(F, size=9)
    return (F / (background + 1e-9)).max()


def analyze(paths, out_png="result.png"):
    names = [os.path.basename(p) for p in paths]
    res = [noise_residual(load_gray(p)) for p in paths]
    alt = [None if (a := load_gray_alt(p)) is None else noise_residual(a) for p in paths]
    R, Z = correlation_matrix(res, alt)
    peak = np.array([periodic_peak_score(r) for r in res])   # information only
    labels = group_by_camera(Z)
    sizes = np.bincount(labels)
    best = Z.argmax(axis=1)

    print(f"images: {len(names)}   match: z > 10 sigmas\n")
    cam_ids = {}
    for i, name in enumerate(names):
        if sizes[labels[i]] > 1:
            cam_ids.setdefault(labels[i], len(cam_ids) + 1)
            verdict = f"camera {cam_ids[labels[i]]}"
        else:
            verdict = "unmatched"
        print(f"{name}  r={R[i, best[i]]:+.3f}  "
              f"z={Z[i, best[i]]:6.1f}  peak={peak[i]:5.1f}  {verdict}")

    print("\nsummary")
    for lab, cid in cam_ids.items():
        print(f"  camera {cid}: {sizes[lab]} images")
    print(f"  unmatched: {int((sizes[labels] == 1).sum())} images")
    report(names, R, Z, peak, labels, out_png)


def report(names, R, Z, peak, labels, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = np.argsort(labels * 1e3 + np.arange(len(R)))
    zb = Z.max(axis=1)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    im = ax[0].imshow(R[np.ix_(order, order)], cmap="magma")
    ax[0].set_title("Noise correlation (grouped)")
    fig.colorbar(im, ax=ax[0], label="correlation r")
    ax[1].bar(range(len(R)), zb[order], color="tab:orange")
    ax[1].axhline(10, ls="--", c="k")
    ax[1].set_yscale("symlog")
    ax[1].set_title("Best match (sigmas above chance)")
    ax[2].scatter(zb, peak, c="tab:orange")
    ax[2].set_xscale("symlog")
    ax[2].set_xlabel("best match z")
    ax[2].set_ylabel("spectral peak score (info only)")
    ax[2].set_title("Match strength vs spectral peak")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"saved {out_png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="folder with images")
    ap.add_argument("--out", default="result.png")
    a = ap.parse_args()
    exts = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
    files = sorted(sum((glob.glob(os.path.join(a.folder, e))
                        for e in exts), []))
    analyze(files, a.out)
