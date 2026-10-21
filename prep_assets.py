"""
Computes every real number / image the video needs, using the functions of prnu_detect.py (25-image demo folder).
Result is cached in assets.pkl (render.py only draws).
"""
import glob, json, os, pickle, re, sys
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prnu_detect as P

files = sorted(sum((glob.glob(os.path.join(HERE, "photos39", e)) for e in ("*.jpg", "*.png", "*.tif")), []))
names = [os.path.basename(f) for f in files]
truth = json.load(open(os.path.join(HERE, "truth39.json")))
out_txt = open(os.path.join(HERE, "run_output.txt")).read().splitlines()
A = {"names": names, "truth": truth, "run_output": out_txt}

# ---- verdicts exactly as printed by the script
V = {}
for ln in out_txt:
    m = re.match(r"(\S+\.(?:jpg|png|tif))\s+r=(\S+)\s+z=\s*(\S+)\s+peak=\s*(\S+)\s+(.*)", ln)
    if m: V[m.group(1)] = dict(r=float(m.group(2)), z=float(m.group(3)), peak=float(m.group(4)), verdict=m.group(5).strip())
A["verdict"] = [V[n]["verdict"] for n in names]
A["cam"] = [int(v.split()[1]) if v.startswith("camera") else 0 for v in A["verdict"]]
A["is_ai"] = np.array([truth[n]["kind"] == "ai" for n in names])     # GROUND TRUTH (the detector no longer labels AI)
A["kind"] = [truth[n]["kind"] for n in names]          # real / sim / ai
A["label"] = [truth[n]["label"] for n in names]
A["printed"] = V

# ---- residuals / correlations (same functions as the script)
res = {n: P.noise_residual(P.load_gray(f)) for n, f in zip(names, files)}
alt = [None if (a := P.load_gray_alt(f)) is None else P.noise_residual(a) for f in files]   # portrait photos: both rotation directions
R, Z = P.correlation_matrix([res[n] for n in names], alt)
peak = np.array([P.periodic_peak_score(res[n]) for n in names])
Rc, Zc = R, Z
for i, n in enumerate(names):                             # consistency with the printed run
    j = Zc[i].argmax()
    assert abs(peak[i] - V[n]["peak"]) < 0.06, (n, peak[i], V[n]["peak"])
    assert abs(Rc[i, j] - V[n]["r"]) < 0.0006 and abs(Zc[i, j] - V[n]["z"]) < 0.06, (n, Rc[i, j], V[n]["r"])
A.update(R=Rc, Z=Zc, peak=peak,
         zbest=np.array([Zc[i].max() for i in range(len(names))]), rbest=np.array([Rc[i, Zc[i].argmax()] for i in range(len(names))]))
print("peak range  AI:", peak[A["is_ai"]].min(), peak[A["is_ai"]].max(), " camera/other:", peak[~A["is_ai"]].min(), peak[~A["is_ai"]].max())

# ---- display images
FOCUS = {"girl.jpg": (0.55, 0.5), "cat_plane.jpg": (0.66, 0.55), "red_shirt.png": (0.5, 0.33), "man.jpg": (0.5, 0.45),
         "AI_generated_person_01.jpg": (0.5, 0.33), "AI_generated_person_02.jpg": (0.5, 0.33), "AI_generated_person_03.jpg": (0.5, 0.4),
         "AI_generated_person_04.jpg": (0.5, 0.4), "AI_generated_person_05.jpeg": (0.4, 0.2), "AI_generated_person_06.jpg": (0.5, 0.4)}
def load_disp(path, maxside=1500):
    im = Image.open(path); im.draft("RGB", (maxside, maxside))
    im = ImageOps.exif_transpose(im).convert("RGB"); im.thumbnail((maxside, maxside), Image.LANCZOS); return im
def cover(im, w, h, focus=(0.5, 0.5)):
    s = max(w / im.width, h / im.height); iw, ih = im.width * s, im.height * s
    x0 = min(max(focus[0] * iw - w / 2, 0), iw - w); y0 = min(max(focus[1] * ih - h / 2, 0), ih - h)
    return im.resize((int(round(iw)), int(round(ih))), Image.LANCZOS).crop((int(x0), int(y0), int(x0) + w, int(y0) + h))
DISP = {}
for n in names:
    p = truth[n]["disp"]; k = os.path.basename(p)
    if p not in DISP: DISP[p] = load_disp(p)
A["thumb"] = [cover(DISP[truth[n]["disp"]], 320, 320, FOCUS.get(os.path.basename(truth[n]["disp"]), (0.5, 0.5))) for n in names]
A["wide"] = [cover(DISP[truth[n]["disp"]], 480, 300, FOCUS.get(os.path.basename(truth[n]["disp"]), (0.5, 0.5))) for n in names]
idx = {truth[n]["disp"].split("/")[-1]: i for i, n in enumerate(names)}       # original file -> index
A["idx"] = idx
print("indices", {k: idx[k] for k in ("girl.jpg", "cat_plane.jpg", "red_shirt.png", "man.jpg", "20230505_125151.jpg", "20240518_203321.jpg")})

# ---- S2: real fingerprint estimate from the six camera-1 photos
cam1 = [n for n, v, t in zip(names, A["verdict"], A["kind"]) if v == "camera 1" and truth[n]["group"] == "cam_samsung"]
assert len(cam1) == 6
setA = [n for n in cam1 if n[:15] in ("20230504_131116", "20230527_214315", "20240509_210037")]
setB = [n for n in cam1 if n not in setA]
UW = "20240520_070750.jpg"
Kavg = np.mean([res[n] for n in cam1], axis=0)
Aavg, Bavg = np.mean([res[n] for n in setA], axis=0), np.mean([res[n] for n in setB], axis=0)
rAB, zAB = P.correlate(Aavg, Bavg); rAU, zAU = P.correlate(Aavg, res[UW])
print("avg3 vs avg3:", round(rAB, 4), round(zAB, 1), "| avg3 vs ultrawide:", round(rAU, 4), round(zAU, 1))
cmap_pts = [(0.0, (13, 13, 12)), (0.45, (120, 70, 18)), (0.8, (242, 169, 59)), (1.0, (255, 247, 228))]
def cmap_amber(v):
    v = np.clip(v, 0, 1); xs = [s[0] for s in cmap_pts]
    return np.stack([np.interp(v, xs, [s[1][c] for s in cmap_pts]) for c in range(3)], -1).astype(np.uint8)
def density(a, b, size=420, lim=4.0, bins=130):
    a = ((a - a.mean()) / a.std()).ravel(); b = ((b - b.mean()) / b.std()).ravel()
    H, _, _ = np.histogram2d(a, b, bins=bins, range=[[-lim, lim], [-lim, lim]])
    v = np.log1p(H); v = v / v.max()
    return Image.fromarray(cmap_amber(v[::-1] ** 1.3)).resize((size, size), Image.BICUBIC)
A["dens_AB"], A["dens_AU"] = density(Aavg, Bavg), density(Aavg, res[UW])
A["fp_stats"] = dict(rAB=rAB, zAB=zAB, rAU=rAU, zAU=zAU, setA=setA, setB=setB, UW=UW, cam1=cam1)
c0 = 700
K16 = Kavg[c0:c0 + 16, c0:c0 + 16]; A["K16"] = K16 / Kavg.std()
A["K64"] = Kavg[c0:c0 + 64, c0:c0 + 64] / Kavg.std()
def gray_img(a, gain=3.0, size=None, resample=Image.NEAREST):
    v = 128 + a / (gain * a.std()) * 127
    im = Image.fromarray(np.clip(v, 0, 255).astype(np.uint8)).convert("RGB")
    return im.resize(size, resample) if size else im
A["K64_img"] = gray_img(A["K64"], 2.2, (384, 384))
A["A64_img"] = gray_img(Aavg[c0:c0 + 64, c0:c0 + 64], 2.2, (384, 384))
A["B64_img"] = gray_img(Bavg[c0:c0 + 64, c0:c0 + 64], 2.2, (384, 384))
A["U64_img"] = gray_img(res[UW][c0:c0 + 64, c0:c0 + 64], 2.2, (384, 384))

# ---- S3: noise extraction on three real photos (girl, flowers, red shirt)
def crop_item(key, x0, y0, w=192, h=130, pad=40):
    im = DISP_FULL[key]
    g = np.asarray(im.convert("L"), dtype=float)
    win = g[y0 - pad:y0 + h + pad, x0 - pad:x0 + w + pad]
    blur_g = gaussian_filter(win, 1.5); raw = (win - blur_g)[pad:-pad, pad:-pad]
    rs = P.noise_residual(win)[pad:-pad, pad:-pad]
    rgb = np.asarray(im.convert("RGB"), dtype=float)[y0 - pad:y0 + h + pad, x0 - pad:x0 + w + pad]
    blur_rgb = np.stack([gaussian_filter(rgb[..., c], 1.5) for c in range(3)], -1)[pad:-pad, pad:-pad]
    ph = rgb[pad:-pad, pad:-pad]
    up = lambda a, mode="RGB": Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGB").resize((w * 2, h * 2), Image.NEAREST)
    sc = 128 + rs / (3.2 * rs.std()) * 127
    full = cover(im, 290, 216)
    s = max(290 / im.width, 216 / im.height); ox = (im.width * s - 290) / 2; oy = (im.height * s - 216) / 2
    rect = (x0 * s - ox, y0 * s - oy, (x0 + w) * s - ox, (y0 + h) * s - oy)
    hist, edges = np.histogram(raw.ravel(), bins=33, range=(-8, 8))
    return dict(photo=up(ph), blur=up(blur_rgb), res=up(np.repeat(sc[..., None], 3, 2)), full=full, rect=rect,
                sigma=float(raw.std()), hist=hist, mean_lum=float(g.mean()), size=im.size)
DISP_FULL = {"flowers": load_disp(os.path.join(HERE, "originals", "20230505_125151.jpg"), 4000),
             "red": load_disp(os.path.join(HERE, "originals", "red_shirt.png"), 3000)}
print("sizes", {k: v.size for k, v in DISP_FULL.items()})
A["ext"] = {"flowers": crop_item("flowers", 1000, 1500), "red": crop_item("red", 300, 760)}
for k, v in A["ext"].items(): print(k, "noise sigma", round(v["sigma"], 2), "mean lum", round(v["mean_lum"], 1))

# ---- S4: cross-correlation peaks (same camera vs different camera)
def xcorr_win(a, b, half=50):
    a = a - a.mean(); b = b - b.mean(); n = a.shape[0]
    cc = np.fft.irfft2(np.fft.rfft2(a) * np.conj(np.fft.rfft2(b)), s=a.shape) / (np.linalg.norm(a) * np.linalg.norm(b))
    floor = float(cc.std()); cs = np.fft.fftshift(cc); c = n // 2
    return cs[c - half:c + half + 1, c - half:c + half + 1], floor
PAIR_SAME = ("20240518_203321.jpg", "20240520_131048.jpg"); PAIR_DIFF = ("20240518_203321.jpg", UW)
A["xc"] = {}
for key, (a, b) in (("same", PAIR_SAME), ("diff", PAIR_DIFF)):
    win, floor = xcorr_win(res[a], res[b])
    r, z = P.correlate(res[a], res[b])
    A["xc"][key] = dict(win=win, floor=floor, r=r, z=z, a=a, b=b, peak=float(win[50, 50]))
    print("xcorr", key, "r", round(r, 4), "z", round(z, 1), "peak/floor", round(win[50, 50] / floor, 1), "max-off-peak/floor", round(np.abs(np.delete(win.ravel(), 50 * 101 + 50)).max() / floor, 1))
A["res_prev"] = {n: gray_img(P.center(res[n], 256), 3.2, (230, 230), Image.BICUBIC) for n in (PAIR_SAME + (UW,))}

# ---- S5: AI portraits vs camera photos: noise maps + best-match z (AI images have no sensor)
ai_names = sorted([n for i, n in enumerate(names) if A["is_ai"][i]], key=lambda n: truth[n]["label"])
cam_ex = ["20240509_210037.jpg", "20230505_125151.jpg", "20240518_203321.jpg"]
tile = lambda n: gray_img(P.center(res[n], 256), 3.2, (300, 300), Image.BICUBIC)
A["ai"] = dict(cam=[dict(name=n, i=names.index(n), img=tile(n), z=float(A["zbest"][names.index(n)])) for n in cam_ex],
               ai=[dict(name=n, i=names.index(n), img=tile(n), z=float(A["zbest"][names.index(n)])) for n in ai_names[:3]],
               z_ai=[float(A["zbest"][names.index(n)]) for n in ai_names], peak_ai=[float(peak[names.index(n)]) for n in ai_names])
print("AI best z", [round(z, 1) for z in A["ai"]["z_ai"]], "AI peaks", [round(p, 1) for p in A["ai"]["peak_ai"]])

# ---- fingerprint ridge overlay for the title scene
def make_fingerprint(size=560, seed=11):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    dx, dy = (x - size / 2) / (size * 0.36), (y - size / 2 - 20) / (size * 0.44)
    r = np.sqrt(dx ** 2 + dy ** 2) + 1e-4; th = np.arctan2(dy, dx)
    warp = gaussian_filter(rng.normal(size=(size, size)), 40); warp /= warp.std()
    phase = 34 * r + 3.2 * np.sin(2 * th + 1.0) * r * np.exp(-r) + 1.2 * warp + 1.0 * np.sin(5 * th) * r ** 2
    return np.clip(1 - np.abs(np.sin(phase)) * 2.6, 0, 1).astype(np.float32), r.astype(np.float32)
A["fp_line"], A["fp_r"] = make_fingerprint()

pickle.dump(A, open(os.path.join(HERE, "assets.pkl"), "wb"), protocol=4)
print("assets.pkl written", os.path.getsize(os.path.join(HERE, "assets.pkl")) // 1024, "KB")
