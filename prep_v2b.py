#-----------------------------------
#  IMAGE FINGERPRINT PRNU PROJECT 
#
# Author:
# hadi Sarhangi Fard
#
#-----------------------------------

"""
prep_v2.py - every number / image of PART 2 (modern PRNU) -> assets_v2.pkl
  * 4 camera fingerprints (A33 main, A40, S7, Nokia 2.2) estimated from real photos
  * controlled tamper set built from real photos (crop, resize+JPEG, rotate+crop+resize, splice, retouch)
  * geometry search (rotation x scale x shift) with PCE, tamper-localisation maps, attribution matrix
Uses only prnu_v2.py functions.
"""
import glob, json, os, pickle, sys, time
import numpy as np, cv2
from PIL import Image
import prnu_v2 as P

HERE = os.path.dirname(os.path.abspath(__file__)); O = os.path.join(HERE, "originals"); R = os.path.join(HERE, "raw_extract")
TS = os.path.join(HERE, "tamper_set"); os.makedirs(TS, exist_ok=True)
T0 = time.time(); log = lambda *a: print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)
NO = json.load(open(os.path.join(HERE, "nikon_orient.json")))          # rotation k of every Nikon photo (found by orient_nikon.py)
TRUTH = json.load(open(os.path.join(HERE, "truth39.json")))
PH39 = os.path.join(HERE, "photos39")
def nik(grp): return sorted(PH39 + "/" + f for f, v in TRUTH.items() if v["group"] == grp)
A33 = ["20230504_131116", "20230505_125151", "20230527_214315", "20240509_210037", "20240518_203321", "20240520_131048"]
CAMS = {"A33": [O + f"/{m}.jpg" for m in A33], "A40": [R + "/A40_00.jpg", R + "/A40_02.jpg"],
        "S7": [R + "/S7_00.jpg", R + "/S7_03.jpg", R + "/S7_04.jpg"], "NOKIA": [R + "/nokia_01.jpg", R + "/nokia_02.jpg"],
        "NIKON_A": nik("cam_nikon_a"), "NIKON_B": nik("cam_nikon_b"), "NIKON_C": nik("cam_nikon_c")}
LABEL = {"A33": "Samsung A33", "A40": "Samsung A40", "S7": "Samsung S7", "NOKIA": "Nokia 2.2",
         "NIKON_A": "Nikon 3008×2000", "NIKON_B": "Nikon 4288×2848", "NIKON_C": "Nikon 4928×3264"}

def gray_landscape(p):
    g = P.load_gray(p)
    if os.path.basename(p) in {f for grp in NO.values() for f in grp}:              # Nikon TIFFs: portrait shots were stored rotated either way
        k = next(v[os.path.basename(p)]["k"] for v in NO.values() if os.path.basename(p) in v)
        g = np.ascontiguousarray(np.rot90(g, k)) if k else g
    else:
        g = np.ascontiguousarray(np.rot90(g)) if g.shape[0] > g.shape[1] else g
    return g.astype(np.float16)          # memory: 13 full frames + noise maps must fit in ~2 GB (precision 0.1 gray level)

# ---------------------------------------------------------------- residuals of all reference photos
G, N = {}, {}
for cam, fl in CAMS.items():
    for f in fl:
        G[f] = gray_landscape(f); N[f] = P.extract_noise(G[f]).astype(np.float16)
    log("residuals", cam)
def K_of(files):
    return P.estimate_fingerprint([G[f] for f in files], [N[f] for f in files])
KFULL = {c: K_of(fl) for c, fl in CAMS.items()}
REFS = {c: P.Reference(K) for c, K in KFULL.items()}
log("fingerprints done")
A = dict(cams=list(CAMS), label=LABEL, n_ref={c: len(f) for c, f in CAMS.items()}, shapes={c: KFULL[c].shape for c in CAMS})

# ---------------------------------------------------------------- fingerprint pixel patterns (display)
c0 = (700, 1400)
A["fp"] = {}
for c, K in KFULL.items():
    y, x = c0[0], c0[1]; t = K[y:y + 64, x:x + 64] / K.std()
    A["fp"][c] = dict(p64=t.astype(np.float32), std=float(K.std()), vals=(K[y:y + 12, x:x + 12] / K.std()).astype(np.float32))
# cross-correlation between the 4 fingerprints (should be ~0)
names = list(CAMS); M = np.zeros((len(CAMS), len(CAMS)))
for i, a in enumerate(names):
    for j, b in enumerate(names):
        ka, kb = KFULL[a], KFULL[b]; h, w = min(ka.shape[0], kb.shape[0]), min(ka.shape[1], kb.shape[1])
        x, y = ka[:h, :w].ravel(), kb[:h, :w].ravel(); M[i, j] = float(np.corrcoef(x, y)[0, 1])
A["fp_corr"] = M

# ---------------------------------------------------------------- old vs new residual on a real crop
g = G[CAMS["A33"][1]].astype(np.float32); y0, x0, h_, w_ = 1500, 1000, 130, 192; pad = 40
win = g[y0 - pad:y0 + h_ + pad, x0 - pad:x0 + w_ + pad]
from scipy.ndimage import gaussian_filter
old = (win - gaussian_filter(win, 1.5))[pad:-pad, pad:-pad]; new = P.extract_noise(win)[pad:-pad, pad:-pad]
Kc = KFULL["A33"][y0:y0 + h_, x0:x0 + w_]
rr = lambda a, b: float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
A["resid"] = dict(old=old.astype(np.float32), new=new.astype(np.float32), r_old=rr(old / (old.std()), Kc * g[y0:y0 + h_, x0:x0 + w_]),
                  r_new=rr(new, Kc * g[y0:y0 + h_, x0:x0 + w_]), s_old=float(old.std()), s_new=float(new.std()))
log("resid compare", A["resid"]["r_old"], A["resid"]["r_new"])

# ---------------------------------------------------------------- controlled tamper set (known ground truth)
def save_jpg(arr, name, q):
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(os.path.join(TS, name), quality=q); return os.path.join(TS, name)
def rgb(p):
    im = np.asarray(Image.open(p).convert("RGB")); return np.ascontiguousarray(np.rot90(im)) if im.shape[0] > im.shape[1] else im
def lanczos(im, s): return np.asarray(Image.fromarray(im).resize((int(round(im.shape[1] * s)), int(round(im.shape[0] * s))), Image.LANCZOS))
src = {k: CAMS["A33"][i] for k, i in dict(flowers2=1, boiler=2, globe=3, gate=4, catbed=5).items()}
def nikfile(grp, short): return next(f for f in CAMS[grp] if short in f)
src.update(nikA=nikfile("NIKON_A", "IMG_1246"), nikB=nikfile("NIKON_B", "IMG_1062"), nikC=nikfile("NIKON_C", "IMG_1196"))
tests = []     # (file, camera truth, description, source key, ground-truth region or None)
im = rgb(src["gate"]); tests.append((save_jpg(im[900:2000, 1500:3100], "T1_crop.jpg", 92), "A33", "crop 1600x1100", "gate", None))
im = rgb(src["flowers2"]); tests.append((save_jpg(lanczos(im, 0.644), "T2_resize.jpg", 85), "A33", "resize x0.644 + JPEG q85", "flowers2", None))
im0 = np.rot90(rgb(src["globe"])).copy(); im0 = im0[500:2700, 400:2100]
tests.append((save_jpg(lanczos(im0, 0.8), "T3_rot_crop_resize.jpg", 88), "A33", "rotate 90 + crop + resize x0.8 + JPEG q88", "globe", None))
tests.append((save_jpg(lanczos(im0, 0.5), "T6_heavy.jpg", 80), "A33", "rotate + crop + resize x0.5 + q80 (too much)", "globe", None))
im = rgb(src["boiler"]).copy(); sp = rgb(CAMS["S7"][1]); py, px, sz = 1100, 1700, 700
im[py:py + sz, px:px + sz] = sp[1000:1000 + sz, 1500:1500 + sz]
tests.append((save_jpg(im, "T4_splice.jpg", 90), "A33", "700x700 patch pasted from another camera (S7)", "boiler", (py, px, sz, sz)))
im = rgb(src["catbed"]).copy(); msk = np.zeros(im.shape[:2], np.uint8); msk[1000:1500, 1500:2100] = 255
im = cv2.inpaint(im, msk, 5, cv2.INPAINT_TELEA)
tests.append((save_jpg(im, "T5_retouch.jpg", 90), "A33", "600x500 area removed + inpainted", "catbed", (1000, 1500, 500, 600)))
def rgb_k(p, k):
    im = np.asarray(Image.open(p).convert("RGB")); return np.ascontiguousarray(np.rot90(im, k)) if k else im
def kof(p): return next(v[os.path.basename(p)]["k"] for v in NO.values() if os.path.basename(p) in v)
im = rgb_k(src["nikA"], kof(src["nikA"])); tests.append((save_jpg(im[300:1500, 400:2200], "X1_nikonA_crop.jpg", 92), "NIKON_A", "Nikon 3008×2000 · crop 1800x1200", "nikA", None))
im = rgb_k(src["nikB"], kof(src["nikB"])); tests.append((save_jpg(lanczos(im, 0.7), "X2_nikonB_resize.jpg", 85), "NIKON_B", "Nikon 4288×2848 · resize x0.7 + JPEG q85", "nikB", None))
# user's own manipulated photos (ground truth unknown -> detector decides)
for f, d in (("20240421_174950.jpg", "wisteria · 16:9 mode (4000x2250)"), ("20240520_070750.jpg", "corridor · ultra-wide lens"),
             ("girl.jpg", "girl · resized copy 2576x1658"), ("cat_plane.jpg", "cat at plane window · resized 1437x2576"),
             ("man.jpg", "portrait · 640x640 enhanced"), ("red_shirt.png", "self-portrait · edited PNG")):
    tests.append((os.path.join(O, f), None, d, f, None))
tests.sort(key=lambda t: (t[1] is None, os.path.basename(t[0])[0] == 'X', os.path.basename(t[0])))
log("tamper set written")

# refs without the source photo (leave-one-out) for controlled tests
def ref_loo(cam, src_file):
    fl = [f for f in CAMS[cam] if f != src_file]; return P.Reference(K_of(fl)) if fl else None

def pce_full(g, ref, hyp):
    """Full-frame PCE + correlation r at a hypothesis."""
    m = P.match(g, ref, hyp["scale"], hyp["rot"], None)
    if "c" not in m: return dict(pce=float(hyp["pce"]), r=0.0, shift=hyp["shift"], scale=hyp["scale"], rot=hyp["rot"], gshape=(0, 0), null=1.0)
    c = m.pop("c")
    prod, gg, Wn, _ = P.test_prod(g, hyp["scale"], hyp["rot"], None)
    y, x = m["shift"]; Kc = ref.K[y:y + gg.shape[0], x:x + gg.shape[1]]
    r = float((Wn * gg * Kc).sum() / (np.linalg.norm(Wn) * np.linalg.norm(gg * Kc) + 1e-9)) if Kc.shape == gg.shape else 0.0
    m["r"] = r; m["c"] = c; m["gshape"] = gg.shape; m["null"] = float(m["null"]); return m

SCALES = [1.0, 0.8, 0.644, 0.5]          # common resize factors (long side 2576 px = 0.644) - tried on every rotation
def identify(g, ref):
    """rotation x scale scan with a large patch (a small patch is too weak after resize + JPEG)."""
    sc = sorted(set(SCALES) | {round(max(g.shape) / ref.shape[1], 4)}, reverse=True)
    return P.search(g, ref, sc, rots=(0, 1, 2, 3), patch=1400)


import pickle as _pk
OLD = _pk.load(open(os.path.join(HERE, "assets_v2_backup.pkl"), "rb"))
OLDROW = {r["file"]: r for r in OLD["res"]}
OLDCAMS = ["A33", "A40", "S7", "NOKIA"]
CACHE = os.path.join(HERE, "cache_v2b.json")
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
def save_cache(): json.dump(cache, open(CACHE, "w"))
RES = []
for (f, truth, desc, key, region) in tests:
    g = None; fn = os.path.basename(f)
    row = dict(file=fn, truth=truth, desc=desc, key=key, region=region, by_cam={})
    old = OLDROW.get(fn)
    if old: row["shape"] = old["shape"]
    for cam in CAMS:
        ck = fn + "|" + cam
        if old and cam in old["by_cam"]: row["by_cam"][cam] = old["by_cam"][cam]; continue
        if ck in cache: row["by_cam"][cam] = cache[ck]; continue
        if g is None: g = P.load_gray(f); row["shape"] = g.shape
        ref = REFS[cam]
        if truth == cam: ref = ref_loo(cam, src[key])
        top = identify(g, ref)
        full = pce_full(g, ref, top) if (top["pce"] > 150 and top.get("ok", True)) else dict(pce=float(top["pce"]), r=0.0, shift=top["shift"], scale=top["scale"], rot=top["rot"], gshape=g.shape)
        d_ = dict(pce=float(full["pce"]), r=float(full.get("r", 0)), scale=float(full["scale"]), rot=int(full["rot"]), shift=[int(full["shift"][0]), int(full["shift"][1])])
        cache[ck] = d_; save_cache(); row["by_cam"][cam] = d_
        log(fn, cam, "PCE", round(d_["pce"], 1), "s", round(d_["scale"], 4), "rot", d_["rot"])
    best_cam = max(row["by_cam"], key=lambda c: row["by_cam"][c]["pce"]); row["best"] = best_cam
    row["verdict"] = best_cam if row["by_cam"][best_cam]["pce"] > 100 else "no match"
    RES.append(row)
log("tests done")
NEW = [c for c in CAMS if c not in OLDCAMS]
GEN = [tuple(x) for x in OLD["gen"]]; IMP = [tuple(x) for x in OLD["imp"]]
for cam in NEW:
    fl = CAMS[cam]
    for f in fl:
        rest = [x for x in fl if x != f]
        if not rest: continue
        ref = P.Reference(K_of(rest))
        if G[f].shape == ref.shape: GEN.append((os.path.basename(f), cam, float(P.pce_of(P.xcorr(N[f].astype(np.float32) * G[f], ref), (1, 1))[0])))
def imp_pce(f, ref):
    if G[f].shape == ref.shape: return float(P.pce_of(P.xcorr(N[f].astype(np.float32) * G[f], ref), (1, 1))[0])
    h = min(G[f].shape[0], ref.shape[0]); w = min(G[f].shape[1], ref.shape[1])
    return float(P.match(G[f][:h, :w], ref, 1.0, 0, None)["pce"])
for cam, fl in CAMS.items():
    for other in CAMS:
        if other == cam or (cam in OLDCAMS and other in OLDCAMS): continue
        for f in fl: IMP.append((os.path.basename(f), cam, other, imp_pce(f, REFS[other])))
log("genuine/impostor", len(GEN), len(IMP))
A = dict(OLD); A.update(cams=list(CAMS), label=LABEL, n_ref={c: len(f) for c, f in CAMS.items()}, shapes={c: KFULL[c].shape for c in CAMS}, fp=A["fp"] if "fp" in A else None)
A["fp"] = dict(OLD["fp"])
c0 = (700, 1400)
for c in NEW:
    K = KFULL[c]; y, x = c0[0], c0[1]; t_ = K[y:y + 64, x:x + 64] / K.std()
    A["fp"][c] = dict(p64=t_.astype(np.float32), std=float(K.std()), vals=(K[y:y + 12, x:x + 12] / K.std()).astype(np.float32))
names_ = list(CAMS); M = np.zeros((len(CAMS), len(CAMS)))
for i, a_ in enumerate(names_):
    for j, b_ in enumerate(names_):
        ka, kb = KFULL[a_], KFULL[b_]; h, w = min(ka.shape[0], kb.shape[0]), min(ka.shape[1], kb.shape[1])
        M[i, j] = float(np.corrcoef(ka[:h, :w].ravel(), kb[:h, :w].ravel())[0, 1])
A["fp_corr"] = M; A["gen"], A["imp"] = GEN, IMP; A["res"] = RES
A["tests_src"] = {k: os.path.basename(v) for k, v in src.items()}
pickle.dump(A, open(os.path.join(HERE, "assets_v2.pkl"), "wb"), protocol=4)
print("genuine min/med:", min(x[2] for x in GEN), np.median([x[2] for x in GEN]), "| impostor max:", max(x[3] for x in IMP))
print({r["file"]: (r["verdict"], {c: round(v["pce"]) for c, v in r["by_cam"].items()}) for r in RES})
