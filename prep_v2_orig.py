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
A33 = ["20230504_131116", "20230505_125151", "20230527_214315", "20240509_210037", "20240518_203321", "20240520_131048"]
CAMS = {"A33": [O + f"/{m}.jpg" for m in A33], "A40": [R + "/A40_00.jpg", R + "/A40_02.jpg"],
        "S7": [R + "/S7_00.jpg", R + "/S7_03.jpg", R + "/S7_04.jpg"], "NOKIA": [R + "/nokia_01.jpg", R + "/nokia_02.jpg"]}
LABEL = {"A33": "Samsung A33", "A40": "Samsung A40", "S7": "Samsung S7", "NOKIA": "Nokia 2.2"}

def gray_landscape(p):
    g = P.load_gray(p)
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
names = list(CAMS); M = np.zeros((4, 4))
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
# user's own manipulated photos (ground truth unknown -> detector decides)
for f, d in (("20240421_174950.jpg", "wisteria · 16:9 mode (4000x2250)"), ("20240520_070750.jpg", "corridor · ultra-wide lens"),
             ("girl.jpg", "girl · resized copy 2576x1658"), ("cat_plane.jpg", "cat at plane window · resized 1437x2576"),
             ("man.jpg", "portrait · 640x640 enhanced"), ("red_shirt.png", "self-portrait · edited PNG")):
    tests.append((os.path.join(O, f), None, d, f, None))
tests.sort(key=lambda t: (t[1] is None, os.path.basename(t[0])))
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

RES = []
for (f, truth, desc, key, region) in tests:
    g = P.load_gray(f); row = dict(file=os.path.basename(f), truth=truth, desc=desc, key=key, region=region, shape=g.shape, by_cam={})
    for cam in CAMS:
        ref = ref_loo(cam, CAMS[cam][[os.path.basename(x) for x in CAMS[cam]].index(key + ".jpg")] if (truth == cam and False) else None) if False else None
        ref = REFS[cam]
        # controlled tests: remove the source photo from its own camera's reference
        if truth == cam:
            srcf = src[key]; ref = ref_loo(cam, srcf)
        top = identify(g, ref)
        full = pce_full(g, ref, top) if (top["pce"] > 150 and top.get("ok", True)) else dict(pce=float(top["pce"]), r=0.0, shift=top["shift"], scale=top["scale"], rot=top["rot"], gshape=g.shape)
        row["by_cam"][cam] = dict(pce=float(full["pce"]), r=float(full.get("r", 0)), scale=float(full["scale"]), rot=int(full["rot"]), shift=[int(full["shift"][0]), int(full["shift"][1])])
        if "c" in full: full["_c"] = full.pop("c"); row.setdefault("_keep", {})[cam] = (full, ref)
        log(row["file"], cam, "PCE", round(full["pce"], 1), "s", round(full["scale"], 4), "rot", full["rot"])
    best_cam = max(row["by_cam"], key=lambda c: row["by_cam"][c]["pce"]); row["best"] = best_cam
    row["verdict"] = best_cam if row["by_cam"][best_cam]["pce"] > 100 else "no match"
    RES.append(row)

# ---------------------------------------------------------------- untouched real photos (leave-one-out) = genuine vs impostor
GEN, IMP = [], []
for cam, fl in CAMS.items():
    for f in fl:
        rest = [x for x in fl if x != f]
        if not rest: continue
        ref = P.Reference(K_of(rest)); pce = P.pce_of(P.xcorr(N[f].astype(np.float32) * G[f], ref), (1, 1))[0] if G[f].shape == ref.shape else None
        if pce is not None: GEN.append((os.path.basename(f), cam, float(pce)))
    for other in CAMS:
        if other == cam: continue
        for f in fl:
            ref = REFS[other]
            if G[f].shape == ref.shape:
                IMP.append((os.path.basename(f), cam, other, float(P.pce_of(P.xcorr(N[f].astype(np.float32) * G[f], ref), (1, 1))[0])))
            else:
                h = min(G[f].shape[0], ref.shape[0]); w = min(G[f].shape[1], ref.shape[1])
                m = P.match(G[f][:h, :w], ref, 1.0, 0, None); IMP.append((os.path.basename(f), cam, other, float(m["pce"])))
log("genuine/impostor", len(GEN), len(IMP))
A["gen"], A["imp"] = GEN, IMP

# ---------------------------------------------------------------- display data from kept matches: surfaces, curves, tamper maps
def cwin(c, idx, null, half=50):
    ys = (np.arange(-half, half + 1) + idx[0]) % c.shape[0]; xs = (np.arange(-half, half + 1) + idx[1]) % c.shape[1]
    return (c[np.ix_(ys, xs)] / null).astype(np.float32)

def row_of(name): return next(r for r in RES if r["file"] == name)
wis = row_of("20240421_174950.jpg")
full_w, ref_w = wis["_keep"]["A33"]; c = full_w["_c"]
A["surf_wis_match"] = cwin(c, np.unravel_index(np.argmax(c[:751, :1]), c[:751, :1].shape) if False else (full_w["shift"][0] % c.shape[0], full_w["shift"][1] % c.shape[1]), full_w["null"])
# wrong-camera surface for the same photo (S7): take its own peak
full_s, _ = wis["_keep"].get("S7", (None, None)) if "S7" in wis.get("_keep", {}) else (None, None)
if full_s is None:
    g = P.load_gray(os.path.join(O, "20240421_174950.jpg")); mm = P.match(g[:2250, :3800], REFS["S7"], 1.0, 0, None); c2 = mm["c"]
    full_s = dict(_c=c2, shift=mm["shift"], null=mm["null"])
c2 = full_s["_c"]; A["surf_wis_wrong"] = cwin(c2, (full_s["shift"][0] % c2.shape[0], full_s["shift"][1] % c2.shape[1]), full_s["null"])
A["wis_geom"] = dict(shift=[int(v) for v in full_w["shift"]], gshape=[int(v) for v in full_w["gshape"]], ksh=[int(v) for v in ref_w.shape], pce=float(full_w["pce"]))
log("surfaces")

# PCE vs scale curve: resized copy T2 (sharp) and the girl photo (flat)
def curve(g, ref, rot, smin, smax, step, patch=768):
    out = []
    for s in np.arange(smin, smax, step):
        m = P.match(g, ref, float(s), rot, patch); out.append((float(s), float(m["pce"])))
    return out
t2 = row_of("T2_resize.jpg"); ref2 = ref_loo("A33", src["flowers2"]); g2 = P.load_gray(os.path.join(TS, "T2_resize.jpg"))
b = t2["by_cam"]["A33"]; A["curve_T2"] = dict(pts=curve(g2, ref2, b["rot"], 0.60, 0.69, 0.0006), true=b["scale"], rot=b["rot"])
A["curve_girl"] = dict(pts=[], rot=0)   # placeholder: curves.py recomputes the real curve (cache/sweep_girl_A33.json was not shipped)
log("curves")

# tamper localisation maps for T4 (splice) and T5 (retouch) + T3/T2 clean reference maps
A["maps"] = {}
for nm in ("T4_splice.jpg", "T5_retouch.jpg", "T2_resize.jpg"):
    r = row_of(nm); full, ref = r["_keep"]["A33"]; g = P.load_gray(os.path.join(TS, nm))
    hyp = dict(rot=full["rot"], scale=full["scale"], shift=full["shift"])
    grid, rmap, gg, org = P.local_map(g, ref, hyp, win=128, stride=32)
    th = cv2.resize(gg, (gg.shape[1] // 8, gg.shape[0] // 8), interpolation=cv2.INTER_AREA)
    A["maps"][nm] = dict(rmap=cv2.resize(rmap, (rmap.shape[1] // 8, rmap.shape[0] // 8), interpolation=cv2.INTER_AREA).astype(np.float32),
                         thumb=th.astype(np.float32), region=r["region"], shape=gg.shape, med=float(np.median(rmap)), org=[int(org[0]), int(org[1])])
    rgb_t = np.asarray(Image.open(os.path.join(TS, nm)).convert("RGB")); A["maps"][nm]["rgb"] = np.ascontiguousarray(np.rot90(rgb_t, full["rot"]))[::8, ::8] if False else None
log("maps")

# strip internals and save
for r in RES: r.pop("_keep", None)
A["res"] = RES; A["tests_src"] = {k: os.path.basename(v) for k, v in src.items()}
pickle.dump(A, open(os.path.join(HERE, "assets_v2.pkl"), "wb"), protocol=4)
print(json.dumps([{k: (v if k != "by_cam" else {c: round(d["pce"], 1) for c, d in v.items()}) for k, v in r.items() if k in ("file", "desc", "by_cam", "verdict", "truth")} for r in RES], indent=1))
print("genuine PCE min/med:", min(x[2] for x in GEN), np.median([x[2] for x in GEN]), "| impostor max:", max(x[3] for x in IMP))
print("fp_corr\n", np.round(A["fp_corr"], 4)); print("resid", {k: round(v, 3) for k, v in A["resid"].items() if not hasattr(v, "shape")})
