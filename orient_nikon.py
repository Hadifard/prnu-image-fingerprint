"""Finds the rotation direction (k=1 or 3) of every portrait Nikon TIFF so that all photos of one body share one sensor frame."""
import json, os, sys, time
import numpy as np
import prnu_v2 as P
HERE = os.path.dirname(os.path.abspath(__file__))
truth = json.load(open(os.path.join(HERE, "truth39.json")))
T0 = time.time()
groups = {}
for fn, v in truth.items():
    if v["group"].startswith("cam_nikon"): groups.setdefault(v["group"], []).append(fn)
out = {}
for grp, fl in sorted(groups.items()):
    G = {f: P.load_gray(os.path.join(HERE, "photos39", f)) for f in fl}
    land = [f for f in fl if G[f].shape[0] <= G[f].shape[1]]; port = [f for f in fl if f not in land]
    Nl = {f: P.extract_noise(G[f]) for f in land}
    K0 = P.estimate_fingerprint([G[f] for f in land], [Nl[f] for f in land]); ref = P.Reference(K0)
    # sanity: leave-one-out PCE of the landscape photos themselves
    info = {}
    for f in land:
        rest = [x for x in land if x != f]
        if rest:
            Kr = P.Reference(P.estimate_fingerprint([G[x] for x in rest], [Nl[x] for x in rest]))
            info[f] = dict(k=0, pce=float(P.pce_of(P.xcorr(Nl[f] * G[f], Kr), (1, 1))[0]))
        else: info[f] = dict(k=0, pce=None)
    for f in port:
        best = None
        for k in (1, 3):
            g = np.ascontiguousarray(np.rot90(G[f], k)); n = P.extract_noise(g)
            if g.shape != ref.shape: continue
            pce = float(P.pce_of(P.xcorr(n * g, ref), (1, 1))[0])
            info.setdefault(f, {})[f"pce_k{k}"] = pce
            if best is None or pce > best[1]: best = (k, pce)
        info[f].update(k=best[0], pce=best[1])
    out[grp] = info
    print(grp, ref.shape, {f[:8]: (v["k"], round(v["pce"], 1) if v["pce"] else None, {a: round(b, 1) for a, b in v.items() if a.startswith("pce_")}) for f, v in info.items()}, f"[{time.time()-T0:.0f}s]", flush=True)
json.dump(out, open(os.path.join(HERE, "nikon_orient.json"), "w"), indent=1)
