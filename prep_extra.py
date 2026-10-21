"""Extra real data + charts for the video (adds keys to assets.pkl). Uses the functions of prnu_detect.py."""
import os, pickle, sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import prnu_detect as P
A = pickle.load(open(os.path.join(HERE, "assets.pkl"), "rb"))
names = A["names"]; fs = A["fp_stats"]; cam1 = fs["cam1"]
res = {n: P.noise_residual(P.load_gray(os.path.join(HERE, "photos39", n))) for n in cam1}

# --- fingerprint grows when photos are averaged: match of avg(n photos) against a held-out photo
held = cam1[-1]; rest = cam1[:-1]
steps = []
c0 = 700
for n in (1, 2, 3, 5):
    avg = np.mean([res[x] for x in rest[:n]], axis=0)
    r, z = P.correlate(avg, res[held])
    t = avg[c0:c0 + 64, c0:c0 + 64]; v = 128 + t / (3.0 * t.std()) * 127
    steps.append(dict(n=n, r=float(r), z=float(z), img=Image.fromarray(np.clip(v, 0, 255).astype(np.uint8)).convert("RGB").resize((256, 256), Image.NEAREST)))
A["avg_steps"] = steps; A["held"] = held
print("avg steps z:", [(s["n"], round(s["z"], 1)) for s in steps])

# --- pair z distributions (non-AI pairs)
Z = A["Z"]; ai = A["is_ai"]; N = len(names)
grp = [A["truth"][n]["group"] for n in names]                      # ground truth: which phone took the photo
same, diff = [], []
for i in range(N):
    for j in range(i + 1, N):
        if ai[i] or ai[j]: continue
        (same if grp[i] == grp[j] and grp[i] not in ("none", "ai") else diff).append(float(Z[i, j]))
A["z_same"], A["z_diff"] = same, diff
print("pairs same", len(same), "min z", round(min(same), 1), "| diff", len(diff), "max z", round(max(diff), 1), "| same pairs below z=10:", sum(1 for v in same if v < 10))

# --- 3D cross-correlation surfaces (same scale for both)
amber = LinearSegmentedColormap.from_list("amber", [(0, "#3a260c"), (0.4, "#8a531c"), (0.75, "#f2a93b"), (1, "#fff7e4")])
def surface(win, floor, zmax, path_size=(1280, 960)):
    n = win.shape[0]; h = n // 2
    X, Y = np.meshgrid(np.arange(-h, h + 1), np.arange(-h, h + 1)); Zs = win / floor
    fig = plt.figure(figsize=(path_size[0] / 200, path_size[1] / 200), dpi=200, facecolor="#0d0d0c")
    ax = fig.add_subplot(111, projection="3d", facecolor="#0d0d0c")
    ax.plot_surface(X, Y, Zs, cmap=amber, vmin=-4, vmax=9, rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax.set_zlim(-8, zmax); ax.view_init(elev=27, azim=-58)
    ax.set_axis_off(); ax.set_box_aspect((1, 1, 0.75), zoom=1.18)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.canvas.draw(); im = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB"); plt.close(fig)
    return im
zmax = max(A["xc"]["same"]["win"].max() / A["xc"]["same"]["floor"], 10) * 1.05
A["surf_same"] = surface(A["xc"]["same"]["win"], A["xc"]["same"]["floor"], zmax)
A["surf_diff"] = surface(A["xc"]["diff"]["win"], A["xc"]["diff"]["floor"], zmax)
A["xc_peak"] = {k: float(A["xc"][k]["win"][50, 50] / A["xc"][k]["floor"]) for k in ("same", "diff")}
print("surface zmax", round(zmax, 1), A["xc_peak"])
pickle.dump(A, open(os.path.join(HERE, "assets.pkl"), "wb"), protocol=4)
A["surf_same"].resize((640, 480)).save(os.path.join(HERE, "chk_surf_same.png")); A["surf_diff"].resize((640, 480)).save(os.path.join(HERE, "chk_surf_diff.png"))
print("ok", list(A.keys())[-6:])
