"""
Builds the 39-image demo folder (photos39/) = the 25-image folder + 14 new TIFF photos of three Nikon bodies.
New photos get neutral names IMG_xxxx.tif so the file name gives nothing away. truth39.json holds the ground truth.
Camera groups of the new photos are inferred from the sensor frame size (all 14 files say NIKON CORPORATION in EXIF).
"""
import json, os, shutil
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
SRC25, SER, OUT = os.path.join(HERE, "photos25"), os.path.join(HERE, "final_series"), os.path.join(HERE, "photos39")
GROUP = {(4288, 2848): ("cam_nikon_b", "Nikon · 4288×2848"), (4928, 3264): ("cam_nikon_c", "Nikon · 4928×3264"), (3008, 2000): ("cam_nikon_a", "Nikon · 3008×2000")}
if os.path.exists(OUT): shutil.rmtree(OUT)
shutil.copytree(SRC25, OUT)
truth = json.load(open(os.path.join(HERE, "truth25.json")))
for k, v in truth.items(): v["disp"] = v["disp"].replace("/home/claude/PRNU_project", HERE)
rng = np.random.default_rng(39)
files = sorted(f for f in os.listdir(SER) if f.lower().endswith((".tif", ".tiff")))
nums = sorted(rng.choice(np.arange(1000, 9999), 400, replace=False).tolist())
used = {os.path.splitext(f)[0] for f in os.listdir(OUT)}
nums = [n for n in nums if f"IMG_{n}" not in used][:len(files)]
order = list(rng.permutation(len(files)))
cnt = {}
for k, i in enumerate(order):
    f = files[i]; p = os.path.join(SER, f)
    with Image.open(p) as im: size = tuple(sorted(im.size, reverse=True))
    grp, lab = GROUP[size]; cnt[grp] = cnt.get(grp, 0) + 1
    fn = f"IMG_{nums[k]}.tif"; shutil.copy(p, os.path.join(OUT, fn))
    truth[fn] = dict(kind="real", group=grp, label=f"{lab} · photo {cnt[grp]}", disp=p)
json.dump(truth, open(os.path.join(HERE, "truth39.json"), "w"), indent=1)
print(len(truth), "files;", cnt)
