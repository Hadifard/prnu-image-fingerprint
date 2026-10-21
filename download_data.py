"""Downloads and unpacks the input data from the GitHub release into the folders the scripts expect.
   python download_data.py          (needs internet; ~400 MB)"""
import io, os, shutil, urllib.request, zipfile
BASE = "https://github.com/Hadifard/PRNU_project_repo/releases/download/v1.0.0/"
ZIPS = {"originals.zip": "originals", "raw_extract.zip": "raw_extract", "ai_new.zip": "ai_new", "tamper_set.zip": "tamper_set",
        "photos25.zip": "photos25", "final_series_imgs.zip": "final_series"}
FILES = {"Ava.and.Sophi.jpg": "Ava.and.Sophi.jpg", "fingerprint.on.lens.jpg": "fingerprint.on.lens.jpg",
         "Examples-of-device-fingerprints-PRNU-top-and-model-fingerprints-noiseprint-bottom.webp": "Examples-fingerprints.webp",
         "Apple.iPhone.5c.jpg": "phone_iphone5c.jpg", "Samsung.Galaxy.S3.Mini.jpg": "phone_s3mini.jpg"}
def get(name):
    print("downloading", name, flush=True)
    return urllib.request.urlopen(BASE + name).read()
for z, folder in ZIPS.items():
    if os.path.isdir(folder) and os.listdir(folder): continue
    tmp = folder + "_tmp"; zipfile.ZipFile(io.BytesIO(get(z))).extractall(tmp)
    inner = [os.path.join(tmp, x) for x in os.listdir(tmp)]
    src = inner[0] if len(inner) == 1 and os.path.isdir(inner[0]) else tmp         # some zips contain one top-level folder
    shutil.move(src, folder); shutil.rmtree(tmp, ignore_errors=True)
for name, dst in FILES.items():
    if not os.path.exists(dst): open(dst, "wb").write(get(name))
print("done")
