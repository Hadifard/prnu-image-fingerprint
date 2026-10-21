#-----------------------------------
#  IMAGE FINGERPRINT PRNU PROJECT 
#
# Author:
# hadi Sarhangi Fard
#
#-----------------------------------
"""Runs the whole project, step by step (about 40 min on a 2-core CPU, most of it Part 2).

  python run_all.py              data -> detector -> assets -> Part 2 -> both videos
  python run_all.py --no-video   everything except the (slow) video renders
"""
import os, shutil, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); PY = sys.executable
def run(args, out=None):
    print("\n$ python", *args, flush=True)
    if out:
        with open(os.path.join(HERE, out), "w") as fh: subprocess.check_call([PY, *args], cwd=HERE, stdout=fh)
        print(open(os.path.join(HERE, out)).read())
    else: subprocess.check_call([PY, *args], cwd=HERE)
if __name__ == "__main__":
    run(["download_data.py"])
    run(["make_demo39.py"])                                                          # 39-image demo folder + truth39.json
    run(["prnu_detect.py", "--folder", "photos39", "--out", "result.png"], out="run_output.txt")
    run(["orient_nikon.py"])                                                         # rotation direction of portrait Nikon TIFFs
    run(["prep_assets.py"]); run(["prep_extra.py"])                                  # numbers and images of Part 1
    run(["prep_v2_orig.py"]); run(["curves.py"])                                     # Part 2 on the 4 phones
    shutil.copy(os.path.join(HERE, "assets_v2.pkl"), os.path.join(HERE, "assets_v2_backup.pkl"))
    run(["prep_v2b.py"])                                                             # Part 2 extended to 7 cameras (3 Nikon bodies)
    if "--no-video" not in sys.argv:
        run(["render_video.py", "full"]); run(["render_teaser.py", "full"])
