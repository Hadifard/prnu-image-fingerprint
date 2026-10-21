#-----------------------------------
#  IMAGE FINGERPRINT PRNU PROJECT 
#
# Author:
# hadi Sarhangi Fard
#
#-----------------------------------



"""
PRNU video (1920x1080, 16:9 - works on YouTube and LinkedIn).  English subtitles only, no speech, no music.
Every number and image comes from assets.pkl / assets_v2.pkl, computed by prep_assets.py, prep_extra.py, prep_v2.py and curves.py.

  python render_video.py preview 3 14 25     -> PNG stills in ./preview/ (times in animation seconds)
  python render_video.py full                -> PRNU_Fingerprint_1080p.mp4  (resumable: finished chunks are kept)
"""
import json, inspect, os, pickle, re, subprocess, sys, textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
from scipy.ndimage import gaussian_filter
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Token

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prnu_detect as P

W, H, FPS = 1920, 1080, 20
SLOW = 1.5          # playback slow-down (animation seconds -> video seconds)
OUT_NAME = "PRNU_Fingerprint_1080p.mp4"
BG, PANEL, LINE = (13, 13, 12), (22, 22, 21), (58, 57, 53)
TXT, DIM, DARK = (243, 241, 236), (140, 137, 130), (30, 30, 28)
AMBER, GREEN, PINK, RED = (242, 169, 59), (123, 216, 143), (240, 143, 192), (255, 91, 58)
BLUE, PURPLE = (120, 190, 255), (186, 150, 255)
GITHUB = "https://github.com/Hadifard"

A = pickle.load(open(os.path.join(HERE, "assets.pkl"), "rb"))
names = A["names"]; N = len(names)
is_ai = A["is_ai"]; cam = A["cam"]; kind = A["kind"]; verdict = A["verdict"]
Z = A["Z"]; R = A["R"]; peak = A["peak"]; zbest = A["zbest"]
sim = [k == "sim" for k in kind]
IDX = A["idx"]
TEAL, LIME, MAGENTA = (80, 215, 215), (190, 228, 80), (230, 90, 200)
GROUP_COL = {"cam_samsung": AMBER, "cam_sim_a": GREEN, "cam_s7": PINK, "cam_a40": BLUE, "cam_nokia": PURPLE,
             "cam_nikon_a": TEAL, "cam_nikon_b": LIME, "cam_nikon_c": MAGENTA}
GROUP_NAME = {"cam_samsung": "Samsung A33", "cam_sim_a": "simulated sensor", "cam_s7": "Samsung S7", "cam_a40": "Samsung A40", "cam_nokia": "Nokia 2.2",
              "cam_nikon_a": "Nikon 3008×2000", "cam_nikon_b": "Nikon 4288×2848", "cam_nikon_c": "Nikon 4928×3264"}
from collections import Counter
CAMGRP = {c: Counter(A["truth"][names[i]]["group"] for i in range(N) if cam[i] == c).most_common(1)[0][0] for c in sorted(set(cam) - {0})}
CAMCOL = {c: GROUP_COL[g] for c, g in CAMGRP.items()}          # detector camera id -> colour of the camera it really is

# ------------------------------------------------------------------ helpers
_fc = {}
def font(kind_, size):
    k = (kind_, size)
    if k not in _fc:
        path = {"m": "/usr/share/fonts/opentype/inter/Inter-Medium.otf", "s": "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf",
                "b": "/usr/share/fonts/opentype/inter/Inter-Bold.otf", "mono": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                "monob": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"}[kind_]
        _fc[k] = ImageFont.truetype(path, size)
    return _fc[k]

def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def smooth(x): x = clamp(x); return x * x * (3 - 2 * x)
def eout(x): x = clamp(x); return 1 - (1 - x) ** 3
def back(x):
    x = clamp(x); c1 = 1.70158; c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2
def prog(t, a, b): return clamp((t - a) / (b - a))
def lerp(a, b, t): return a + (b - a) * t
def mix(c, a, bg=BG): return tuple(int(bg[i] + (c[i] - bg[i]) * clamp(a)) for i in range(3))
def lerpc(c1, c2, t): return tuple(int(lerp(c1[i], c2[i], clamp(t))) for i in range(3))
def sc_col(i): return RED if is_ai[i] else (CAMCOL.get(cam[i], DIM) if cam[i] else DIM)

def fade_paste(cv, img, xy, a=1.0):
    if a <= 0.004: return
    x, y = int(xy[0]), int(xy[1])
    if a >= 0.996: cv.paste(img, (x, y)); return
    box = (x, y, x + img.width, y + img.height)
    cv.paste(Image.blend(cv.crop(box), img, a), (x, y))

def rgba_paste(cv, sprite, xy, a=1.0):
    if a <= 0.004: return
    if a < 0.996:
        sprite = sprite.copy(); sprite.putalpha(sprite.getchannel("A").point(lambda v: int(v * a)))
    cv.paste(sprite, (int(xy[0]), int(xy[1])), sprite)

def cover(img, w, h, resample=Image.LANCZOS, fx=0.5, fy=0.5):
    s = max(w / img.width, h / img.height)
    im = img.resize((max(w, int(round(img.width * s))), max(h, int(round(img.height * s)))), resample)
    l, t = int((im.width - w) * fx), int((im.height - h) * fy)
    return im.crop((l, t, l + w, t + h))

def tag(d, xy, s, color, size=20, bold=False, a=1.0, anchor="la", bg=BG):
    d.text(xy, s, font=font("monob" if bold else "mono", size), fill=mix(color, a, bg), anchor=anchor)

def txt(d, xy, s, color=TXT, kind_="m", size=36, a=1.0, anchor="la", bg=BG):
    d.text(xy, s, font=font(kind_, size), fill=mix(color, a, bg), anchor=anchor)

def fit_size(s, kind_, size, max_w):
    while size > 12 and font(kind_, size).getlength(s) > max_w: size -= 2
    return size

def frame_rect(d, box, color=LINE, w=2): d.rectangle(box, outline=color, width=w)
def panel(d, box, a=1.0, r=14): d.rounded_rectangle(box, r, fill=mix(PANEL, a), outline=mix(LINE, a), width=2)

def head(d, label, title, t=1.0):
    a = smooth(t / 0.3)
    tag(d, (80, 44), label, DIM, 20, False, a)
    txt(d, (80, 72), title, TXT, "s", 46, a)

def u8rgb(a):
    a = np.clip(a, 0, 255).astype(np.uint8); return Image.fromarray(np.stack([a] * 3, -1))

AMBER_MAP = [(0.0, (13, 13, 12)), (0.45, (120, 70, 18)), (0.8, (242, 169, 59)), (1.0, (255, 247, 228))]
def cmap_amber(v):
    v = np.clip(v, 0, 1); xs = [s[0] for s in AMBER_MAP]
    return np.stack([np.interp(v, xs, [s[1][c] for s in AMBER_MAP]) for c in range(3)], -1).astype(np.uint8)

_sprites = {}
def dot(r, col, ring=None, ss=4):
    """Antialiased filled circle sprite (RGBA)."""
    k = (r, col, ring)
    if k not in _sprites:
        s = (2 * r + 4) * ss; im = Image.new("RGBA", (s, s), (0, 0, 0, 0)); dd = ImageDraw.Draw(im)
        dd.ellipse((2 * ss, 2 * ss, s - 2 * ss, s - 2 * ss), fill=col + (255,))
        if ring: dd.ellipse((2 * ss, 2 * ss, s - 2 * ss, s - 2 * ss), outline=ring + (255,), width=2 * ss)
        _sprites[k] = im.resize((2 * r + 4, 2 * r + 4), Image.LANCZOS)
    return _sprites[k]

_cm = {}
def circ_thumb(img, r, ring_col, ring_w=4, ss=4):
    """Circular thumbnail with colored ring (cached)."""
    k = (id(img), r, ring_col)
    if k not in _cm:
        s = 2 * r + 2 * ring_w; im = Image.new("RGBA", (s * ss, s * ss), (0, 0, 0, 0)); dd = ImageDraw.Draw(im)
        dd.ellipse((0, 0, s * ss - 1, s * ss - 1), fill=ring_col + (255,))
        inner = Image.new("L", (s * ss, s * ss), 0); ImageDraw.Draw(inner).ellipse((ring_w * ss, ring_w * ss, (s - ring_w) * ss - 1, (s - ring_w) * ss - 1), fill=255)
        ph = img.resize((s * ss, s * ss), Image.LANCZOS).convert("RGBA"); im.paste(ph, (0, 0), inner)
        _cm[k] = im.resize((s, s), Image.LANCZOS)
    return _cm[k]

def aaline_layer(lines, size, ss=2):
    """Pre-render antialiased lines [(x0,y0,x1,y1,color,width)] into an RGBA layer."""
    lay = Image.new("RGBA", (size[0] * ss, size[1] * ss), (0, 0, 0, 0)); dd = ImageDraw.Draw(lay)
    for (x0, y0, x1, y1, col, wd) in lines: dd.line((x0 * ss, y0 * ss, x1 * ss, y1 * ss), fill=col + (255,), width=int(wd * ss))
    return lay.resize(size, Image.LANCZOS)

# ------------------------------------------------------------------ captions
CAP_FONT = 46
def build_caption(text):
    text = re.sub(r"\*([.!?:]+)", r"\1*", text)
    words, hl = [], False
    for tok in re.split(r"(\*)", text):
        if tok == "*": hl = not hl; continue
        for w in tok.split(): words.append((w, hl))
    f = font("m", CAP_FONT); sp = f.getlength(" ")
    lines, cur, curw = [], [], 0
    for w, h in words:
        ww = f.getlength(w); g = (sp + (14 if (h or cur[-1][1]) else 0)) if cur else 0
        if cur and curw + g + ww > 1560: lines.append(cur); cur, curw, g = [], 0, 0
        cur.append((w, h, ww)); curw += g + ww
    lines.append(cur)
    sprites = []; lh = 66; y0 = 962 - lh * len(lines) // 2
    for li, ln in enumerate(lines):
        gaps = [sp + (14 if (ln[k][1] or ln[k + 1][1]) else 0) for k in range(len(ln) - 1)]
        lw = sum(w for _, _, w in ln) + sum(gaps); x = (W - lw) / 2
        for k, (w, h, ww) in enumerate(ln):
            pad = 12; sw, sh = int(ww + 2 * pad), 62
            im = Image.new("RGBA", (sw, sh), (0, 0, 0, 0)); dd = ImageDraw.Draw(im)
            dd.text((pad, 31), w, font=f, fill=(135, 206, 250, 255), anchor="lm")
            sprites.append((im, int(x - pad), int(y0 + li * lh)))
            x += ww + (gaps[k] if k < len(gaps) else 0)
    return sprites

def draw_caption(cv, T):
    for (t0, t1, _), spr in zip(CAPS, CAP_SPR):
        if not (t0 <= T < t1 + 0.4): continue
        fo = 1 - prog(T, t1 - 0.1, t1 + 0.3)
        for k, (im, x, y) in enumerate(spr):
            a = smooth((T - t0 - 0.05 * k) / 0.2) * fo
            if a <= 0.01: continue
            yy = int(y + 8 * (1 - a))
            if a < 0.99:
                im = im.copy(); im.putalpha(im.getchannel("A").point(lambda v: int(v * a)))
            cv.paste(im, (x, yy), im)

# ------------------------------------------------------------------ photos from disk (display only)
ORIG = os.path.join(HERE, "originals")
def load_disp(name, maxside=1600):
    im = Image.open(os.path.join(ORIG, name)); im.draft("RGB", (maxside, maxside))
    im = ImageOps.exif_transpose(im).convert("RGB"); im.thumbnail((maxside, maxside), Image.LANCZOS); return im
PH = {k: load_disp(v) for k, v in dict(girl="girl.jpg", cat="cat_plane.jpg", red="red_shirt.png", man="man.jpg", flowers1="20230504_131116.jpg",
                                       flowers2="20230505_125151.jpg", globe="20240509_210037.jpg", gate="20240518_203321.jpg").items()}
FOCUS = dict(girl=(0.55, 0.5), cat=(0.62, 0.55), red=(0.5, 0.35), man=(0.5, 0.45), flowers1=(0.5, 0.5), flowers2=(0.5, 0.5), globe=(0.5, 0.5), gate=(0.5, 0.5))
def pcover(k, w, h): return cover(PH[k], w, h, fx=FOCUS[k][0], fy=FOCUS[k][1])

# ------------------------------------------------------------------ fingerprint ridge overlay
def make_fingerprint(w, h, rx, ry, seed=11, dens=30):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = (x - w / 2) / rx, (y - h / 2 - 10) / ry
    r = np.sqrt(dx ** 2 + dy ** 2) + 1e-4; th = np.arctan2(dy, dx)
    warp = gaussian_filter(rng.normal(size=(h, w)), 45); warp /= warp.std()
    phase = dens * r + 3.2 * np.sin(2 * th + 1.0) * r * np.exp(-r) + 1.2 * warp + 1.0 * np.sin(5 * th) * r ** 2
    line = np.clip(1 - np.abs(np.sin(phase)) * 2.6, 0, 1)
    return line.astype(np.float32), r.astype(np.float32)
TP = 780
FP_LINE, FP_R = make_fingerprint(TP, TP, 300, 340, dens=30)
EFP_LINE, EFP_R = make_fingerprint(400, 440, 150, 190, seed=5, dens=26)

# ------------------------------------------------------------------ sensor grid from the real fingerprint estimate
K64 = A["K64"]; GC, GR, GCELL = 32, 18, 34
KG = K64[:GR, :GC]
def sensor_grid(real):
    im = Image.new("RGB", (GC * GCELL, GR * GCELL), BG); dd = ImageDraw.Draw(im)
    for r_ in range(GR):
        for c_ in range(GC):
            g = int(150 + (np.clip(KG[r_, c_], -2.4, 2.4) * 36 if real else 0))
            dd.rectangle((c_ * GCELL + 1, r_ * GCELL + 1, c_ * GCELL + GCELL - 2, r_ * GCELL + GCELL - 2), fill=(g, g, int(g * 0.96)))
    return im
GRID_UNI, GRID_REAL = sensor_grid(False), sensor_grid(True)
CALL = []
for (r_, c_) in sorted([(r_, c_) for r_ in range(1, GR - 1) for c_ in range(2, GC - 2)], key=lambda p: -abs(KG[p])):
    if all(abs(r_ - r2) + abs(c_ - c2) > 9 for r2, c2 in CALL): CALL.append((r_, c_))
    if len(CALL) == 4: break

# ------------------------------------------------------------------ code chunks (taken from the real script)
def src_lines(func):
    lines, start = inspect.getsourcelines(func)
    return [(start + i, l.rstrip("\n")) for i, l in enumerate(lines)]

# ------------------------------------------------------------------ small scene helpers
def beat_in(cv, lt, dur=0.2):
    """Quick fade-from-background at the start of a beat (hard-cut feel with no flash)."""
    a = smooth(lt / dur)
    if a < 0.999: cv.paste(Image.blend(Image.new("RGB", (W, H), BG), cv, a), (0, 0))

def count_text(v, fmt="{:.0f}"): return fmt.format(v)

def chip(d, x, y, s, col, a=1.0, size=24, pad=18, bg=BG, filled=False):
    w = int(font("monob", size).getlength(s)) + 2 * pad; h = size + 22
    if filled: d.rounded_rectangle((x, y, x + w, y + h), 10, fill=mix(col, a, bg))
    else: d.rounded_rectangle((x, y, x + w, y + h), 10, outline=mix(col, a, bg), width=3)
    d.text((x + w // 2, y + h // 2), s, font=font("monob", size), fill=mix(BG if filled else col, a, bg), anchor="mm")
    return w

def tile(cv, d, img, xy, size, a=1.0, col=LINE, w=2, resample=Image.LANCZOS):
    if img.size != (size, size) and not isinstance(size, tuple): img = img.resize((size, size), resample)
    fade_paste(cv, img, xy, a)
    if a > 0.02: frame_rect(d, (xy[0] - 1, xy[1] - 1, xy[0] + img.width, xy[1] + img.height), mix(col, a), w)

THUMB = A["thumb"]
def thumb(i, s): return THUMB[i].resize((s, s), Image.LANCZOS)
TH30 = [thumb(i, 30) for i in range(N)]

# ================================================================== SCENE 1 · TITLE (0-6)
TSEQ_K = ["gate", "flowers1", "globe", "cat", "red", "girl"]
TSEQ = {k: pcover(k, int(TP * 1.1), int(TP * 1.1)) for k in TSEQ_K}
FP_EDGE = np.clip((1.0 - FP_R) / 0.12, 0, 1)
TITLE_L1, TITLE_L2 = "Every photo", "is a fingerprint"
def sceneTitle(cv, d, t):
    px, py = 1060, 100
    if t < 2.5: k = min(int(t / 0.5), 4); t0 = k * 0.5; span = 0.5
    else: k = 5; t0 = 2.5; span = 3.5
    img = TSEQ[TSEQ_K[k]]; lt = t - t0
    z = 1 + 0.07 * lt / span
    cw = int(img.width / z); l = (img.width - cw) // 2
    ph = img.crop((l, l, l + cw, l + cw)).resize((TP, TP), Image.BILINEAR)
    cv.paste(ph, (px, py))
    p = eout(lt / (0.32 if k < 5 else 1.3)) * 1.3
    grow = np.clip((p - FP_R) / 0.10, 0, 1)
    al = (FP_LINE * FP_EDGE * grow * 0.92 * 255).astype(np.uint8)
    cv.paste((246, 244, 238), (px, py), Image.fromarray(al))
    if lt < 0.22:                                           # quick amber sweep on every cut
        sx = px + int(TP * eout(lt / 0.22)); d.rectangle((sx - 5, py, sx, py + TP), fill=AMBER)
    frame_rect(d, (px - 1, py - 1, px + TP, py + TP), LINE)
    tag(d, (px, py + TP + 14), "REAL PHOTOS · ILLUSTRATED FINGERPRINT", DIM, 18, False, smooth(t / 0.8))
    # left column
    tag(d, (80, 170), "PRNU · THE SENSOR'S PATTERN NOISE", AMBER, 22, True, smooth(t / 0.4))
    for n_, (s, col, t0_) in enumerate(((TITLE_L1, TXT, 0.15), (TITLE_L2, AMBER, 0.4))):
        sz = fit_size(s, "b", 132, 900); a = smooth((t - t0_) / 0.25); off = int(60 * (1 - back((t - t0_) / 0.55)))
        txt(d, (76, 230 + n_ * 150 + off), s, col, "b", sz, a)
    txt(d, (80, 590), "Trace the camera that took it.", TXT, "m", 42, smooth((t - 1.3) / 0.4))
    txt(d, (80, 650), "Images no camera ever shot carry none.", AMBER, "m", 42, smooth((t - 1.8) / 0.4))
    x = 80
    for k_, (s, col) in enumerate((("UNIQUE", AMBER), ("STABLE", GREEN), ("INVISIBLE", PINK))):
        a = back((t - 2.6 - 0.18 * k_) / 0.35)
        if a > 0.02: x += chip(d, x, 770 + int(30 * (1 - clamp(a))), s, col, clamp(a), 26) + 20

# ================================================================== SCENE 2 · WHY (6-20)
def whyA(cv, d, t):
    head(d, "FIG. 01 · THE SENSOR", "Inside every camera", t)
    gx, gy = 80, 190
    rows = int(GR * eout(prog(t, 0.1, 1.7)))
    if rows > 0:
        cv.paste(GRID_UNI.crop((0, 0, GC * GCELL, rows * GCELL)), (gx, gy))
        # light-sweep along the newest row
        if rows < GR: d.rectangle((gx, gy + rows * GCELL - 3, gx + GC * GCELL, gy + rows * GCELL + 1), fill=AMBER)
    frame_rect(d, (gx - 1, gy - 1, gx + GC * GCELL, gy + rows * GCELL), LINE)
    tag(d, (gx, gy + GR * GCELL + 16), "A TINY PATCH OF THE SENSOR · EACH SQUARE IS ONE LIGHT-SENSING PIXEL", DIM, 18, False, smooth(prog(t, 0.6, 1.4)))
    tag(d, (1230, 200), "ONE PHOTO FROM THIS PHONE", DIM, 20, True, smooth(prog(t, 0.2, 0.8)))
    txt(d, (1226, 240), "4000 × 3000", TXT, "b", 80, smooth(prog(t, 0.3, 0.9)))
    v = int(12_000_000 * eout(prog(t, 0.5, 2.2)))
    txt(d, (1226, 350), f"{v:,}", AMBER, "b", 96, smooth(prog(t, 0.4, 0.9)))
    txt(d, (1230, 470), "pixels", TXT, "m", 40, smooth(prog(t, 1.0, 1.6)))
    txt(d, (1230, 530), "each one measures light on its own", DIM, "m", 28, smooth(prog(t, 1.5, 2.1)))
    txt(d, (1230, 580), "and none is exactly like its neighbour.", AMBER, "m", 28, smooth(prog(t, 2.0, 2.6)))

def load_path(path, maxside=1600):
    im = Image.open(path); im.draft("RGB", (maxside, maxside))
    im = ImageOps.exif_transpose(im).convert("RGB"); im.thumbnail((maxside, maxside), Image.LANCZOS); return im
AVA = load_path(os.path.join(HERE, "Ava.and.Sophi.jpg"), 1400)
WHYP_D = 5.2
_ava_crop = AVA.crop((int(AVA.width * 0.50), int(AVA.height * 0.28), int(AVA.width * 0.72), int(AVA.height * 0.46)))   # eye region of the younger girl
def whyP(cv, d, t):
    head(d, "FIG. 02 · FROM PHOTO TO PIXELS", "A photo is a grid of light measurements", t)
    gx, gy, gw, gh = 80, 190, GC * GCELL, GR * GCELL           # same frame as the sensor grid of the next figures
    if t < 1.5:                                                 # 2 s of video: the photo itself
        a = smooth(t / 0.3); im = cover(AVA, gw, gh, fy=0.4); fade_paste(cv, im, (gx, gy), a); frame_rect(d, (gx - 1, gy - 1, gx + gw, gy + gh), mix(LINE, a), 2)
        tag(d, (gx, gy + gh + 16), "A PHOTO: AVA AND SOPHI", DIM, 18, True, a)
    else:
        u = prog(t, 1.5, 3.3)                                   # zoom into the eyes, then coarser and coarser pixels
        z = eout(prog(t, 1.5, 2.3)); full = cover(AVA, gw, gh, fy=0.4)
        l, tp, r_, b_ = lerp(0, AVA.width * 0.49, z), lerp(0, AVA.height * 0.365, z), lerp(AVA.width, AVA.width * 0.67, z), lerp(AVA.height, AVA.height * 0.50, z)
        crop = AVA.crop((int(l), int(tp), int(r_), int(b_))); im = cover(crop, gw, gh, fy=0.5)
        n = int(lerp(gw, GC, smooth(prog(t, 2.0, 3.2))))
        if n < gw: im = im.resize((max(n, GC), max(int(n * gh / gw), GR)), Image.BILINEAR).resize((gw, gh), Image.NEAREST)
        fade_paste(cv, im, (gx, gy), 1.0); frame_rect(d, (gx - 1, gy - 1, gx + gw, gy + gh), LINE, 2)
        if t > 3.2:                                              # the checkerboard: one square = one pixel of the sensor
            g_ = smooth(prog(t, 3.2, 3.9)); cv.paste(Image.blend(im, GRID_REAL, g_), (gx, gy))
        lab = "ZOOM IN · PIXELS GET VISIBLE" if t < 3.2 else "THE CHECKERBOARD: ONE SQUARE = ONE PIXEL OF THE SENSOR"
        tag(d, (gx, gy + gh + 16), lab, AMBER if t >= 3.2 else DIM, 18, True)
    if t > 3.9:
        x = 1230; a = smooth(prog(t, 3.9, 4.5))
        tag(d, (x, 200), "EVERY PIXEL = ONE NUMBER", DIM, 20, True, a)
        txt(d, (x, 240), "light × (1 + ε)", AMBER, "b", 64, a)
        txt(d, (x, 340), "ε is the pixel's own tiny,", DIM, "m", 28, smooth(prog(t, 4.2, 4.8))); txt(d, (x, 380), "fixed flaw: the pattern", DIM, "m", 28, smooth(prog(t, 4.3, 4.9)))
        txt(d, (x, 420), "we are going to extract.", TXT, "m", 28, smooth(prog(t, 4.4, 5.0)))

def whyB(cv, d, t):
    head(d, "FIG. 02 · THE FLAW", "Same light in. Different value out.", t)
    gx, gy = 80, 190
    real = smooth(prog(t, 0.3, 1.3))
    cv.paste(Image.blend(GRID_UNI, GRID_REAL, real), (gx, gy))
    frame_rect(d, (gx - 1, gy - 1, gx + GC * GCELL, gy + GR * GCELL), LINE)
    lab = "REAL FINGERPRINT ESTIMATE · SAMSUNG SM-A336E · 6 PHOTOS AVERAGED" if real > 0.5 else "IDEAL SENSOR · every pixel identical"
    tag(d, (gx, gy + GR * GCELL + 16), lab, AMBER if real > 0.5 else DIM, 18, True)
    for k, (r_, c_) in enumerate(CALL):
        a = back(prog(t, 1.4 + 0.3 * k, 1.8 + 0.3 * k))
        if a <= 0.02: continue
        cx, cy = gx + c_ * GCELL + GCELL // 2, gy + r_ * GCELL + GCELL // 2
        rr = int(22 * clamp(a)); d.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), outline=TXT, width=3)
        right = cx < 700; lx = cx + 64 if right else cx - 64
        d.line((cx + (rr if right else -rr), cy, lx, cy - 30), fill=TXT, width=2)
        bw = 104; bx0 = lx if right else lx - bw
        d.rectangle((bx0, cy - 56, bx0 + bw, cy - 18), fill=TXT)
        tag(d, (bx0 + bw // 2, cy - 37), f"{KG[r_, c_]:+.1f} σ", BG, 22, True, 1, "mm", bg=TXT)
    # same light in / different value out
    bx = 1230; tag(d, (bx, 200), "SAME LIGHT IN", TXT, 20, True, smooth(prog(t, 0.2, 0.7)))
    for k in range(12):
        a = smooth(prog(t, 0.3 + 0.03 * k, 0.7 + 0.03 * k)); h_ = int(90 * a)
        d.rectangle((bx + k * 48, 340 - h_, bx + k * 48 + 36, 340), fill=mix(DIM, a))
    tag(d, (bx, 372), "DIFFERENT VALUE OUT", AMBER, 20, True, smooth(prog(t, 1.2, 1.8)))
    for k in range(12):
        a = smooth(prog(t, 1.3 + 0.04 * k, 1.8 + 0.04 * k)); h_ = int(90 * (1 + 0.2 * np.clip(KG[0, k], -2.4, 2.4)) * a)
        d.rectangle((bx + k * 48, 570 - h_, bx + k * 48 + 36, 570), fill=mix(AMBER, a))
    tag(d, (bx, 586), "differences exaggerated for display", DIM, 16, False, smooth(prog(t, 2.0, 2.6)))
    a = smooth(prog(t, 2.4, 3.0))
    tag(d, (bx, 650), "value = light × (1 + ε)", TXT, 30, True, a)
    tag(d, (bx, 698), "ε: tiny, fixed, unique to each pixel", AMBER, 22, False, a)
    tag(d, (bx, 742), "this is PRNU", TXT, 26, True, smooth(prog(t, 3.2, 3.8)))

A64, B64, U64, K64I = A["A64_img"], A["B64_img"], A["U64_img"], A["K64_img"]
FS_ = A["fp_stats"]
def whyC(cv, d, t):
    head(d, "FIG. 03 · WHY IT IS A FINGERPRINT", "Unique. Stable. Invisible.", t)
    cards = [("UNIQUE", AMBER, (A64, "MAIN LENS"), (U64, "ULTRA-WIDE LENS"), f"z = {FS_['zAU']:.0f}", "Same phone, other lens: no match."),
             ("STABLE", GREEN, (A64, "PHOTOS 1-3"), (B64, "PHOTOS 4-6"), f"z = {FS_['zAB']:.0f}", "Different scenes, same pattern."),
             ("INVISIBLE", PINK, (thumb(IDX["20240518_203321.jpg"], 230), "WHAT YOU SEE"), (K64I, "WHAT IT ADDS · BOOSTED"), "below the eye", "Far too faint to notice in a photo.")]
    for k, (title, col, t1, t2, big_, sub) in enumerate(cards):
        a = back(prog(t, 0.15 + 0.4 * k, 0.65 + 0.4 * k)); ca = clamp(a)
        if ca <= 0.02: continue
        x0 = 80 + k * 610; y0 = 180 + int(40 * (1 - ca)); w_, h_ = 580, 640
        panel(d, (x0, y0, x0 + w_, y0 + h_), ca)
        d.rectangle((x0, y0, x0 + w_, y0 + 8), fill=mix(col, ca))
        txt(d, (x0 + 30, y0 + 34), title, col, "b", 44, ca, bg=PANEL)
        for j, (im, lab) in enumerate((t1, t2)):
            tx = x0 + 40 + j * 270
            fade_paste(cv, im.resize((230, 230), Image.LANCZOS if j == 0 and k == 2 else Image.NEAREST), (tx, y0 + 120), ca)
            frame_rect(d, (tx - 1, y0 + 119, tx + 230, y0 + 351), mix(LINE, ca), 2)
            tag(d, (tx, y0 + 362), lab, DIM, 15, True, ca, bg=PANEL)
        txt(d, (x0 + 30, y0 + 430), big_, col, "b", 66, smooth((t - 0.9 - 0.4 * k) / 0.3), bg=PANEL)
        lines = [sub]
        txt(d, (x0 + 30, y0 + 530), sub, TXT, "m", 26, smooth((t - 1.1 - 0.4 * k) / 0.3), bg=PANEL)

AVG = A["avg_steps"]
def whyD(cv, d, t):
    head(d, "FIG. 04 · AVERAGING", "More photos, stronger fingerprint", t)
    tag(d, (200, 190), "NOISE PATTERN OF AVERAGED PHOTOS  →  MATCH STRENGTH (z) AGAINST A HELD-OUT PHOTO", DIM, 18, True, smooth(t / 0.4))
    x0 = 200; gap = 156
    for k, s in enumerate(AVG):
        a = back(prog(t, 0.15 + 0.3 * k, 0.55 + 0.3 * k)); ca = clamp(a)
        x = x0 + k * (256 + gap)
        fade_paste(cv, s["img"], (x, 250), ca); frame_rect(d, (x - 1, 249, x + 256, 506), mix(AMBER, ca), 3)
        txt(d, (x, 520), f"{s['n']} photo" + ("s" if s["n"] > 1 else ""), TXT, "s", 34, ca)
        v = s["z"] * eout(prog(t, 0.5 + 0.3 * k, 1.5 + 0.3 * k))
        txt(d, (x, 574), f"z = {v:.0f}", AMBER, "b", 60, ca)
        bw = int(256 * clamp(s["z"] / 320) * eout(prog(t, 0.6 + 0.3 * k, 1.6 + 0.3 * k)))
        d.rectangle((x, 660, x + 256, 676), fill=mix(DARK, ca)); d.rectangle((x, 660, x + bw, 676), fill=mix(AMBER, ca))
        if k < len(AVG) - 1: tag(d, (x + 256 + gap // 2, 378), "→", AMBER, 44, True, ca, "mm")
    txt(d, (200, 730), "Random noise differs in every photo and cancels out. The sensor's pattern stays.", DIM, "m", 28, smooth(prog(t, 1.6, 2.2)))

def sceneWhy(cv, d, t):
    P0 = 3.5; P1 = P0 + WHYP_D
    if t < P0: whyA(cv, d, t)
    elif t < P1: whyP(cv, d, t - P0); beat_in(cv, t - P0)
    elif t < P1 + 4: whyB(cv, d, t - P1); beat_in(cv, t - P1)
    elif t < P1 + 7.5: whyC(cv, d, t - P1 - 4); beat_in(cv, t - P1 - 4)
    else: whyD(cv, d, t - P1 - 7.5); beat_in(cv, t - P1 - 7.5)

# ================================================================== SCENE 3 · EXTRACT
EXT = [("flowers", "flowers2", "PHOTO 1 · FLOWERS", "3000 × 4000 px · original from the phone", (0.0, 4.7)),
       ("red", "red", "PHOTO 2 · SELF-PORTRAIT", "864 × 1184 px · edited PNG", (4.7, 9.5))]
EXT_D = EXT[-1][4][1]
EXT_FULL = {k: cover(PH[pk], 560, 417) for k, pk, *_ in EXT}
EXT_T = {k: {n: A["ext"][k][n].resize((352, 238), Image.LANCZOS) for n in ("photo", "blur", "res")} for k, *_ in EXT}
def extractOne(cv, d, t, key, pk, title, sub, n_, tot):
    E = A["ext"][key]
    head(d, f"FIG. 05 · EXTRACT THE NOISE · {n_}/{len(EXT)}", "Photo − blur = noise", min(t + 1, 1))
    fx, fy = 80, 190
    fade_paste(cv, EXT_FULL[key], (fx, fy), eout(t / 0.3)); frame_rect(d, (fx - 1, fy - 1, fx + 560, fy + 417), LINE)
    s = 560 / 290; rc = E["rect"]; box = [fx + rc[0] * s, fy + rc[1] * s, fx + rc[2] * s, fy + rc[3] * s]
    pulse = 0.6 + 0.4 * np.sin(t * 9); d.rectangle(box, outline=mix(AMBER, pulse), width=4)
    txt(d, (fx, fy + 440), title, TXT, "s", 30, smooth(t / 0.3)); tag(d, (fx, fy + 484), sub, DIM, 18, False, smooth(t / 0.3))
    tag(d, (fx, fy + 520), "zoom on the highlighted patch", AMBER, 18, False, smooth(t / 0.3))
    xs = [690, 1080, 1470]; y = 190
    labels = ["PHOTO", "BLURRED  (scene only)", "NOISE  (photo − blur)"]
    for k, nm in enumerate(("photo", "blur", "res")):
        a = eout(prog(t, 0.25 + 0.55 * k, 0.65 + 0.55 * k))
        im = EXT_T[key][nm]
        if nm == "blur" and a < 1: im = Image.blend(EXT_T[key]["photo"], im, a)
        if nm == "res":
            sw = int(352 * eout(prog(t, 1.35, 2.2)))
            if sw > 0:
                cv.paste(im.crop((0, 0, sw, 238)), (xs[k], y))
                if sw < 352: d.rectangle((xs[k] + sw - 3, y, xs[k] + sw, y + 238), fill=AMBER)
            frame_rect(d, (xs[k] - 1, y - 1, xs[k] + 352, y + 238), mix(AMBER, a), 3)
        else:
            fade_paste(cv, im, (xs[k], y), a); frame_rect(d, (xs[k] - 1, y - 1, xs[k] + 352, y + 238), mix(LINE, a), 2)
        tag(d, (xs[k], y + 250), labels[k], AMBER if k == 2 else TXT, 18, True, a)
    tag(d, (1066 + 2, y + 119), "−", AMBER, 44, True, smooth(prog(t, 0.8, 1.2)), "mm"); tag(d, (1456 + 2, y + 119), "=", AMBER, 44, True, smooth(prog(t, 1.3, 1.7)), "mm")
    hx, hy, hw, hh = 690, 520, 1132, 330
    ha = smooth(prog(t, 2.2, 2.8)); panel(d, (hx, hy, hx + hw, hy + hh), ha)
    tag(d, (hx + 24, hy + 18), "DISTRIBUTION OF NOISE VALUES IN THE PATCH", DIM, 18, True, ha, bg=PANEL)
    hist = E["hist"].astype(float); mx = np.sqrt(hist.max()); bw_ = (hw - 80) / len(hist)
    for b in range(len(hist)):
        hgt = int((hh - 130) * np.sqrt(hist[b]) / mx * eout(prog(t, 2.4 + 0.012 * b, 3.0 + 0.012 * b)))
        d.rectangle((hx + 40 + b * bw_ + 2, hy + hh - 50 - hgt, hx + 40 + (b + 1) * bw_ - 2, hy + hh - 50), fill=mix(AMBER, ha, PANEL))
    tag(d, (hx + 40, hy + hh - 38), "-8", DIM, 16, False, ha, bg=PANEL); tag(d, (hx + hw - 40, hy + hh - 38), "+8", DIM, 16, False, ha, "ra", bg=PANEL)
    sg = E["sigma"] * eout(prog(t, 2.6, 3.5))
    txt(d, (hx + hw - 40, hy + 24), f"σ = {sg:.1f}", AMBER, "b", 54, ha, "ra", PANEL)
    tag(d, (hx + hw - 40, hy + 92), "gray levels · the invisible part", DIM, 17, False, ha, "ra", bg=PANEL)
def sceneExtract(cv, d, t):
    for n_, (key, pk, title, sub, (a, b)) in enumerate(EXT):
        if a <= t < b or n_ == len(EXT) - 1:
            extractOne(cv, d, t - a, key, pk, title, sub, n_ + 1, b - a)
            if n_ > 0: beat_in(cv, t - a)
            break

# ================================================================== SCENE 4 · COMPARE (34-46)
DAB, DAU = A["dens_AB"].resize((500, 500), Image.BICUBIC), A["dens_AU"].resize((500, 500), Image.BICUBIC)
def comA(cv, d, t):
    head(d, "FIG. 06 · CORRELATION", "Is the noise the same?", t)
    for k, (im, x, col, lab, sub, r_, z_) in enumerate(((DAB, 250, AMBER, "SAME SENSOR", "photos 1-3  vs  photos 4-6", FS_["rAB"], FS_["zAB"]),
                                                      (DAU, 1170, DIM, "OTHER LENS", "photos 1-3  vs  ultra-wide photo", FS_["rAU"], FS_["zAU"]))):
        a = eout(prog(t, 0.2 + 0.3 * k, 0.7 + 0.3 * k)); y = 200
        tag(d, (x, y - 4), lab, col, 22, True, a); tag(d, (x + 230, y - 2), sub, DIM, 16, False, a)
        rv = int(500 * eout(prog(t, 0.4 + 0.3 * k, 1.6 + 0.3 * k)))
        if rv > 0:
            m = Image.new("L", (500, 500), 0); ImageDraw.Draw(m).ellipse((250 - rv // 2 * 1.45, 250 - rv // 2 * 1.45, 250 + rv // 2 * 1.45, 250 + rv // 2 * 1.45), fill=255)
            cv.paste(im, (x, y + 30), m)
        frame_rect(d, (x - 1, y + 29, x + 500, y + 530), mix(col if k == 0 else LINE, a), 3 if k == 0 else 2)
        if k == 0 and a > 0.5: d.line((x, y + 530, x + 500, y + 30), fill=mix(AMBER, 0.5 * a), width=2)
        tag(d, (x, y + 546), "noise in photo A  →", DIM, 16, False, a)
        v = prog(t, 1.6 + 0.3 * k, 2.4 + 0.3 * k)
        txt(d, (x, y + 580), f"r = {r_ * eout(v):.3f}", col, "b", 54, smooth(prog(t, 1.5 + 0.3 * k, 2.0 + 0.3 * k)))
        txt(d, (x + 290, y + 594), f"z = {z_ * eout(v):.0f}", TXT, "s", 40, smooth(prog(t, 1.5 + 0.3 * k, 2.0 + 0.3 * k)))
    tag(d, (250, 160), "EACH DOT = ONE PIXEL: ITS NOISE IN PHOTO A (x) AND IN PHOTO B (y)", DIM, 17, False, smooth(t / 0.4))

SURF_S = A["surf_same"].resize((880, 660), Image.LANCZOS); SURF_D = A["surf_diff"].resize((880, 660), Image.LANCZOS)
RP = A["res_prev"]
def comB(cv, d, t):
    head(d, "FIG. 07 · CROSS-CORRELATION", "Slide one noise map over the other", t)
    xs = [50, 990]
    for k, (surf, col, lab, a_n, b_n, val, tc) in enumerate(((SURF_S, AMBER, "SAME CAMERA", "20240518_203321.jpg", "20240520_131048.jpg", A["xc_peak"]["same"], 0.2),
                                                           (SURF_D, DIM, "DIFFERENT SENSOR (ultra-wide lens)", "20240518_203321.jpg", "20240520_070750.jpg", A["xc_peak"]["diff"], 2.6))):
        a = eout(prog(t, tc, tc + 0.5)); x = xs[k]
        fade_paste(cv, surf, (x, 215), a)
        tag(d, (x + 20, 190), lab, col, 22, True, a)
        for j, nm in enumerate((a_n, b_n)):
            fade_paste(cv, RP[nm].resize((110, 110), Image.LANCZOS), (x + 650 + 120 * j - 30, 182), a)
        bxp, byp = x + int(0.509 * 880), 215 + int(0.25 * 660)
        v = eout(prog(t, tc + 0.6, tc + 1.6))
        if k == 0 and v > 0:
            d.line((bxp, byp - 8, bxp, byp - 36), fill=AMBER, width=2)
            d.rounded_rectangle((bxp - 120, byp - 90, bxp + 120, byp - 36), 10, fill=mix(AMBER, a))
            d.text((bxp, byp - 63), f"peak = {val * v:.0f} × noise", font=font("monob", 26), fill=BG, anchor="mm")
        if k == 1 and v > 0:
            d.rounded_rectangle((x + 300, 640, x + 640, 696), 10, outline=mix(DIM, a), width=2)
            d.text((x + 470, 668), f"no peak ({val:+.1f} × noise)", font=font("monob", 24), fill=mix(TXT, a), anchor="mm")

GRP = [A["truth"][n]["group"] for n in names]
GCOL, GNAME = GROUP_COL, GROUP_NAME
PAIR_SAME, PAIR_DIFF = [], []
for _i in range(N):
    for _j in range(_i + 1, N):
        if is_ai[_i] or is_ai[_j]: continue
        if GRP[_i] == GRP[_j] and GRP[_i] not in ("none", "ai"): PAIR_SAME.append((float(Z[_i, _j]), GRP[_i]))
        else: PAIR_DIFF.append(float(Z[_i, _j]))
ZD_MAX = max(PAIR_DIFF); MISSED = [(z_, g_) for z_, g_ in PAIR_SAME if z_ < 10]
def comC(cv, d, t):
    head(d, "FIG. 08 · EVERY PAIR OF REAL PHOTOS", "No false matches. Some weak ones are missed.", t)
    x0, x1 = 160, 1760; ymid = [330, 600]
    def X(z): return x0 + (z + 10) / 240 * (x1 - x0)
    d.line((x0, 760, x1, 760), fill=LINE, width=2)
    for zt in (0, 50, 100, 150, 200):
        d.line((X(zt), 754, X(zt), 766), fill=DIM, width=2); tag(d, (X(zt), 780), f"{zt}", DIM, 18, False, 1, "ma")
    tag(d, (x1, 812), "z = sigmas above chance", DIM, 18, False, 1, "ra")
    tx = X(10); la = smooth(prog(t, 0.2, 0.7))
    d.line((tx, 200, tx, 760), fill=mix(TXT, la), width=3); tag(d, (tx + 10, 188), "threshold z = 10", TXT, 20, True, la)
    rng = np.random.default_rng(4)
    for row, (arr, lab) in enumerate(((PAIR_DIFF, f"DIFFERENT CAMERAS · {len(PAIR_DIFF)} pairs"), (PAIR_SAME, f"SAME CAMERA · {len(PAIR_SAME)} pairs"))):
        tag(d, (x0, ymid[row] - 96), lab, DIM if row == 0 else AMBER, 22, True, smooth(prog(t, 0.2 + 0.4 * row, 0.7 + 0.4 * row)))
        jit = rng.uniform(-70, 70, len(arr))
        for k, it in enumerate(arr):
            z_, col = (it, DIM) if row == 0 else (it[0], GCOL[it[1]])
            a = back(prog(t, 0.6 + 0.4 * row + 0.012 * k, 1.0 + 0.4 * row + 0.012 * k))
            if a <= 0.02: continue
            r_ = int(11 * min(a, 1.2)); cv.paste(dot(r_, col), (int(X(z_)) - r_ - 2, int(ymid[row] + jit[k]) - r_ - 2), dot(r_, col))
    ga = smooth(prog(t, 2.2, 2.8))
    txt(d, (x0, 436), f"Wrong-camera pairs: highest z = {ZD_MAX:.1f}  →  zero false matches", GREEN, "s", 32, ga)
    names_missed = ", ".join(sorted({GNAME[g_] for _, g_ in MISSED}))
    txt(d, (x0, 684), f"{len(MISSED)} same-camera pairs stay below 10: {names_missed}", DIM, "m", 26, smooth(prog(t, 3.0, 3.6)))
    for k, (g_, c_) in enumerate(GCOL.items()):
        a = smooth(prog(t, 2.6 + 0.1 * k, 3.1 + 0.1 * k)); lx = x0 + (k % 4) * 400; ly = 722 + (k // 4) * 34
        rgba_paste(cv, dot(8, c_), (lx - 10, ly), a); tag(d, (lx + 14, ly), GNAME[g_], c_, 18, True, a)

def sceneCompare(cv, d, t):
    if t < 4: comA(cv, d, t)
    elif t < 9: comB(cv, d, t - 4); beat_in(cv, t - 4)
    else: comC(cv, d, t - 9); beat_in(cv, t - 9)

# ================================================================== SCENE 5 · AI PORTRAITS (no sensor, no fingerprint)
AIX = A["ai"]
AI_Z = AIX["z_ai"]; AI_PK = AIX["peak_ai"]
AIA_D, AIB_D = 5.2, 4.2
def aiA(cv, d, t):
    head(d, "FIG. 09 · AI PORTRAITS", "No sensor, no fingerprint", t)
    S = 250
    for g, (x0, lab, col, items) in enumerate(((110, "REAL CAMERA PHOTOS · noise map", AMBER, AIX["cam"]), (1010, "AI-GENERATED PORTRAITS · noise map", RED, AIX["ai"]))):
        a_ = smooth(prog(t, 0.1 + 0.4 * g, 0.6 + 0.4 * g))
        d.line((x0, 215, x0 + 810, 215), fill=mix(col, a_), width=3); tag(d, (x0, 178), lab, col, 22, True, a_)
        for k, it in enumerate(items):
            aa = back(prog(t, 0.3 + 0.5 * g + 0.18 * k, 0.7 + 0.5 * g + 0.18 * k)); ca = clamp(aa); x = x0 + k * (S + 30); y = 250
            fade_paste(cv, it["img"].resize((S, S), Image.BICUBIC), (x, y), ca); frame_rect(d, (x - 1, y - 1, x + S, y + S), mix(col, ca), 3)
            tag(d, (x, y + S + 14), f"best match z = {it['z'] * eout(prog(t, 1.6, 2.6)):.1f}", col, 20, True, smooth(prog(t, 1.4, 2.0)))
    g_ = smooth(prog(t, 2.6, 3.3))
    txt(d, (110, 620), f"z = {min(c['z'] for c in AIX['cam']):.0f} – {max(c['z'] for c in AIX['cam']):.0f}", AMBER, "b", 72, g_); txt(d, (110, 710), "photos of the same phone find each other", DIM, "m", 28, g_)
    txt(d, (1010, 620), f"z = {min(AI_Z):.1f} – {max(AI_Z):.1f}", RED, "b", 72, g_); txt(d, (1010, 710), f"all {len(AI_Z)} AI portraits: chance level, below 10", DIM, "m", 28, g_)
    txt(d, (1010, 756), "no camera to match: but also no proof of AI.", TXT, "m", 28, smooth(prog(t, 3.6, 4.2)))

def aiB(cv, d, t):
    head(d, "FIG. 10 · TRIED: SPECTRAL PEAK SCORE", "A simple spectral test does not flag them", t)
    x0, x1, y = 160, 1760, 520; vmax = 26.0
    def X(v): return x0 + v / vmax * (x1 - x0)
    d.line((x0, y + 150, x1, y + 150), fill=LINE, width=2)
    for v in range(0, 27, 5): d.line((X(v), y + 144, X(v), y + 156), fill=DIM, width=2); tag(d, (X(v), y + 168), f"{v}", DIM, 18, False, 1, "ma")
    tag(d, (x1, y + 200), "spectral peak score (peak ÷ background)", DIM, 18, False, 1, "ra")
    order = sorted(range(N), key=lambda i: peak[i]); lanes = []
    for k, i in enumerate(order):
        a = back(prog(t, 0.3 + 0.07 * k, 0.75 + 0.07 * k))
        if a <= 0.02: continue
        lane = 0
        while any(abs(X(peak[i]) - px) < 34 and l == lane for (px, l) in lanes): lane += 1
        lanes.append((X(peak[i]), lane)); yy = y + 110 - lane * 44 - int(240 * (1 - clamp(a)))
        col = RED if is_ai[i] else (AMBER if cam[i] else DIM); r_ = 17; cv.paste(dot(r_, col), (int(X(peak[i])) - r_ - 2, yy - r_ - 2), dot(r_, col))
    g = smooth(prog(t, 2.0, 2.6)); pa = peak[is_ai]; pn = peak[~is_ai]
    d.line((X(pa.min()), y + 190, X(pa.max()), y + 190), fill=mix(RED, g), width=6);
    txt(d, (160, 250), f"AI portraits: {pa.min():.1f} – {pa.max():.1f}", RED, "s", 36, g)
    txt(d, (160, 300), f"all other photos: {pn.min():.1f} – {pn.max():.1f}", AMBER, "s", 36, g)
    txt(d, (160, 360), "The ranges overlap: this score cannot separate them.", TXT, "m", 30, smooth(prog(t, 3.0, 3.6)))
    for k, (lab, col_) in enumerate((("AI portrait", RED), ("camera photo", AMBER), ("no match", DIM))):
        rgba_paste(cv, dot(7, col_), (x0 + 6 + k * 190, y + 211), g); tag(d, (x0 + 28 + k * 190, y + 214), lab, col_, 17, True, g)

def sceneAI(cv, d, t):
    if t < AIA_D: aiA(cv, d, t)
    else: aiB(cv, d, t - AIA_D); beat_in(cv, t - AIA_D)

# ================================================================== SCENE 6 · CODE
CHUNKS = [("NOISE RESIDUAL", "photo - blur(photo)", src_lines(P.noise_residual), "removes the scene, keeps the noise"),
          ("COMPARE", "z = r * sqrt(N) > 10", src_lines(P.correlate) + [(None, "")] + src_lines(P.group_by_camera), "same sensor = same pattern")]
def colorize(code):
    cols = []
    for tt, val in lex(code, PythonLexer()):
        if tt in Token.Comment: c = DIM
        elif tt in Token.Literal.String: c = GREEN
        elif tt in Token.Literal.Number: c = PINK
        elif tt in Token.Keyword or tt in Token.Name.Builtin: c = AMBER
        elif tt in Token.Name.Function: c = (255, 255, 255)
        else: c = TXT
        cols += [c] * len(val)
    return cols[:len(code)]
CHUNK_DATA = []
for ch in CHUNKS:
    ls = [l for l in ch[2]]
    while ls and ls[-1][1] == "": ls.pop()
    code = "\n".join(l[1] for l in ls); CHUNK_DATA.append((ls, code, colorize(code)))
MAXC = max(len(l[1]) for cd in CHUNK_DATA for l in cd[0])
CODE_FS = max(18, min(28, int((1100 - 120) / MAXC / 0.602)))
CODE_ADV = font("mono", CODE_FS).getlength("M")
CODE_PER = 3.6; CODE_D = CODE_PER * len(CHUNKS)
def sceneCode(cv, d, t):
    head(d, "FIG. 11 · CODE", "The whole method, prnu_detect.py", t)
    per = CODE_PER; k = min(len(CHUNKS) - 1, int(t // per)); lt = t - per * k
    ls, code, cols = CHUNK_DATA[k]
    lh = int(CODE_FS * 1.5); wy = 180; wh = 70 + lh * len(ls) + 30
    d.rounded_rectangle((80, wy, 1180, wy + wh), 14, fill=PANEL, outline=LINE, width=2)
    for i in range(3): d.ellipse((104 + i * 26, wy + 22, 118 + i * 26, wy + 36), fill=(70, 69, 65))
    tag(d, (630, wy + 29), "prnu_detect.py", DIM, 20, False, 1, "mm", bg=PANEL)
    total = len(code); n = int(round(total * prog(lt, 0.15, 2.4))); off = 0
    for li, (no, text) in enumerate(ls):
        vis = clamp(n - off, -1, len(text)); y = wy + 70 + li * lh
        if vis >= 0:
            if no is not None: d.text((80 + 64, y), str(no), font=font("mono", 18), fill=(95, 93, 88), anchor="ra")
            x = 80 + 90; j = 0; vis = int(vis)
            while j < vis:
                c = cols[off + j]; j2 = j
                while j2 < vis and cols[off + j2] == c: j2 += 1
                d.text((x + j * CODE_ADV, y), text[j:j2], font=font("mono", CODE_FS), fill=c); j = j2
            if n < total and off <= n <= off + len(text) and int(t * 5) % 2 == 0:
                d.rectangle((x + vis * CODE_ADV, y + 2, x + vis * CODE_ADV + 12, y + CODE_FS + 6), fill=AMBER)
        off += len(text) + 1
    for s in range(len(CHUNKS)):
        cx, cy = 1250, 180 + s * 230; act = s == k
        d.rounded_rectangle((cx, cy, cx + 590, cy + 200), 12, fill=PANEL if act else BG, outline=(AMBER if act else LINE), width=3 if act else 2)
        bgc = PANEL if act else BG
        tag(d, (cx + 26, cy + 22), f"0{s + 1}", AMBER if act else DIM, 40, True, 1, bg=bgc)
        txt(d, (cx + 100, cy + 24), CHUNKS[s][0], TXT if act else DIM, "s", 30, 1, bg=bgc)
        tag(d, (cx + 26, cy + 100), CHUNKS[s][1], AMBER if act else DIM, 24, True, 1, bg=bgc)
        tag(d, (cx + 26, cy + 150), CHUNKS[s][3], DIM, 18, False, 1, bg=bgc)
    if lt < 0.2: d.rectangle((1250, 180 + k * 230, 1250 + 590, 180 + k * 230 + 200), outline=TXT, width=3)

# ================================================================== SCENE 7 · RUN (39 images)
outs = [l for l in A["run_output"] if not l.startswith("saved")]
RANK = np.argsort(np.argsort(is_ai * 100000 + (np.array(cam) == 0) * 50000 + np.array(cam) * 1000 + np.arange(N)))
NCOL = 8
GX0, GY0, GCEL, GGAP = 80, 190, 100, 10
def grid_xy(i): return GX0 + (i % NCOL) * (GCEL + GGAP), GY0 + (i // NCOL) * (GCEL + GGAP)
TAG_T = [1.9 + 0.12 * i for i in range(N)]
PROMPT = "$ python prnu_detect.py --folder photos39/"
def term_events():
    ev = []; tail0 = TAG_T[-1] + 0.5
    for i, l in enumerate(outs):
        if i < 2: ev.append((1.6, l))
        elif i < 2 + N: ev.append((TAG_T[i - 2], l))
        else: ev.append((tail0 + 0.18 * (i - 2 - N), l))
    return ev
TERM_EV = term_events()
TH_G = [thumb(i, GCEL) for i in range(N)]
RUNA_D, RUNB_D, RUNC_D, RUND_D = 10.5, 4.0, 4.5, 8.0
RUN_D = RUNA_D + RUNB_D + RUNC_D + RUND_D
N_REAL = sum(1 for k in kind if k == "real"); N_SIM = sum(1 for k in kind if k == "sim"); N_AI = int(is_ai.sum())

def runA(cv, d, t):
    head(d, "FIG. 12 · RUN", f"Run it on {N} images", t)
    for i in range(N):
        x, y = grid_xy(i); a = min(back(prog(t, 0.1 + 0.02 * i, 0.4 + 0.02 * i)), 1)
        fade_paste(cv, TH_G[i], (x, y), a)
        tagged = t >= TAG_T[i]; col = sc_col(i) if tagged else LINE
        frame_rect(d, (x - 1, y - 1, x + GCEL, y + GCEL), mix(col, a), 4 if tagged else 2)
        if tagged:
            d.rectangle((x, y + GCEL - 22, x + GCEL - 1, y + GCEL - 1), fill=mix(col, a))
            lab = f"CAMERA {cam[i]}" if cam[i] else "NO MATCH"
            tag(d, (x + GCEL // 2, y + GCEL - 11), lab, BG, 13, True, a, "mm", bg=col)
            if t - TAG_T[i] < 0.2: frame_rect(d, (x - 5, y - 5, x + GCEL + 4, y + GCEL + 4), mix(TXT, a), 3)
        if sim[i] or is_ai[i]:
            d.rectangle((x + 4, y + 4, x + 38, y + 21), fill=mix(BG, a)); tag(d, (x + 21, y + 13), "AI" if is_ai[i] else "SIM", RED if is_ai[i] else TXT, 12, True, a, "mm")
    nt = sum(1 for i in range(N) if t >= TAG_T[i]); nl = sum(1 for i in range(N) if t >= TAG_T[i] and cam[i] > 0)
    yb = GY0 + ((N + NCOL - 1) // NCOL) * (GCEL + GGAP) + 16
    txt(d, (GX0, yb), f"{nl} / {nt} photos linked to a camera", AMBER, "b", 40, smooth(prog(t, 1.9, 2.4)))
    tag(d, (GX0, yb + 62), "Portrait photos are tried in both rotation directions.", DIM, 18, False, smooth(prog(t, 2.4, 3.0)))
    tx, ty, tw, th = 1000, 190, 840, 745
    d.rounded_rectangle((tx, ty, tx + tw, ty + th), 14, fill=PANEL, outline=LINE, width=2)
    for i in range(3): d.ellipse((tx + 24 + i * 26, ty + 20, tx + 38 + i * 26, ty + 34), fill=(70, 69, 65))
    pn = int(len(PROMPT) * prog(t, 0.3, 1.5)); rows = [(PROMPT[:pn] + ("_" if t < 1.7 and int(t * 5) % 2 == 0 else ""), TXT)]
    for tm, s in TERM_EV:
        if t >= tm:
            c = TXT; m = re.search(r"(camera \d|unmatched)\s*$", s)
            if m and "r=" in s: c = CAMCOL[int(m.group(1)[-1])] if "camera" in m.group(1) else DIM
            elif s.startswith("summary") or s.startswith("images:"): c = DIM
            rows.append((s, c))
    rows = rows[-28:]
    for k, (s, c) in enumerate(rows): d.text((tx + 22, ty + 54 + k * 24), s, font=font("mono", 17), fill=c)

def runB(cv, d, t):
    head(d, "FIG. 13 · SIMILARITY MATRIX", "Sort by verdict: blocks appear", t)
    cell = 17; x0, y0 = 240, 250; e = smooth(prog(t, 1.2, 2.8)); ma = smooth(t / 0.3)
    tag(d, (x0, y0 - 66), "NOISE CORRELATION · every image vs every image", AMBER if e > 0.5 else DIM, 20, True, ma)
    for i in range(N):
        pi = lerp(i, RANK[i], e)
        for j in range(N):
            pj = lerp(j, RANK[j], e); v = clamp((Z[i, j] - 3) / 25) ** 0.8
            col = tuple(int(x) for x in cmap_amber(np.array([v]))[0]) if v > 0.02 else (24, 24, 22)
            x, y = x0 + pj * cell, y0 + pi * cell; d.rectangle((x, y, x + cell - 2, y + cell - 2), fill=col)
    for i in range(N):
        c = lerpc(DIM, sc_col(i), e); pi = lerp(i, RANK[i], e)
        d.rectangle((x0 - 18, y0 + pi * cell + 2, x0 - 8, y0 + pi * cell + cell - 3), fill=c)
        d.rectangle((x0 + pi * cell + 2, y0 - 18, x0 + pi * cell + cell - 3, y0 - 8), fill=c)
    la = smooth(prog(t, 2.6, 3.2))
    n_other = sum(1 for i in range(N) if cam[i] == 0 and not is_ai[i])
    items = [(CAMCOL[c], f"camera {c} · {cam.count(c)} photos · {GROUP_NAME[CAMGRP[c]]}" + (" (sim.)" if CAMGRP[c] == "cam_sim_a" else "")) for c in sorted(CAMCOL)]
    items += [(RED, f"AI portraits · {N_AI} (no match)"), (DIM, f"other no match · {n_other}")]
    for k, (col, s) in enumerate(items):
        d.rectangle((1000, 280 + k * 46, 1020, 300 + k * 46), fill=mix(col, ma)); tag(d, (1036, 280 + k * 46), s, TXT, 21, True, ma)
    yy = 280 + len(items) * 46 + 30
    txt(d, (1000, yy), "bright cell = strong match (z)", DIM, "m", 28, la); txt(d, (1000, yy + 44), "AI portraits stay dark:", DIM, "m", 28, la); txt(d, (1000, yy + 88), "no sensor, no match.", RED, "m", 28, la)

NET_RING = {"cam_samsung": (330, 480, 185), "cam_nikon_b": (810, 400, 95), "cam_nikon_a": (1150, 400, 95), "cam_s7": (810, 735, 70), "cam_sim_a": (1150, 745, 70)}
def net_layout():
    pos = {}
    for c in sorted(CAMGRP):
        cx, cy, rad = NET_RING[CAMGRP[c]]; mem = [i for i in range(N) if cam[i] == c]
        for k, i in enumerate(mem):
            if CAMGRP[c] == "cam_sim_a": pos[i] = (cx - 70 + 140 * k, cy)
            else: a = 2 * np.pi * k / len(mem) - np.pi / 2; pos[i] = (cx + rad * np.cos(a), cy + rad * np.sin(a))
    nm = [i for i in range(N) if cam[i] == 0 and not is_ai[i]]
    for k, i in enumerate(nm): pos[i] = (1470 + (k % 4) * 110, 345 + (k // 4) * 95)
    ai = [i for i in range(N) if is_ai[i]]
    for k, i in enumerate(ai): pos[i] = (1500 + (k % 3) * 120, 795 + (k // 3) * 90)
    return pos
NET = net_layout()
NET_EDGES = [(i, j) for i in range(N) for j in range(i + 1, N) if Z[i, j] > 10 and not is_ai[i] and not is_ai[j]]
NET_LAYER = aaline_layer([(*NET[i], *NET[j], tuple(int(c * 0.8) for c in CAMCOL.get(cam[i], DIM)), int(clamp(Z[i, j] / 30, 1.5, 6))) for i, j in NET_EDGES], (W, H))
NET_NODE = {i: circ_thumb(THUMB[i], 36, sc_col(i)) for i in range(N)}
def runC(cv, d, t):
    head(d, "FIG. 14 · FINGERPRINT NETWORK", "Matches link into camera clusters", t)
    tag(d, (80, 170), "EDGE = SAME CAMERA (z > 10) · THICKER = STRONGER", AMBER, 20, True, smooth(t / 0.4))
    rgba_paste(cv, NET_LAYER, (0, 0), smooth(prog(t, 1.0, 1.8)))
    for i in range(N):
        x, y = NET[i]; a = min(back(prog(t, 0.05 + 0.03 * i, 0.5 + 0.03 * i)), 1)
        sp = NET_NODE[i]; s_ = max(int(sp.width * a), 2)
        sp2 = sp.resize((s_, s_), Image.LANCZOS) if s_ != sp.width else sp
        cv.paste(sp2, (int(x - s_ / 2), int(y - s_ / 2)), sp2)
    n_other = sum(1 for i in range(N) if cam[i] == 0 and not is_ai[i])
    labs = []
    for c in sorted(CAMGRP):
        g_ = CAMGRP[c]; cx, cy, rad = NET_RING[g_]; mem = [i for i in range(N) if cam[i] == c]; zs = [zbest[i] for i in mem]
        ly = cy + (rad if g_ != "cam_sim_a" else 0) + 62
        nm_ = "SIMULATED PATTERN" if g_ == "cam_sim_a" else GROUP_NAME[g_].upper()
        labs.append((cx, ly, f"{nm_} · {len(mem)} photos", CAMCOL[c], "ma"))
        labs.append((cx, ly + 28, f"z = {min(zs):.0f} to {max(zs):.0f}", DIM, "ma"))
    labs += [(1650, 255, f"NO MATCH · {n_other} (weak or edited)", DIM, "ma"), (1620, 735, f"AI PORTRAITS · {N_AI} · NO MATCH", RED, "ma")]
    for (lx, ly, s, col, anc) in labs: tag(d, (lx, ly), s, col, 20, True, smooth((t - 1.6) / 0.6), anc)

def result_rows():
    wis = names.index("20240421_174950.jpg")
    spec = [("Samsung A33 · main lens", AMBER, [i for i in range(N) if GRP[i] == "cam_samsung" or i == wis], True)]
    for g_, title in (("cam_nikon_b", "Nikon · 4288×2848 sensor"), ("cam_nikon_a", "Nikon · 3008×2000 sensor"), ("cam_nikon_c", "Nikon · 4928×3264 sensor")):
        spec.append((title, GROUP_COL[g_], [i for i in range(N) if GRP[i] == g_], True))
    spec += [("Samsung S7", PINK, [i for i in range(N) if GRP[i] == "cam_s7"], True), ("Nokia 2.2", PURPLE, [i for i in range(N) if GRP[i] == "cam_nokia"], True),
             ("Samsung A40", BLUE, [i for i in range(N) if GRP[i] == "cam_a40"], True), ("Simulated sensor (girl, cat)", GREEN, [i for i in range(N) if GRP[i] == "cam_sim_a"], True),
             ("AI-generated portraits", RED, [i for i in range(N) if is_ai[i]], False),
             ("Ultra-wide, edited, resized", DIM, [i for i in range(N) if GRP[i] == "none" and i != wis], False)]
    out = []
    for title, col, idxs, expect in spec:
        m = [i for i in idxs if cam[i] > 0]; zs = [zbest[i] for i in idxs]
        out.append(dict(title=title, col=col, idxs=idxs, n=len(idxs), m=len(m), zlo=min(zs), zhi=max(zs), expect=expect))
    return out
RROWS = result_rows()
CROSS_FALSE = sum(1 for i in range(N) for j in range(i + 1, N) if Z[i, j] > 10 and GRP[i] != GRP[j])
N_LINK = sum(r["m"] for r in RROWS if r["expect"]); N_EXP = sum(r["n"] for r in RROWS if r["expect"])
def runD(cv, d, t):
    head(d, "FIG. 15 · RESULTS", "Which photos were linked to their camera?", t)
    rh = 66; y0 = 170
    for k, r in enumerate(RROWS):
        a = smooth((t - 0.15 - 0.25 * k) / 0.4); y = y0 + k * rh
        if a <= 0.02: continue
        panel(d, (80, y, 1840, y + rh - 7), a); d.rectangle((80, y, 88, y + rh - 7), fill=mix(r["col"], a))
        x = 108
        for i in r["idxs"][:7]: fade_paste(cv, thumb(i, 36), (x, y + 11), a); x += 42
        txt(d, (500, y + 4), r["title"], TXT, "s", 26, a, bg=PANEL)
        zt = f"best match z = {r['zlo']:.0f} – {r['zhi']:.0f}" if r["zhi"] >= 10 else f"best match z = {r['zlo']:.1f} – {r['zhi']:.1f}"
        tag(d, (500, y + 37), zt, DIM, 15, False, a, bg=PANEL)
        if r["expect"]:
            c = GREEN if r["m"] == r["n"] else (AMBER if r["m"] else RED); s = f"{r['m']} / {r['n']} LINKED"
        else:
            c = GREEN if r["m"] == 0 else RED; s = "NO MATCH · CORRECT" if r["m"] == 0 else f"{r['m']} FALSE MATCHES"
        chip(d, 1480, y + 8, s, c, a, 19, 14, PANEL)
    b = smooth(prog(t, 3.6, 4.4)); yb = y0 + len(RROWS) * rh + 4
    txt(d, (80, yb), f"{N_LINK} of {N_EXP} photos with a known camera were linked. False matches between different cameras: {CROSS_FALSE}. Highest wrong-camera z: {ZD_MAX:.1f}.", GREEN, "s", 27, b)
    tag(d, (80, yb + 44), "Photos with no match are not proven fake: edited copies, other lenses and weak sensors give no match too.", DIM, 18, False, smooth(prog(t, 4.4, 5.0)))

def sceneRun(cv, d, t):
    if t < RUNA_D: runA(cv, d, t)
    elif t < RUNA_D + RUNB_D: runB(cv, d, t - RUNA_D); beat_in(cv, t - RUNA_D)
    elif t < RUNA_D + RUNB_D + RUNC_D: runC(cv, d, t - RUNA_D - RUNB_D); beat_in(cv, t - RUNA_D - RUNB_D)
    else: runD(cv, d, t - RUNA_D - RUNB_D - RUNC_D); beat_in(cv, t - RUNA_D - RUNB_D - RUNC_D)

# ================================================================== SCENE 8 · LIMITS
LIM = [("wisteria", IDX["20240421_174950.jpg"], AMBER, "16:9 crop: pixels no longer align", "missed", "MISS"),
       ("a40", [i for i in range(N) if GRP[i] == "cam_a40"][0], BLUE, "Samsung A40: its two photos share no pattern", "not matched", "MISS"),
       ("nokia", [i for i in range(N) if GRP[i] == "cam_nokia"][0], PURPLE, f"Nokia 2.2: weak pattern, z = {zbest[[i for i in range(N) if GRP[i] == 'cam_nokia'][0]]:.1f}", "below the threshold", "MISS"),
       ("corridor", IDX["20240520_070750.jpg"], GREEN, "ultra-wide lens = other sensor", "no match, correct", "OK"),
       ("ai", next(i for i in range(N) if A["label"][i] == "AI-generated portrait 2"), RED, "AI portrait: no sensor", "no match, correct", "OK"),
       ("self-portrait", IDX["red_shirt.png"], GREEN, "edited PNG: pattern gone", "no match, correct", "OK")]
LIM_D = 6.0
def sceneLimits(cv, d, t):
    head(d, "FIG. 16 · HONEST LIMITS", "What weakens a fingerprint", t)
    for k, (nm, i, col, l1, l2, mark) in enumerate(LIM):
        c_, r_ = k % 2, k // 2; x0 = 80 + c_ * 900; y0 = 190 + r_ * 235
        a = back(prog(t, 0.15 + 0.2 * k, 0.5 + 0.2 * k)); ca = clamp(a)
        if ca <= 0.02: continue
        off = int(24 * (1 - ca))
        panel(d, (x0, y0 + off, x0 + 860, y0 + 205 + off), ca)
        fade_paste(cv, thumb(i, 160), (x0 + 22, y0 + 22 + off), ca)
        txt(d, (x0 + 210, y0 + 28 + off), l1, TXT, "s", fit_size(l1, "s", 30, 620), ca, bg=PANEL)
        txt(d, (x0 + 210, y0 + 78 + off), l2, col, "m", 28, ca, bg=PANEL)
        chip(d, x0 + 210, y0 + 128 + off, mark, col, ca, 20, 14, PANEL)
    tag(d, (80, 892), "Use original files straight from the camera: every resize or re-encode weakens the fingerprint.", DIM, 19, False, smooth(prog(t, 1.4, 2.0)))

# ================================================================== SCENE 9 · END (92-100)
HFP = Image.fromarray((EFP_LINE * np.clip((1 - EFP_R) / 0.12, 0, 1) * 255).astype(np.uint8))
def sceneEnd(cv, d, t):
    a = eout(t / 0.8)
    txt(d, (960, 330), "Thank you", TXT, "b", 170, a, "ma")
    txt(d, (960, 540), "Hadi Sarhangi Fard", DIM, "m", 30, smooth(prog(t, 0.5, 1.1)), "ma")
    tag(d, (960, 650), "To see more:", DIM, 28, False, smooth(prog(t, 1.0, 1.6)), "ma")
    for k, (lab, url) in enumerate((("PORTFOLIO", "https://hadifard.github.io/"), ("GITHUB", "https://github.com/Hadifard"))):
        b_ = eout(prog(t, 1.4 + 0.4 * k, 2.1 + 0.4 * k)); y = 720 + k * 80
        tag(d, (500, y), lab, AMBER, 22, True, b_); d.text((690, y - 4), url, font=font("mono", 30), fill=mix(TXT, b_))

# ================================================================== PART 2 · MODERN PRNU (92-157)
A2 = pickle.load(open(os.path.join(HERE, "assets_v2.pkl"), "rb"))
CAMS2, CLAB = A2["cams"], A2["label"]
BLUE = (120, 190, 255)
CCOL = {"A33": AMBER, "A40": GREEN, "S7": PINK, "NOKIA": BLUE, "NIKON_A": TEAL, "NIKON_B": LIME, "NIKON_C": MAGENTA}
RES2 = A2["res"]
P2_FOOT = "PART 2 · real photos from 7 cameras · computed by prnu_v2.py"

def gimg(a, gain=2.4, size=None, resample=Image.NEAREST):
    v = np.clip(128 + a / (gain * a.std() + 1e-9) * 127, 0, 255).astype(np.uint8)
    im = Image.fromarray(v).convert("RGB")
    return im.resize(size, resample) if size else im

def load_any(path, s):
    im = Image.open(path); im.draft("RGB", (s * 2, s * 2))
    im = ImageOps.exif_transpose(im).convert("RGB"); return cover(im, s, s)

# ---- P1 · five upgrades (92-99)
FIXES = [("NOISE", "Gaussian blur residual", "Wavelet + Wiener residual", AMBER),
         ("REFERENCE", "compare photo vs photo", "Maximum-likelihood camera fingerprint", GREEN),
         ("SCORE", "z = r · √N", "PCE: peak ÷ correlation energy", PINK),
         ("GEOMETRY", "one fixed centre crop", "search rotation × scale × shift", BLUE),
         ("TAMPERING", "one verdict per photo", "local correlation map of the photo", RED)]
def p2Upgrades(cv, d, t):
    head(d, "PART 2 · MODERN PRNU", "Five upgrades to the detector", t)
    tag(d, (80, 190), "BEFORE", DIM, 22, True, smooth(t / 0.4)); tag(d, (1080, 190), "NOW", AMBER, 22, True, smooth(t / 0.4))
    for k, (lab, old, new, col) in enumerate(FIXES):
        a = back(prog(t, 0.3 + 0.7 * k, 0.9 + 0.7 * k)); ca = clamp(a)
        if ca <= 0.02: continue
        y = 240 + k * 128 + int(30 * (1 - ca)); panel(d, (80, y, 1840, y + 108), ca)
        d.rectangle((80, y, 90, y + 108), fill=mix(col, ca))
        tag(d, (116, y + 20), f"0{k + 1}  {lab}", col, 22, True, ca, bg=PANEL)
        txt(d, (116, y + 52), old, DIM, "m", 34, ca, bg=PANEL)
        lw = int(font("m", 34).getlength(old)); d.line((116, y + 72, 116 + int(lw * smooth(prog(t, 0.9 + 0.7 * k, 1.3 + 0.7 * k))), y + 72), fill=mix(DIM, ca, PANEL), width=3)
        tag(d, (1030, y + 54), "→", col, 44, True, ca, "mm", PANEL)
        txt(d, (1080, y + 40), new, TXT, "s", 38, smooth(prog(t, 0.9 + 0.7 * k, 1.4 + 0.7 * k)), bg=PANEL)

# ---- P2 · four fingerprints (+ the real reference photos they are estimated from)
FPT = {c: gimg(A2["fp"][c]["p64"], 2.2, (232, 232)) for c in CAMS2}
REF_FILES = {"A33": [os.path.join(ORIG, f"{m}.jpg") for m in ("20230504_131116", "20230505_125151", "20230527_214315", "20240509_210037", "20240518_203321", "20240520_131048")],
             "A40": [os.path.join(HERE, "raw_extract", f) for f in ("A40_00.jpg", "A40_02.jpg")], "S7": [os.path.join(HERE, "raw_extract", f) for f in ("S7_00.jpg", "S7_03.jpg", "S7_04.jpg")],
             "NOKIA": [os.path.join(HERE, "raw_extract", f) for f in ("nokia_01.jpg", "nokia_02.jpg")]}
_TR = json.load(open(os.path.join(HERE, "truth39.json")))
for _g, _c in (("cam_nikon_a", "NIKON_A"), ("cam_nikon_b", "NIKON_B"), ("cam_nikon_c", "NIKON_C")):
    REF_FILES[_c] = sorted(os.path.join(HERE, "photos39", f) for f, v in _TR.items() if v["group"] == _g)
REF_TH = {c: [load_any(p, 34) for p in fl] for c, fl in REF_FILES.items()}
P2FP_D = 7.0
def p2Fingerprints(cv, d, t):
    head(d, "FIG. 17 · SEVEN REAL FINGERPRINTS", "Same method, seven different patterns", t)
    for k, c in enumerate(CAMS2):
        a = back(prog(t, 0.15 + 0.2 * k, 0.6 + 0.2 * k)); ca = clamp(a); x = 70 + k * 258; y = 200
        fade_paste(cv, FPT[c], (x, y), ca); frame_rect(d, (x - 1, y - 1, x + 232, y + 232), mix(CCOL[c], ca), 4)
        txt(d, (x, y + 250), CLAB[c], CCOL[c], "b", fit_size(CLAB[c], "b", 26, 236), ca)
        tag(d, (x, y + 292), f"from {A2['n_ref'][c]} real photos", DIM, 15, False, ca)
        tag(d, (x, y + 314), f"{A2['shapes'][c][1]}×{A2['shapes'][c][0]}", DIM, 15, False, ca)
        for j, th in enumerate(REF_TH[c][:6]): fade_paste(cv, th, (x + (j % 6) * 38, y + 346), smooth(prog(t, 0.9 + 0.2 * k + 0.05 * j, 1.3 + 0.2 * k + 0.05 * j)))
    M = A2["fp_corr"]; n_ = len(CAMS2); off = max(abs(M[i, j]) for i in range(n_) for j in range(n_) if i != j)
    a = smooth(prog(t, 2.6, 3.3))
    txt(d, (80, 700), "64×64 pixel patch of each fingerprint, contrast boosted. Thumbnails: the photos it comes from.", DIM, "m", 26, a)
    txt(d, (80, 746), f"Fingerprints of different cameras are uncorrelated: max |r| = {off:.4f}", TXT, "m", 30, a)
    txt(d, (80, 800), "New: three Nikon bodies (14 TIFF photos) join the four phones.", AMBER, "m", 28, smooth(prog(t, 3.4, 4.1)))

# ---- P3 · better noise + the three formulas (109-117)
RS = A2["resid"]
RS_OLD, RS_NEW = gimg(RS["old"], 3.0, (480, 325), Image.BICUBIC), gimg(RS["new"], 3.0, (480, 325), Image.BICUBIC)
def p2Noise(cv, d, t):
    head(d, "FIG. 19 · BETTER NOISE, BETTER SCORE", "Wavelet denoiser keeps the fingerprint", t)
    for k, (im, lab, col, rv, sg) in enumerate(((RS_OLD, "OLD · photo − gaussian blur", DIM, RS["r_old"], RS["s_old"]), (RS_NEW, "NEW · wavelet + Wiener shrinkage", AMBER, RS["r_new"], RS["s_new"]))):
        a = eout(prog(t, 0.2 + 0.5 * k, 0.7 + 0.5 * k)); x = 80 + k * 540
        fade_paste(cv, im, (x, 210), a); frame_rect(d, (x - 1, 209, x + 480, 536), mix(col, a), 3)
        tag(d, (x, 556), lab, col, 18, True, a)
        v = eout(prog(t, 1.2 + 0.4 * k, 2.2 + 0.4 * k))
        txt(d, (x, 600), f"r = {rv * v:.2f}", col, "b", 70, smooth(prog(t, 1.1 + 0.4 * k, 1.6 + 0.4 * k)))
        tag(d, (x, 690), "correlation with the camera fingerprint", DIM, 17, False, smooth(prog(t, 1.4, 2.0)))
    tag(d, (1230, 300), "→", AMBER, 80, True, smooth(prog(t, 1.6, 2.2)), "mm")
    x0 = 1190
    for k, (a1, a2_, col) in enumerate((("W = I − denoise(I)", "residual of every photo", TXT), ("K = Σ Wᵢ·Iᵢ / Σ Iᵢ²", "maximum-likelihood fingerprint", AMBER),
                                        ("PCE = peak² / mean(rest²)", "over every shift: the decision score", GREEN))):
        a = smooth(prog(t, 2.6 + 0.9 * k, 3.2 + 0.9 * k)); y = 330 + k * 150
        panel(d, (x0, y, x0 + 650, y + 124), a); d.rectangle((x0, y, x0 + 8, y + 124), fill=mix(col, a))
        tag(d, (x0 + 30, y + 24), a1, col, 27, True, a, bg=PANEL); txt(d, (x0 + 30, y + 74), a2_, DIM, "m", 26, a, bg=PANEL)
    txt(d, (80, 770), "PCE > 60 is the classic decision level (Goljan & Fridrich).", DIM, "m", 28, smooth(prog(t, 4.6, 5.4)))
    txt(d, (80, 816), "One noise-only peak is expected near 40-50: so the claim needs PCE far above it.", TXT, "m", 28, smooth(prog(t, 5.2, 6.0)))

# ---- P4 · geometry search (117-127)
def surface3d(win, zmax, size=(960, 720)):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    amber = LinearSegmentedColormap.from_list("amber", [(0, "#3a260c"), (0.4, "#8a531c"), (0.75, "#f2a93b"), (1, "#fff7e4")])
    n = win.shape[0]; h = n // 2; X, Y = np.meshgrid(np.arange(-h, h + 1), np.arange(-h, h + 1))
    fig = plt.figure(figsize=(size[0] / 200, size[1] / 200), dpi=200, facecolor="#0d0d0c"); ax = fig.add_subplot(111, projection="3d", facecolor="#0d0d0c")
    ax.plot_surface(X, Y, win, cmap=amber, vmin=-4, vmax=9, rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax.set_zlim(-8, zmax); ax.view_init(elev=27, azim=-58); ax.set_axis_off(); ax.set_box_aspect((1, 1, 0.75), zoom=1.18); fig.subplots_adjust(0, 0, 1, 1)
    fig.canvas.draw(); im = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB"); plt.close(fig); return im
_zm = float(A2["surf_wis_match"].max()) * 1.05
SURF_M = surface3d(A2["surf_wis_match"], _zm).resize((760, 570), Image.LANCZOS); SURF_W = surface3d(A2["surf_wis_wrong"], _zm).resize((760, 570), Image.LANCZOS)
WG = A2["wis_geom"]
def logx(v, lo, hi, x0, x1): return x0 + (np.log10(max(v, lo)) - np.log10(lo)) / (np.log10(hi) - np.log10(lo)) * (x1 - x0)
def curve_panel(cv, d, t, box, pts, title, col, xr, mark=None, a=1.0):
    x0, y0, x1, y1 = box; panel(d, box, a); tag(d, (x0 + 20, y0 + 16), title, col, 18, True, a, bg=PANEL)
    px0, px1, py0, py1 = x0 + 70, x1 - 30, y1 - 50, y0 + 70
    def X(s): return px0 + (s - xr[0]) / (xr[1] - xr[0]) * (px1 - px0)
    def Y(p): return py0 + (np.log10(max(p, 1.0)) / 4.0) * (py1 - py0)
    d.line((px0, py0, px1, py0), fill=LINE, width=2); d.line((px0, py0, px0, py1), fill=LINE, width=2)
    for e in range(0, 5): d.line((px0 - 6, Y(10 ** e), px0, Y(10 ** e)), fill=DIM, width=2); tag(d, (px0 - 12, Y(10 ** e)), f"{10 ** e:g}", DIM, 14, False, a, "rm", PANEL)
    d.line((px0, Y(60), px1, Y(60)), fill=mix(TXT, 0.8 * a, PANEL), width=2); tag(d, (px1, Y(60) - 22), "PCE 60", TXT, 15, True, a, "ra", PANEL)
    n = int(len(pts) * eout(prog(t, 0.2, 2.2))); P_ = [(X(s), Y(p)) for s, p in pts[:n]]
    if len(P_) > 1: d.line(P_, fill=mix(col, a, PANEL), width=3)
    for xv in np.linspace(xr[0], xr[1], 5): tag(d, (X(xv), py0 + 12), f"{xv:.3f}", DIM, 14, False, a, "ma", PANEL)
    tag(d, (px1, py0 + 34), "resize factor tried →", DIM, 14, False, a, "ra", PANEL)
    if mark and t > 2.2:
        s_, p_ = mark; m = smooth(prog(t, 2.2, 2.7)); d.ellipse((X(s_) - 12, Y(p_) - 12, X(s_) + 12, Y(p_) + 12), outline=mix(TXT, m, PANEL), width=3)
        tag(d, (X(s_) + 24, Y(p_) - 6), f"s = {s_:.3f}  ·  PCE {p_:,.0f}", TXT, 20, True, m, "lm", PANEL)
def p2Geometry(cv, d, t):
    if t < 5.0:
        head(d, "FIG. 20 · CROPPED PHOTO, SAME SENSOR", "Where on the sensor was this photo taken?", t)
        bx, by, bw = 80, 220, 700; sc_ = bw / WG["ksh"][1]; a = eout(t / 0.4)
        d.rectangle((bx, by, bx + bw, by + int(WG["ksh"][0] * sc_)), outline=mix(LINE, a), width=3, fill=mix(PANEL, a))
        tag(d, (bx, by - 30), f"SENSOR FRAME  {WG['ksh'][1]} × {WG['ksh'][0]}", DIM, 18, True, a)
        gy, gx = WG["shift"]; gh, gw = WG["gshape"]; r_ = prog(t, 0.6, 1.8)
        d.rectangle((bx + gx * sc_, by + gy * sc_, bx + (gx + gw) * sc_, by + (gy + gh * eout(r_)) * sc_), outline=AMBER, width=4, fill=mix(AMBER, 0.25 * a))
        tag(d, (bx, by + int(WG["ksh"][0] * sc_) + 24), f"16:9 PHOTO  {gw} × {gh}  ·  found at row {gy}, column {gx}", AMBER, 20, True, smooth(prog(t, 1.5, 2.2)))
        txt(d, (bx, by + int(WG["ksh"][0] * sc_) + 70), "The old centre-crop compared the wrong pixels: it missed this photo.", DIM, "m", 26, smooth(prog(t, 2.2, 3.0)))
        txt(d, (bx, by + int(WG["ksh"][0] * sc_) + 112), "Searching every shift locks onto the right ones.", TXT, "m", 26, smooth(prog(t, 2.8, 3.6)))
        for k, (im, lab, col, v) in enumerate(((SURF_M, "SAMSUNG A33 · same camera", AMBER, WG["pce"]), (SURF_W, "SAMSUNG S7 · other camera", DIM, None))):
            aa = eout(prog(t, 0.4 + 0.5 * k, 1.0 + 0.5 * k)); x = 820 + (k % 1) * 0; y = 170 + k * 400
            fade_paste(cv, im.resize((540, 405), Image.LANCZOS), (1330 if False else 1000 + k * 0, y), aa) if False else fade_paste(cv, im.resize((520, 390), Image.LANCZOS), (1000 + k * 440, 330), aa)
            tag(d, (1000 + k * 440, 300), lab, col, 20, True, aa)
            if k == 0: txt(d, (1000, 740), f"PCE = {WG['pce'] * eout(prog(t, 1.5, 2.6)):,.0f}", AMBER, "b", 64, smooth(prog(t, 1.4, 2.0)))
            else: txt(d, (1440, 740), "no peak", DIM, "b", 64, smooth(prog(t, 1.8, 2.4)))
    else:
        tt = t - 5.0; head(d, "FIG. 21 · RESIZED COPY", "Scan the resize factor: a sharp peak appears", min(1, tt / 0.3 + 0.5))
        ct, cg = A2["curve_T2"], A2["curve_girl"]; pk = max(ct["pts"], key=lambda q: q[1])
        curve_panel(cv, d, tt, (80, 190, 940, 840), ct["pts"], "RESIZED COPY OF A SAMSUNG A33 PHOTO", AMBER, (0.628, 0.660), pk, smooth(tt / 0.3))
        gp = [(s, p) for s, p in cg["pts"]]; gk = max(gp, key=lambda q: q[1])
        curve_panel(cv, d, tt, (980, 190, 1840, 840), gp, "GIRL PHOTO vs A33 · identical scan", DIM, (0.628, 0.660), None, smooth(tt / 0.3))
        a = smooth(prog(tt, 2.8, 3.5)); txt(d, (1020, 860), f"best of {len(gp)} scales × 2 rotations: PCE {gk[1]:.0f}", DIM, "m", 26, a)
        txt(d, (120, 860), "True factor 0.644 (long side 4000 → 2576 px).", TXT, "m", 26, smooth(prog(tt, 2.2, 3.0)))

# ---- P5 · results on tampered photos (127-139)
def row_path(r):
    f = r["file"]; p = os.path.join(HERE, "tamper_set", f); return p if os.path.exists(p) else os.path.join(ORIG, f)
ROW_TH = {r["file"]: load_any(row_path(r), 50) for r in RES2}
def p2Results(cv, d, t):
    head(d, "FIG. 22 · TAMPERED PHOTOS vs SEVEN CAMERAS", "Which camera took it?", t)
    n_ = len(CAMS2); cw = 100; cx = [740 + k * (cw + 8) for k in range(n_)]; y0 = 170
    for k, c in enumerate(CAMS2): tag(d, (cx[k] + cw // 2, y0), CLAB[c].replace("Samsung ", "S. ").replace("Nikon ", "N ").replace("×", "x"), CCOL[c], 12, True, smooth(t / 0.4), "ma")
    tag(d, (1640, y0), "VERDICT", DIM, 15, True, smooth(t / 0.4)); tag(d, (110, y0), "PHOTO · WHAT HAPPENED TO IT", DIM, 15, True, smooth(t / 0.4))
    rh = 44
    for i, r in enumerate(RES2):
        a = smooth(prog(t, 0.3 + 0.3 * i, 0.6 + 0.3 * i)); y = 196 + i * rh
        if a <= 0.02: continue
        fade_paste(cv, ROW_TH[r["file"]].resize((38, 38), Image.LANCZOS), (80, y), a)
        sz = fit_size(r["desc"], "m", 21, 570); txt(d, (128, y + 9), r["desc"], TXT, "m", sz, a)
        for k, c in enumerate(CAMS2):
            v = r["by_cam"][c]["pce"]; w_ = int(cw * clamp(np.log10(max(v, 1)) / 4.0) * eout(prog(t, 0.5 + 0.3 * i, 1.2 + 0.3 * i)))
            d.rectangle((cx[k], y + 6, cx[k] + cw, y + 16), fill=mix(DARK, a)); d.rectangle((cx[k], y + 6, cx[k] + max(w_, 2), y + 16), fill=mix(CCOL[c] if v > 100 else DIM, a))
            tag(d, (cx[k], y + 21), f"{v:,.0f}", TXT if v > 100 else DIM, 13, v > 100, a)
        ok = r["verdict"] != "no match"
        if ok:
            good = r["truth"] is None or r["truth"] == r["verdict"]
            chip(d, 1640, y + 4, f"{CLAB[r['verdict']]}" + (" ✓" if r["truth"] else ""), CCOL[r["verdict"]] if good else RED, a, 15, 10)
        else: chip(d, 1640, y + 4, "no match", DIM, a, 15, 10)
    for k in range(n_): d.line((cx[k] + cw * np.log10(100) / 4.0, 192, cx[k] + cw * np.log10(100) / 4.0, 196 + len(RES2) * rh - 4), fill=mix(TXT, 0.35), width=1)
    tag(d, (80, 200 + len(RES2) * rh + 2), "Bars: PCE on a log scale (best of rotation × scale × shift).  Thin line: PCE 100.  ✓ = ground truth known.  The two Nikon edits (X1, X2) are NOT recovered: honest negative result.", DIM, 14, False, smooth(prog(t, 5.0, 5.8)))

# ---- P6 · tamper localisation (139-148)
def heat_img(m, size):
    rm, th, region = gaussian_filter(m["rmap"], 1.2), m["thumb"], m["region"]; med = max(m["med"], 1e-4)
    base = np.stack([np.clip(th / 255.0, 0, 1)] * 3, -1) * 0.55
    v = np.clip(rm / (1.7 * med), 0, 1); heat = cmap_amber(v ** 0.9).astype(np.float32) / 255.0; weak = np.clip((0.45 - rm / med) / 0.45, 0, 1)[..., None]
    heat = heat * (1 - weak) + np.array([1.0, 0.35, 0.23]) * weak
    out = (base * 0.35 + heat * 0.65) * 255
    return Image.fromarray(out.clip(0, 255).astype(np.uint8)).resize(size, Image.BICUBIC)
MAPS = [("T1_crop.jpg", "UNTOUCHED · plain crop"), ("T4_splice.jpg", "SPLICE · patch from another camera"), ("T5_retouch.jpg", "RETOUCH · object removed")]
MAP_IM = {k: heat_img(A2["maps"][k], (560, 420)) for k, _ in MAPS if k in A2["maps"]}
def p2Maps(cv, d, t):
    head(d, "FIG. 23 · WHERE WAS IT EDITED?", "Local correlation map", t)
    for k, (nm, lab) in enumerate(MAPS):
        if nm not in MAP_IM: continue
        a = back(prog(t, 0.2 + 0.4 * k, 0.8 + 0.4 * k)); ca = clamp(a); x = 80 + k * 600; y = 200; m = A2["maps"][nm]
        fade_paste(cv, MAP_IM[nm], (x, y), ca); frame_rect(d, (x - 1, y - 1, x + 560, y + 420), mix(AMBER if k else GREEN, ca), 3)
        tag(d, (x, y + 436), lab, AMBER if k else GREEN, 18, True, ca)
        rm = m["rmap"]; inside = None
        if m["region"]:
            dv = m.get("div", 8); ry, rx, rh_, rw_ = [int(v // dv) for v in m["region"]]; oy, ox = m["org"]; ry -= oy // dv; rx -= ox // dv
            sx, sy = 560 / rm.shape[1], 420 / rm.shape[0]
            bx = (x + rx * sx, y + ry * sy, x + (rx + rw_) * sx, y + (ry + rh_) * sy)
            if prog(t, 2.0, 2.4) > 0: d.rectangle(bx, outline=mix(TXT, smooth(prog(t, 2.0, 2.4))), width=3); tag(d, (bx[0], bx[1] - 24), "REAL EDIT", TXT, 16, True, smooth(prog(t, 2.0, 2.4)))
            mk = np.zeros(rm.shape, bool); mk[max(ry, 0):ry + rh_, max(rx, 0):rx + rw_] = True; inside = float(rm[mk].mean()); outside = float(rm[~mk].mean())
        else: outside = float(rm.mean())
        g_ = smooth(prog(t, 2.6 + 0.3 * k, 3.2 + 0.3 * k))
        if inside is not None:
            txt(d, (x, y + 480), f"r outside the edit  {outside:.3f}", AMBER, "m", 28, g_); txt(d, (x, y + 526), f"r inside the edit   {inside:.3f}", RED, "m", 28, g_)
        else: txt(d, (x, y + 480), f"r median  {float(np.median(rm)):.3f}", GREEN, "m", 28, g_); txt(d, (x, y + 526), "red = dark archway: no light, no pattern", DIM, "m", 25, g_)
    txt(d, (80, 800), "Window 128 px, step 32 px. Amber = fingerprint present, red = missing (edit, or too dark).", DIM, "m", 26, smooth(prog(t, 4.0, 4.8)))
    txt(d, (80, 846), "A pasted patch carries someone else's noise; an inpainted area carries none.", TXT, "m", 26, smooth(prog(t, 4.8, 5.6)))

# ---- P7 · genuine vs impostor + summary (148-157)
def p2Stats(cv, d, t):
    head(d, "FIG. 24 · GENUINE vs IMPOSTOR", "No impostor reaches PCE 60", t)
    gen, imp = A2["gen"], A2["imp"]; x0, x1 = 120, 1100
    def X(v): return logx(v, 1, 1e6, x0, x1)
    d.line((x0, 700, x1, 700), fill=LINE, width=2)
    for e in range(0, 7): d.line((X(10 ** e), 694, X(10 ** e), 706), fill=DIM, width=2); tag(d, (X(10 ** e), 718), f"10^{e}", DIM, 16, False, 1, "ma")
    tag(d, (x1, 750), "PCE (log scale)", DIM, 17, False, 1, "ra")
    la = smooth(prog(t, 0.2, 0.7)); d.line((X(60), 220, X(60), 700), fill=mix(TXT, la), width=3); tag(d, (X(60) + 10, 668), "PCE 60", TXT, 20, True, la)
    rng = np.random.default_rng(7)
    for row, (arr, col, lab, yc) in enumerate(((imp, DIM, f"IMPOSTOR · {len(imp)} wrong-camera comparisons", 330), (gen, AMBER, f"GENUINE · {len(gen)} leave-one-out photos", 590))):
        tag(d, (x0, yc - 100), lab, col, 20, True, smooth(prog(t, 0.3 + 0.4 * row, 0.8 + 0.4 * row)))
        for k, it in enumerate(arr):
            v = it[-1]; a = back(prog(t, 0.6 + 0.4 * row + 0.03 * k, 1.0 + 0.4 * row + 0.03 * k))
            if a <= 0.02: continue
            r_ = int(10 * min(a, 1.2)); cv.paste(dot(r_, col), (int(X(max(v, 1))) - r_ - 2, int(yc + rng.uniform(-50, 50)) - r_ - 2), dot(r_, col))
    gv = [g[-1] for g in gen]; iv = [i[-1] for i in imp]
    npass = sum(1 for v in gv if v > 60)
    low = sorted({g[1] for g in gen if g[-1] <= 60}); lowtxt = ", ".join(CLAB[c] for c in low)
    tag(d, (x0, 790), f"Below 60: {sum(1 for g in gen if g[-1] <= 60)} genuine photos ({lowtxt}). Reported, not hidden.", DIM, 18, False, smooth(prog(t, 3.0, 3.8)))
    stats = [(f"{npass} / {len(gv)}", "genuine photos above PCE 60", AMBER), (f"{max(iv):.0f}", "strongest wrong-camera match (PCE)", DIM),
             (f"{sum(1 for r in RES2 if r['truth'] and r['verdict'] == r['truth'])} / {sum(1 for r in RES2 if r['truth'])}", "edited test photos traced to the right camera", GREEN),
             (f"{sum(1 for r in RES2 if r['truth'] and r['verdict'] not in (r['truth'], 'no match'))}", "wrong-camera attributions", RED)]
    for k, (b_, s_, col) in enumerate(stats):
        a = smooth((t - 1.2 - 0.35 * k) / 0.4); y = 200 + k * 175
        txt(d, (1300, y), b_, col, "b", 80, a); txt(d, (1300, y + 94), s_, DIM, "m", 24, a)


# ================================================================== INTRO (new) · real or fake? -> fingerprints -> why it matters -> how
PURPLE = (186, 150, 255)
AIDIR = os.path.join(HERE, "ai_new")
def load_path(path, maxside=1600):
    im = Image.open(path); im.draft("RGB", (maxside, maxside))
    im = ImageOps.exif_transpose(im).convert("RGB"); im.thumbnail((maxside, maxside), Image.LANCZOS); return im
QUIZ = [("ai", os.path.join(AIDIR, "AI_generated_person_01.jpg"), (0.5, 0.33)), ("ai", os.path.join(AIDIR, "AI_generated_person_02.jpg"), (0.5, 0.33)),
        ("real", os.path.join(ORIG, "red_shirt.png"), (0.5, 0.36)), ("ai", os.path.join(AIDIR, "AI_generated_person_06.jpg"), (0.5, 0.4))]
QW, QH = 400, 500
QIMG = [cover(load_path(p), QW, QH, fx=f[0], fy=f[1]) for _, p, f in QUIZ]
INTRO_FOOT = "Quiz · three AI-generated portraits and one real photograph"

def seq(chunks, t=0.3, per_word=0.14, base=0.28, gap=0.08):
    """Caption chunks -> [(t0, t1, text)] timed by word count; returns (caps, end_time)."""
    out = []
    for c in chunks:
        n = len(re.sub(r"\*", "", c).split()); dd = n * per_word + base
        out.append((t, t + dd, c)); t += dd + gap
    return out, t

# ---- slide 1 · the question
HK_CAPS, HK_END = seq(["How sure are you that the images you see every day", "on *social media,* in *the news,* or on *television* are real or fake by AI?"], 0.3)
HK_D = HK_END + 0.3
_mos = Image.new("RGB", (700, 700), BG)
for _k, _im in enumerate(QIMG):
    _t = cover(_im, 340, 340, fy=0.3).filter(ImageFilter.GaussianBlur(16)); _t = Image.blend(_t, Image.new("RGB", _t.size, BG), 0.35)
    _mos.paste(_t, ((_k % 2) * 360, (_k // 2) * 360))
def introHook(cv, d, t):
    tag(d, (80, 170), "A QUESTION BEFORE THE SCIENCE", AMBER, 22, True, smooth(t / 0.4))
    txt(d, (76, 230 + int(50 * (1 - back(t / 0.6)))), "Real", TXT, "b", 190, smooth(t / 0.3))
    txt(d, (76, 430 + int(50 * (1 - back((t - 0.25) / 0.6)))), "or fake?", AMBER, "b", 190, smooth((t - 0.25) / 0.3))
    x = 80
    for k, s in enumerate(("SOCIAL MEDIA", "NEWS", "TELEVISION")):
        a = back((t - 1.2 - 0.3 * k) / 0.4)
        if a > 0.02: x += chip(d, x, 700 + int(24 * (1 - clamp(a))), s, TXT, clamp(a), 24) + 20
    for k in range(4):
        a = smooth(prog(t, 0.2 + 0.25 * k, 0.7 + 0.25 * k)); fade_paste(cv, _mos.crop(((k % 2) * 360, (k // 2) * 360, (k % 2) * 360 + 340, (k // 2) * 360 + 340)), (1090 + (k % 2) * 360 - 10, 160 + (k // 2) * 360), a)
    q = back(prog(t, 0.9, 1.6))
    if q > 0.02:
        sz = int(420 * clamp(q)); txt(d, (1450, 520), "?", AMBER, "b", max(sz, 20), 1.0, "mm")

# ---- slide 2 · the four-image quiz
QZ_CAPS = [(0.3, 2.0, "Look at these *four images.*"), (2.3, 5.2, "*Which one is real,* and which *three* were generated by *AI?*")]
QZ_REVEAL = 5.2 + 1.35         # 2 s of video time after the question has been shown
QZ_D = QZ_REVEAL + 2.0
QX = [100 + k * (QW + 47) for k in range(4)]; QY = 190
def introQuiz(cv, d, t):
    head(d, "QUIZ", "Which one is real?", t)
    rv = smooth(prog(t, QZ_REVEAL, QZ_REVEAL + 0.5)); dark = Image.new("RGB", (QW, QH), BG)
    for k, (kd, _, _) in enumerate(QUIZ):
        a = back(prog(t, 0.2 + 0.18 * k, 0.7 + 0.18 * k)); ca = clamp(a)
        if ca <= 0.02: continue
        x, y = QX[k], QY + int(30 * (1 - ca)); im = QIMG[k]
        if kd == "ai" and rv > 0: im = Image.blend(im, dark, 0.45 * rv)
        fade_paste(cv, im, (x, y), ca)
        col = LINE
        if rv > 0: col = lerpc(LINE, RED if kd == "ai" else GREEN, rv)
        frame_rect(d, (x - 1, y - 1, x + QW, y + QH), mix(col, ca), 5 if rv > 0 else 2)
        if rv < 0.5: d.ellipse((x + 16, y + 16, x + 76, y + 76), fill=mix(BG, ca)); tag(d, (x + 46, y + 46), str(k + 1), TXT, 34, True, ca, "mm")
        if rv > 0.3:
            lab, c2 = ("AI-GENERATED", RED) if kd == "ai" else ("REAL PHOTOGRAPH", GREEN)
            chip(d, x + 14, y + QH - 70, lab, c2, smooth((rv - 0.3) / 0.5), 22, 14, BG, True)
    a = smooth(prog(t, QZ_REVEAL + 0.3, QZ_REVEAL + 0.9))
    txt(d, (100, 740), "3 generated by AI", RED, "b", 52, a); txt(d, (760, 740), "1 real photograph: the red shirt", GREEN, "b", 52, a)


# ================================================================== INTRO 2 (new) · fingerprints on lenses -> what cameras leave behind -> literature examples -> deep models + maths
LENS_IMG = load_path(os.path.join(HERE, "fingerprint.on.lens.jpg"), 1500)
EXAMPLES_IMG = load_path(os.path.join(HERE, "Examples-fingerprints.webp"), 1600)

# ---- slide A · "Did you know?"  (short, big sentence + the lens photo at the same time, then a 2 s pause)
DID_D = 3.4
def introDid(cv, d, t):
    lh = 770; lw = int(LENS_IMG.width * lh / LENS_IMG.height); lens = LENS_IMG.resize((lw, lh), Image.LANCZOS)
    a = smooth(prog(t, 0.15, 0.8)); x_img = 1920 - lw - 150
    fade_paste(cv, lens, (x_img, 150), a); frame_rect(d, (x_img - 1, 149, x_img + lw, 150 + lh), mix(LINE, a), 2)
    for k, (s, col) in enumerate((("Did you know", TXT), ("cameras leave", TXT), ("fingerprints", AMBER), ("on images?", AMBER))):
        b = smooth(prog(t, 0.2 + 0.3 * k, 0.65 + 0.3 * k)); off = int(28 * (1 - b))
        txt(d, (110, 250 + k * 135 + off), s, col, "b", 116, b)

# ---- slide B · what every camera leaves behind (big centred text, word by word)
TR_TEXT = ("The parts inside every camera leave *subtle marks* on its images: sensor noise, tiny bumps on the lens, and more. "
           "Like a fingerprint, *no two cameras are exactly alike.* The same repeating pattern appears in *every image* that camera takes, "
           "so it can be traced, and in the end *it gives the camera away.*")
def _tr_words(text, size, max_w):
    words = []; hl = False
    for tok in re.split(r"(\*)", text):
        if tok == "*": hl = not hl; continue
        for w in tok.split(): words.append((w, hl))
    f = font("s", size); sp = f.getlength(" "); lines, cur, cw = [], [], 0
    for w, h in words:
        ww = f.getlength(w)
        if cur and cw + sp + ww > max_w: lines.append(cur); cur, cw = [], 0
        cur.append((w, h, ww)); cw += (sp if len(cur) > 1 else 0) + ww
    lines.append(cur); return lines, sp
TR_SIZE = 58; TR_LINES, TR_SP = _tr_words(TR_TEXT, TR_SIZE, 1560)
TR_NW = sum(len(l) for l in TR_LINES); TR_PER = 0.11
TRACE_D = 0.4 + TR_NW * TR_PER + 2.4
def introTraces(cv, d, t):
    tag(d, (80, 60), "WHAT A CAMERA LEAVES BEHIND", AMBER, 22, True, smooth(t / 0.4))
    lh = int(TR_SIZE * 1.42); y0 = 540 - lh * len(TR_LINES) // 2; k = 0
    for li, ln in enumerate(TR_LINES):
        lw = sum(w for _, _, w in ln) + TR_SP * (len(ln) - 1); x = (W - lw) / 2
        for (w, h, ww) in ln:
            a = smooth(prog(t, 0.4 + k * TR_PER, 0.4 + k * TR_PER + 0.3)); k += 1
            if a > 0.01: d.text((x, y0 + li * lh), w, font=font("s", TR_SIZE), fill=mix(AMBER if h else TXT, a))
            x += ww + TR_SP

# ---- slide C · real examples from the literature: device fingerprints (PRNU) vs model fingerprints (Noiseprint)
EXAMPLES_D = 7.0
def introExamples(cv, d, t):
    head(d, "THE FINGERPRINTS OF DIFFERENT CAMERAS", "Every device leaves its own pattern", t)
    iw = 1100; ih = int(EXAMPLES_IMG.height * iw / EXAMPLES_IMG.width); im = EXAMPLES_IMG.resize((iw, ih), Image.LANCZOS)
    a = eout(prog(t, 0.2, 0.9)); fade_paste(cv, im, (80, 190), a); frame_rect(d, (79, 189, 80 + iw, 190 + ih), mix(LINE, a), 2)
    for k, (lab, sub, col) in enumerate((("TOP ROW · PRNU", "the fingerprint of one individual device", AMBER), ("BOTTOM ROW · NOISEPRINT", "the fingerprint of a camera model, learned by a deep neural network", PINK))):
        b = smooth(prog(t, 1.2 + 0.9 * k, 1.8 + 0.9 * k)); y = 260 + k * 250
        panel(d, (1250, y, 1840, y + 210), b); d.rectangle((1250, y, 1258, y + 210), fill=mix(col, b))
        tag(d, (1282, y + 26), lab, col, 24, True, b, bg=PANEL)
        for li, ln in enumerate(textwrap.wrap(sub, 34)): txt(d, (1282, y + 76 + 40 * li), ln, TXT, "m", 28, b, bg=PANEL)
    tag(d, (80, 190 + ih + 18), "Examples from the literature: device fingerprints (PRNU, top), model fingerprints (Noiseprint, bottom).", DIM, 17, False, smooth(prog(t, 1.0, 1.6)))

# ---- slide D · deep models + the maths behind them (subtitle talks, maths and a network sketch build up)
DP_CAPS, DP_END = seq(["*Deep models* extract these patterns,", "then *machine learning* and *correlation* find the images of the same camera,", "and even reveal *the camera model* and its type."], 0.3, per_word=0.15)
DP_D = max(DP_END + 0.4, 9.0)
FORMULAS = [("I = I₀·(1 + K) + ε", "photo = scene × (1 + sensor fingerprint K) + noise", AMBER),
            ("W = I − F(I)", "residual: F = wavelet filter or a CNN denoiser", TXT),
            ("K̂ = Σ Wᵢ·Iᵢ / Σ Iᵢ²", "maximum-likelihood fingerprint of one camera", GREEN),
            ("ρ = corr( W , I·K̂ )", "correlation → same camera? (PCE = peak² / mean(rest²))", PINK),
            ("ŷ = softmax( f_θ(W) )", "a CNN predicts the camera model from the residual", BLUE)]
def _net_diagram(cv, d, t, x0, y0):
    """Input photo -> CNN (feature maps) -> noise residual -> correlation + classifier -> verdict."""
    xs = [x0, x0 + 150, x0 + 420, x0 + 590]; S = 110; cy = y0 + 110
    a0 = smooth(prog(t, 0.2, 0.7)); fade_paste(cv, PH["flowers2"].resize((S, S), Image.LANCZOS), (xs[0], cy - S // 2), a0); frame_rect(d, (xs[0] - 1, cy - S // 2 - 1, xs[0] + S, cy + S // 2), mix(LINE, a0), 2)
    tag(d, (xs[0] + S // 2, cy + 80), "PHOTO", DIM, 15, True, a0, "ma")
    a1 = smooth(prog(t, 0.9, 1.5)); layers = [(100, 5), (76, 7), (54, 9), (34, 11)]; cx = xs[1]
    for k, (h_, n_) in enumerate(layers):
        for j in range(min(3, 1 + k)):
            bx, by = cx + k * 56 + j * 7, cy - h_ // 2 - j * 7
            d.rectangle((bx, by, bx + 18, by + h_), outline=mix(AMBER, a1), width=2, fill=mix(PANEL, a1))
    tag(d, (cx + 100, cy + 80), "CNN · conv + ReLU", AMBER, 15, True, a1, "ma")
    a2 = smooth(prog(t, 1.7, 2.3)); fade_paste(cv, RS_NEW.resize((S, S), Image.BICUBIC), (xs[2], cy - S // 2), a2); frame_rect(d, (xs[2] - 1, cy - S // 2 - 1, xs[2] + S, cy + S // 2), mix(AMBER, a2), 2)
    tag(d, (xs[2] + S // 2, cy + 80), "NOISE PATTERN", AMBER, 15, True, a2, "ma")
    a3 = smooth(prog(t, 2.5, 3.1)); d.rounded_rectangle((xs[3], cy - 80, xs[3] + 240, cy + 70), 12, fill=mix(PANEL, a3), outline=mix(GREEN, a3), width=3)
    txt(d, (xs[3] + 120, cy - 45), "same camera?", TXT, "s", 24, a3, "mm", PANEL); txt(d, (xs[3] + 120, cy), "which model?", TXT, "s", 24, a3, "mm", PANEL)
    tag(d, (xs[3] + 120, cy + 40), "correlation · classifier", GREEN, 14, True, a3, "mm", PANEL)
    for k, (xa, xb) in enumerate(((xs[0] + S + 8, xs[1] - 8), (xs[1] + 245, xs[2] - 8), (xs[2] + S + 8, xs[3] - 8))):
        aa = smooth(prog(t, 0.7 + 0.8 * k, 1.1 + 0.8 * k))
        d.line((xa, cy, xb, cy), fill=mix(TXT, aa), width=3); d.polygon([(xb, cy), (xb - 12, cy - 7), (xb - 12, cy + 7)], fill=mix(TXT, aa))
def introDeep(cv, d, t):
    head(d, "DEEP LEARNING MEETS PRNU", "From noise to the camera: the maths and the models", t)
    for k, (f_, s_, col) in enumerate(FORMULAS):
        a = smooth(prog(t, 0.4 + 0.75 * k, 0.9 + 0.75 * k)); y = 190 + k * 108
        if a <= 0.02: continue
        panel(d, (80, y, 960, y + 94), a); d.rectangle((80, y, 88, y + 94), fill=mix(col, a))
        d.text((112, y + 14), f_, font=font("monob", 34), fill=mix(col, a, PANEL)); tag(d, (112, y + 60), s_, DIM, 16, False, a, bg=PANEL)
    tag(d, (1010, 190), "THE PIPELINE", AMBER, 20, True, smooth(prog(t, 0.2, 0.7)))
    _net_diagram(cv, d, t - 0.0, 1010, 250)
    b = smooth(prog(t, 4.2, 5.0)); panel(d, (1010, 590, 1840, 800), b); d.rectangle((1010, 590, 1018, 800), fill=mix(PURPLE, b))
    tag(d, (1042, 612), "WHERE DEEP LEARNING FITS", PURPLE, 20, True, b, bg=PANEL)
    for li, ln in enumerate(("CNN denoisers replace the wavelet filter (cleaner noise),", "Noiseprint-style networks learn a camera-model fingerprint,", "classifiers read the camera model from the pattern.")):
        txt(d, (1042, 656 + 42 * li), ln, TXT, "m", 26, smooth(prog(t, 4.6 + 0.5 * li, 5.2 + 0.5 * li)), bg=PANEL)

# ---- slide 3 · images have a fingerprint (same camera, different scenes)
FP1_CAPS, FP1_END = seq(["Images also have *a fingerprint.*", "Even when two photographs look completely different, or edited,", "their underlying *sensor noise patterns* (fingerprint) can contain traces of *the same camera.*", "Yes, it is possible to find *the photographer* of an image, and *the type of camera!*"], 0.3)
FP1_D = FP1_END + 0.3
def introFinger(cv, d, t):
    head(d, "THE IDEA", "Different pictures, same hidden pattern", t)
    for k, key in enumerate(("flowers2", "gate")):
        a = back(prog(t, 0.2 + 0.4 * k, 0.8 + 0.4 * k)); ca = clamp(a); y = 200 + k * 330
        fade_paste(cv, pcover(key, 440, 300), (100, y + int(24 * (1 - ca))), ca); frame_rect(d, (99, y - 1 + int(24 * (1 - ca)), 540, y + 300 + int(24 * (1 - ca))), mix(LINE, ca), 2)
        tag(d, (100, y + 312), "TWO DIFFERENT SCENES" if k == 0 else "OTHER DAY · OTHER SCENE", DIM, 17, True, ca)
        la = smooth(prog(t, 1.6 + 0.3 * k, 2.4 + 0.3 * k)); d.line((560, y + 150, 560 + int(320 * la), y + 150 + int((440 - (y + 150)) * la)), fill=mix(AMBER, la), width=4)
    a = smooth(prog(t, 2.2, 3.0)); K = A["K64_img"].resize((420, 420), Image.NEAREST)
    fade_paste(cv, K, (900, 230), a); frame_rect(d, (899, 229, 1320, 650), mix(AMBER, a), 4)
    tag(d, (900, 664), "THE SENSOR'S NOISE PATTERN · BOOSTED · INVISIBLE", AMBER, 18, True, a)
    b = smooth(prog(t, 3.4, 4.2)); txt(d, (1400, 250), "= a fingerprint", TXT, "b", 52, b); txt(d, (1400, 330), "of the camera", AMBER, "b", 52, b)
    for k, s in enumerate(("WHICH PHONE?", "WHICH PHOTOGRAPHER?")):
        c = back(prog(t, 5.0 + 0.5 * k, 5.5 + 0.5 * k))
        if c > 0.02: chip(d, 1400, 470 + k * 90 + int(20 * (1 - clamp(c))), s, GREEN if k == 0 else TXT, clamp(c), 28)

# ---- slide 4 · cropped images
FP2_CAPS, FP2_END = seq(["Even if an image has been *cropped,* we can sometimes find the same fingerprint in the remaining regions", "and determine whether the images may have come from *the same camera.*"], 0.3)
FP2_D = FP2_END + 0.3
def introCrop(cv, d, t):
    head(d, "THE IDEA", "Cropped? The fingerprint can survive", t)
    fx, fy, fw, fh = 100, 210, 720, 540
    im = pcover("gate", fw, fh); fade_paste(cv, im, (fx, fy), smooth(t / 0.4)); frame_rect(d, (fx - 1, fy - 1, fx + fw, fy + fh), LINE, 2)
    s = smooth(prog(t, 0.8, 2.2)); L, T_, R_, B_ = lerp(0, 0.30, s), lerp(0, 0.22, s), lerp(1, 0.78, s), lerp(1, 0.86, s)
    box = (fx + L * fw, fy + T_ * fh, fx + R_ * fw, fy + B_ * fh)
    if s > 0.01:
        sh = Image.new("RGBA", (fw, fh), (13, 13, 12, 190)); hole = Image.new("L", (fw, fh), 255)
        ImageDraw.Draw(hole).rectangle((L * fw, T_ * fh, R_ * fw, B_ * fh), fill=0); sh.putalpha(hole.point(lambda v: int(v * 190 / 255))); cv.paste(sh, (fx, fy), sh)
        d.rectangle(box, outline=AMBER, width=4)
    tag(d, (fx, fy + fh + 16), "CROPPED: PART OF THE PICTURE IS GONE", DIM, 18, True, smooth(prog(t, 1.4, 2.2)))
    a = smooth(prog(t, 2.6, 3.4)); K = A["K64_img"].resize((420, 420), Image.NEAREST)
    d.line((860, 480, 960, 480), fill=mix(AMBER, a), width=4); fade_paste(cv, K, (1000, 250), a); frame_rect(d, (999, 249, 1420, 670), mix(AMBER, a), 4)
    tag(d, (1000, 684), "SAME PATTERN FOUND IN THE REMAINING PIXELS", AMBER, 18, True, a)
    c = back(prog(t, 4.0, 4.6))
    if c > 0.02: chip(d, 1000, 740, "SAME CAMERA?  →  CHECK", GREEN, clamp(c), 28)

# ---- slide 5 · why it matters
AP_CAPS, AP_END = seq(["This technology has important potential applications in *digital forensics,*", "*criminal investigations,* and the verification of *digital evidence.*"], 0.3)
AP_D = AP_END + 0.3
def introApps(cv, d, t):
    head(d, "WHY IT MATTERS", "Where this technology is used", t)
    for k, (title, col) in enumerate(((("DIGITAL", "FORENSICS"), AMBER), (("CRIMINAL", "INVESTIGATIONS"), GREEN), (("VERIFICATION OF", "DIGITAL EVIDENCE"), PINK))):
        a = back(prog(t, 0.2 + 0.4 * k, 0.7 + 0.4 * k)); ca = clamp(a)
        if ca <= 0.02: continue
        x0 = 80 + k * 600; y0 = 250 + int(40 * (1 - ca)); w_, h_ = 570, 420
        panel(d, (x0, y0, x0 + w_, y0 + h_), ca); d.rectangle((x0, y0, x0 + w_, y0 + 8), fill=mix(col, ca))
        tag(d, (x0 + 34, y0 + 50), f"0{k + 1}", col, 64, True, ca, bg=PANEL)
        yy = y0 + 190
        for wd in title: txt(d, (x0 + 34, yy), wd, TXT, "b", fit_size(wd, "b", 48, w_ - 70), ca, bg=PANEL); yy += 62

# ---- slide 6 · the how-question
HW_CAPS, HW_END = seq(["But how can we *extract* such a hidden fingerprint,", "*distinguish it* from ordinary image details, and *compare it mathematically?*", "Let's explore the process *step by step.*"], 0.3)
HW_D = HW_END + 0.3
def introHow(cv, d, t):
    head(d, "THE QUESTION", "Three problems to solve", t)
    for k, (n_, a1, a2, col) in enumerate((("01", "EXTRACT", "the hidden fingerprint", AMBER), ("02", "SEPARATE", "it from ordinary image details", GREEN), ("03", "COMPARE", "it mathematically", PINK))):
        a = back(prog(t, 0.3 + 0.9 * k, 0.9 + 0.9 * k)); ca = clamp(a)
        if ca <= 0.02: continue
        x0 = 80 + k * 610; y0 = 280 + int(40 * (1 - ca)); w_, h_ = 530, 360
        panel(d, (x0, y0, x0 + w_, y0 + h_), ca); d.rectangle((x0, y0, x0 + 10, y0 + h_), fill=mix(col, ca))
        tag(d, (x0 + 40, y0 + 36), n_, col, 40, True, ca, bg=PANEL); txt(d, (x0 + 40, y0 + 120), a1, TXT, "b", 62, ca, bg=PANEL)
        for li, wd in enumerate(textwrap.wrap(a2, 24)): txt(d, (x0 + 40, y0 + 210 + 46 * li), wd, DIM, "m", 34, ca, bg=PANEL)
        if k < 2: tag(d, (x0 + w_ + 40, y0 + h_ // 2), "→", col, 56, True, ca, "mm")

# ================================================================== timeline (built from per-scene durations and captions) + frame renderer
TITLE_CAPS = [(0.3, 2.9, "Every photo carries an *invisible fingerprint.*"), (3.1, 5.9, "It can point to *the camera* behind a photo.")]
_PD = WHYP_D
WHY_CAPS = [(0.3, 3.3, "Inside every camera: a sensor with *millions of pixels.*"), (3.5 + 0.2, 3.5 + 1.4, "Here is *a photo of Ava and Sophi.*"), (3.5 + 1.7, 3.5 + 3.3, "Zoom in: it is *just a grid of pixels.*"),
            (3.5 + 3.4, 3.5 + 5.0, "Each square is *one number.*"),
            (3.5 + _PD + 0.1, 3.5 + _PD + 3.7, "No chip is perfect: each pixel reacts to light *a little differently.*"),
            (3.5 + _PD + 4.0, 3.5 + _PD + 7.2, "The result is a pattern that is *unique, stable and invisible.*"), (3.5 + _PD + 7.6, 3.5 + _PD + 10.4, "Average more photos and *the fingerprint grows.*")]
EXT_CAPS = [(0.3, 4.3, "Blur the photo, subtract it: *only the noise remains.*"), (5.0, 9.3, "Same recipe for *a portrait:* a noise map your eye cannot see.")]
CMP_CAPS = [(0.3, 3.8, "Compare two noise patterns pixel by pixel: *correlation.*"), (4.2, 6.9, "Same camera: *a sharp spike.*"), (7.1, 8.9, "Different camera: *chance.*"),
            (9.2, 11.8, "Different cameras *never reach the threshold.*")]
AI_CAPS = [(0.3, 3.0, "*AI images* have no sensor: nothing to match."), (3.2, 5.0, "Six AI portraits: *z = 2 to 4.*"), (5.6, 9.2, "A spectral-peak score *cannot flag them.* No shortcut.")]
CODE_CAPS = [(0.3, 3.4, "The whole method fits in *a few lines of Python.*"), (3.9, 7.0, "Correlate every pair: *z above 10 means same camera.*")]
RUN_CAPS = [(0.3, 3.9, f"Now run it on *{N} images.*"), (4.1, 9.8, "Each image is *tagged automatically.*"),
            (10.7, 14.0, "Sort the rows and *blocks of matching noise appear.*"), (14.7, 18.5, "Matches link into *camera clusters.*"),
            (19.2, 26.5, "*No false matches.* Weak sensors and AI portraits stay unmatched.")]
LIM_CAPS = [(0.3, 5.7, "Resized or edited copies, and weak sensors, *lose the fingerprint.*")]
P2U_CAPS = [(0.3, 3.0, "Part 2: a *modern PRNU* pipeline."), (3.2, 6.8, "Five upgrades, one goal: *find the camera* of edited photos.")]
P2F_CAPS = [(0.3, 6.5, "Seven real cameras, seven *different* fingerprints.")]
P2N_CAPS = [(0.3, 4.0, "A wavelet denoiser keeps *more fingerprint.*"), (4.4, 7.7, "Peak-to-correlation energy replaces *a plain z-score.*")]
P2G_CAPS = [(0.3, 4.6, "Cropped photo: *search every shift* on the sensor."), (5.3, 9.7, "Resized photo: the right scale gives *a sharp peak.*")]
P2R_CAPS = [(0.3, 5.0, "Cropped, resized, rotated: *camera still found.*"), (5.8, 10.0, "Wrong cameras and heavy edits *stay below the threshold.*"), (10.5, 14.5, "The two Nikon edits are *not recovered:* TIFF to JPEG hurts.")]
P2M_CAPS = [(0.3, 4.2, "Slide a window over the photo: *local correlation.*"), (4.6, 8.8, "Pasted and retouched areas *show up as gaps.*")]
P2S_CAPS = [(0.3, 4.2, "*No impostor* reaches the threshold. Weak sensors fall short."), (4.7, 8.7, "Same method, *honest limits.*")]
DID_CAPS = []
TRACE_CAPS = []
EX_CAPS = [(0.3, 6.6, "*Every camera* has its own pattern.")]
END_CAPS = []
SCENE_LIST = [
    (6.0, sceneTitle, "Real photos · illustrated fingerprint overlay", TITLE_CAPS),
    (HK_D, introHook, INTRO_FOOT, HK_CAPS), (QZ_D, introQuiz, INTRO_FOOT, QZ_CAPS),
    (DID_D, introDid, "", DID_CAPS), (TRACE_D, introTraces, "", TRACE_CAPS), (EXAMPLES_D, introExamples, "Examples from the literature · device fingerprints (PRNU) and model fingerprints (Noiseprint)", EX_CAPS),
    (DP_D, introDeep, "Deep learning + PRNU · the formulas and pipeline behind the method", DP_CAPS), (FP1_D, introFinger, "Real photos · real fingerprint estimate (Samsung SM-A336E)", FP1_CAPS),
    (FP2_D, introCrop, "Real photo · real fingerprint estimate (Samsung SM-A336E)", FP2_CAPS), (AP_D, introApps, "", AP_CAPS), (HW_D, introHow, "", HW_CAPS),
    (3.5 + WHYP_D + 10.5, sceneWhy, "Real fingerprint estimate · Samsung SM-A336E · 6 original photos", WHY_CAPS),
    (EXT_D, sceneExtract, "Real photos · same code as prnu_detect.py", EXT_CAPS),
    (12.0, sceneCompare, "Real photos · values computed by prnu_detect.py", CMP_CAPS),
    (AIA_D + AIB_D, sceneAI, "6 real AI-generated portraits · fingerprint: none found", AI_CAPS),
    (CODE_D, sceneCode, "prnu_detect.py", CODE_CAPS),
    (RUN_D, sceneRun, f"AI / SIM badges = ground truth, not detector output · {N} images: {N_REAL} real files, {N_SIM} simulated pattern, {N_AI} AI", RUN_CAPS),
    (LIM_D, sceneLimits, "EXIF used only as ground truth", LIM_CAPS),
    (7.0, p2Upgrades, P2_FOOT, P2U_CAPS), (P2FP_D, p2Fingerprints, P2_FOOT, P2F_CAPS), (8.0, p2Noise, P2_FOOT, P2N_CAPS), (10.0, p2Geometry, P2_FOOT, P2G_CAPS),
    (15.0, p2Results, P2_FOOT, P2R_CAPS), (9.0, p2Maps, P2_FOOT, P2M_CAPS), (9.0, p2Stats, P2_FOOT, P2S_CAPS),
    (8.01, sceneEnd, "", END_CAPS)]
TL, CAPS = [], []
_t = 0.0
for _dur, _fn, _foot, _caps in SCENE_LIST:
    TL.append((_t, _t + _dur, _fn, _foot))
    for _a, _b, _s in _caps: CAPS.append((_t + _a, _t + _b, _s))
    _t += _dur
DUR = _t
CAP_SPR = [build_caption(c[2]) for c in CAPS]
print(f"timeline: {DUR:.1f} animation s -> {DUR * SLOW:.1f} s video; intro {sum(x[0] for x in SCENE_LIST[1:7]) * SLOW:.1f} s", file=sys.stderr)

def render_frame(T):
    for (s0, s1, fn, foot) in TL:
        if s0 <= T < s1: break
    cv = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(cv)
    lt = T - s0; fn(cv, d, lt)
    wipe = eout(prog(lt, 0.0, 0.34))
    if wipe < 0.999:
        wx = int(W * wipe); out = Image.new("RGB", (W, H), BG)
        if wx > 0: out.paste(cv.crop((0, 0, wx, H)), (0, 0))
        cv = out; d = ImageDraw.Draw(cv); d.rectangle((wx - 5, 0, wx, H), fill=AMBER)
    fo = 1 - smooth(prog(lt, (s1 - s0) - 0.15, s1 - s0))
    if fo < 0.999: cv = Image.blend(Image.new("RGB", (W, H), BG), cv, fo); d = ImageDraw.Draw(cv)
    draw_caption(cv, T)
    d.line((0, 1030, W, 1030), fill=LINE, width=2); d.line((0, 1030, int(W * T / DUR), 1030), fill=AMBER, width=3)
    if foot: d.text((80, 1048), foot, font=font("mono", 19), fill=DIM)
    if s0 < DUR - 8.01: d.text((1840, 1044), GITHUB, font=font("mono", 26), fill=TXT, anchor="ra")   # end card lists the address itself
    return cv

def render_chunk(args):
    idx, f0, f1 = args
    out = os.path.join(HERE, "chunks", f"c{idx:03d}.mp4"); os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out) and os.path.getsize(out) > 1000: return out          # resume after an interruption
    tmp = out + ".part.mp4"
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
                          "-profile:v", "high", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
                          "-threads", "1", "-g", "20", tmp], stdin=subprocess.PIPE)
    for f in range(f0, f1): p.stdin.write(render_frame(f / FPS / SLOW).tobytes())
    p.stdin.close(); p.wait(); os.replace(tmp, out); return out

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "preview":
        os.makedirs(os.path.join(HERE, "preview"), exist_ok=True)
        for s in sys.argv[2:]: render_frame(float(s)).save(os.path.join(HERE, "preview", f"t{float(s):06.2f}.png"))
    elif mode == "full":
        total = int(DUR * SLOW * FPS); per = 150
        jobs = [(i, i * per, min(total, (i + 1) * per)) for i in range((total + per - 1) // per)]
        wk = int(sys.argv[2]) if len(sys.argv) > 2 else 0; nw = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        if nw > 1:
            for j in jobs[wk::nw]: render_chunk(j)
            sys.exit(0)
        outs = [render_chunk(j) for j in jobs]
        with open(os.path.join(HERE, "chunks", "list.txt"), "w") as fh:
            for o in outs: fh.write(f"file '{o}'\n")
        subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", os.path.join(HERE, "chunks", "list.txt"), "-c", "copy",
                               os.path.join(HERE, OUT_NAME)])
        print("done")
