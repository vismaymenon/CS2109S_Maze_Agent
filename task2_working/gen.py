"""Training-data generator for the Task 2 image parser.

Builds synthetic grid cells the same way grid_universe's renderer composes them:
  background (floor, or wall) -> 'other' entities (full size) -> main entity -> up to 4 corner icons.
Every cell is made from the PUBLIC asset variants in ../data/assets, with random colour/brightness
changes so the classifier learns "what a boots/key/wall IS", not "what boots_1.png looks like".
"""
import glob
import os
import random

import numpy as np
from PIL import Image, ImageEnhance

HERE = os.path.dirname(os.path.abspath(__file__))
ASSET_ROOT = os.path.join(HERE, "..", "data", "assets")
F = 16          # classifier input: each cell is shrunk to F x F pixels
HIDDEN = 96     # hidden units of the classifier

# (name, render priority (lower = drawn on top), is corner icon, is background)
CLASSES = [
    ("boots", 4, True, False), ("box", 2, False, False), ("coin", 4, True, False),
    ("exit", 9, False, False), ("gem", 4, True, False), ("ghost", 4, True, False),
    ("human", 0, False, False), ("key", 4, True, False), ("lava", 7, False, False),
    ("locked", 6, False, False), ("opened", 6, False, False), ("shield", 4, True, False),
    ("wall", 9, False, True),
]
NAMES = [c[0] for c in CLASSES]
NAMES14 = NAMES + ["floor"]
INFO = {c[0]: c for c in CLASSES}
SUBICON = 0.4


def load_assets(root=ASSET_ROOT):
    """name -> list of (variant id, RGBA image), including 'floor'."""
    out = {}
    for name in NAMES + ["floor"]:
        files = sorted(glob.glob(os.path.join(root, name, "*.png")),
                       key=lambda f: int(os.path.basename(f).split("_")[-1][:-4]))
        out[name] = [(int(os.path.basename(f).split("_")[-1][:-4]), Image.open(f).convert("RGBA"))
                     for f in files]
    return out


def jitter(img, rng, name=""):
    """Random appearance change that imitates an unseen asset variant."""
    img = img.copy()
    if rng.random() < 0.6:                                  # hue rotation (small for floor/lava/exit)
        a = img.getchannel("A")
        hsv = np.asarray(img.convert("RGB").convert("HSV")).astype(np.int16)
        span = 20 if name in ("floor", "lava", "exit") else 255
        hsv[..., 0] = (hsv[..., 0] + rng.randint(-span, span)) % 256
        rgb = Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB")
        img = Image.merge("RGBA", (*rgb.split(), a))
    if rng.random() < 0.5:                                  # brightness / contrast / saturation
        a = img.getchannel("A")
        rgb = img.convert("RGB")
        rgb = ImageEnhance.Brightness(rgb).enhance(rng.uniform(0.6, 1.4))
        rgb = ImageEnhance.Contrast(rgb).enhance(rng.uniform(0.7, 1.3))
        rgb = ImageEnhance.Color(rgb).enhance(rng.uniform(0.4, 1.6))
        img = Image.merge("RGBA", (*rgb.split(), a))
    if name != "exit" and rng.random() < 0.3:               # mirror (never the EXIT sign: it has text)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.3:                                  # slight zoom
        w = img.size[0]
        n = max(8, int(w * rng.uniform(0.85, 1.1)))
        small = img.resize((n, n), Image.BILINEAR)
        canvas = Image.new("RGBA", img.size, (0, 0, 0, 0))
        if n > w:
            c = (n - w) // 2
            small = small.crop((c, c, c + w, c + w))
            canvas.alpha_composite(small, (0, 0))
        else:
            canvas.alpha_composite(small, (rng.randint(0, w - n), rng.randint(0, w - n)))
        img = canvas
    return img


def compose_cell(assets, labels, rng, size=64, aug=True, variants=None):
    """One RGBA cell showing the given entity names, drawn the way the real renderer draws them."""
    def pick(name):
        img = rng.choice((variants or assets)[name])[1]
        return jitter(img, rng, name) if aug else img

    labels = set(labels)
    cell = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    cell.paste(pick("wall" if "wall" in labels else "floor").resize((size, size), Image.BILINEAR), (0, 0))
    non_bg = [n for n in labels if not INFO[n][3]]
    main = min(non_bg, key=lambda n: INFO[n][1]) if non_bg else None
    icons = sorted([n for n in non_bg if INFO[n][2] and n != main], key=lambda n: INFO[n][1])[:4]
    others = sorted([n for n in non_bg if n != main and n not in icons], key=lambda n: (INFO[n][1], n))
    for n in others + ([main] if main else []):
        cell.alpha_composite(pick(n).resize((size, size), Image.BILINEAR))
    sub = int(size * SUBICON)
    for i, n in enumerate(icons):
        dx = size - sub if i % 2 == 1 else 0
        dy = size - sub if i // 2 == 1 else 0
        cell.alpha_composite(pick(n).resize((sub, sub), Image.BILINEAR), (dx, dy))
    return cell


def random_stack(rng):
    n = rng.choice([2, 2, 2, 3])
    s = set()
    base = rng.choice(["lava", "exit", "wall", "human", "box", "locked", None])
    if base:
        s.add(base)
    while len(s) < n:
        s.add(rng.choice(["coin", "gem", "key", "boots", "shield", "ghost", "human", "box", "exit", "lava"]))
    return list(s)


def main_label(labels):
    """What a person would call the cell: top-priority non-background entity, else wall, else floor."""
    non_bg = [n for n in labels if not INFO[n][3]]
    if non_bg:
        return min(non_bg, key=lambda n: INFO[n][1])
    return "wall" if "wall" in labels else "floor"


def to_feature(img, f=F):
    return np.asarray(img.convert("RGB").resize((f, f), Image.BOX), dtype=np.float32) / 255.0


def make_dataset(assets, n, seed, variants=None, aug=True, f=F):
    """n labelled cells: X (n, f, f, 3) in [0,1], y (n,) index into NAMES14."""
    rng = random.Random(seed)
    X = np.empty((n, f, f, 3), dtype=np.float32)
    y = np.empty(n, dtype=np.int64)
    for i in range(n):
        r = rng.random()
        labels = [] if r < 0.10 else ([rng.choice(NAMES)] if r < 0.88 else random_stack(rng))
        cell = compose_cell(assets, labels, rng, aug=aug, variants=variants)
        if rng.random() < 0.5:                              # imitate the renderer's nearest-neighbour resize
            s = rng.randint(40, 110)
            cell = cell.resize((s, s), Image.NEAREST)
        X[i] = to_feature(cell, f)
        y[i] = NAMES14.index(main_label(labels))
    return X, y


def split_variants(assets, holdout_frac, seed):
    """Per entity, keep some looks aside to test on looks the classifier never trained on."""
    rng = random.Random(seed)
    train, test = {}, {}
    for name, lst in assets.items():
        lst = list(lst)
        rng.shuffle(lst)
        k = max(1, int(round(len(lst) * holdout_frac)))
        test[name], train[name] = lst[:k], lst[k:]
    return train, test
