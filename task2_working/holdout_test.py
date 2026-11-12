"""Does the classifier generalise to asset looks it has never seen?

Keeps a fraction of every entity's looks aside, trains only on the rest, then draws the public levels
with ONLY the kept-aside looks (using the real renderer) and checks the parse.
Usage:  python holdout_test.py [split_seed=0] [holdout_fraction=0.25]
"""
import os
import shutil
import sys

import numpy as np
from sklearn.neural_network import MLPClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gen import HIDDEN, F, load_assets, make_dataset, split_variants
import parser_core as pc
import eval_parse
from grid_adventure.rendering import DEFAULT_ASSET_ROOT

split = int(sys.argv[1]) if len(sys.argv) > 1 else 0
frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
assert pc.F == F

assets = load_assets()
train_looks, test_looks = split_variants(assets, frac, split)

out = os.path.join(HERE, f"heldout_assets_s{split}")        # an asset folder holding only the unseen looks
shutil.rmtree(out, ignore_errors=True)
for name, lst in test_looks.items():
    os.makedirs(os.path.join(out, name))
    for vid, img in lst:
        img.save(os.path.join(out, name, f"{name}_{vid}.png"))
shutil.copytree(os.path.join(DEFAULT_ASSET_ROOT, "sleeping"), os.path.join(out, "sleeping"))

X, y = make_dataset(assets, 24000, split + 1, variants=train_looks)
model = MLPClassifier((HIDDEN,), max_iter=80, random_state=0).fit(pc.all_features(X), y)
w = dict(W1=model.coefs_[0], b1=model.intercepts_[0], W2=model.coefs_[1], b2=model.intercepts_[1])
print(f"trained on {1 - frac:.0%} of the looks; testing on the other {frac:.0%}")
eval_parse.run(w, out)
