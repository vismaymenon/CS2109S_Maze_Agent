"""Train the final classifier on ALL public looks and write the block to paste into the Agent cell.

Output: paste_A_above_class.py  (parser code + the model's weights as one base64 string).
Usage:  python train_final.py
"""
import base64
import io
import os
import sys

import numpy as np
from sklearn.neural_network import MLPClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gen import HIDDEN, F, load_assets, make_dataset
import parser_core as pc

assert pc.F == F
assets = load_assets()
X, y = make_dataset(assets, 40000, 123)
model = MLPClassifier((HIDDEN,), max_iter=80, random_state=0).fit(pc.all_features(X), y)
w = dict(W1=model.coefs_[0], b1=model.intercepts_[0], W2=model.coefs_[1], b2=model.intercepts_[1])

# Compact storage: weight matrices as int8 (+ one scale each), biases as float16.
store = {}
for k, v in w.items():
    if k.startswith("W"):
        s = float(np.abs(v).max() / 127.0)
        store[k + "_q"] = np.round(v / s).astype(np.int8)
        store[k + "_s"] = np.array(s, dtype=np.float32)
    else:
        store[k + "_q"] = v.astype(np.float16)
buf = io.BytesIO()
np.savez_compressed(buf, **store)
b64 = base64.b64encode(buf.getvalue()).decode()
print("model size:", len(b64), "characters of base64;", sum(v.size for v in w.values()), "weights")

with open(os.path.join(HERE, "parser_core.py")) as f:
    core = f.read().split('"""', 2)[2]                    # drop the module docstring
loader = '''

_MODEL_B64 = "%s"


def load_model():
    d = np.load(io.BytesIO(base64.b64decode(_MODEL_B64)))
    return dict(W1=d["W1_q"].astype(np.float32) * float(d["W1_s"]), b1=d["b1_q"].astype(np.float32),
                W2=d["W2_q"].astype(np.float32) * float(d["W2_s"]), b2=d["b2_q"].astype(np.float32))


ENTITY_TYPES = {
    "boots": SpeedPowerUpEntity, "box": BoxEntity, "coin": CoinEntity, "exit": ExitEntity,
    "gem": GemEntity, "ghost": PhasingPowerUpEntity, "human": AgentEntity, "key": KeyEntity,
    "lava": LavaEntity, "locked": LockedDoorEntity, "opened": UnlockedDoorEntity,
    "shield": ShieldPowerUpEntity, "wall": WallEntity,
}
''' % b64
header = "# ---- Task 2 image parser: paste this block ABOVE `class Agent` ----\nimport base64\nimport io\n"
with open(os.path.join(HERE, "paste_A_above_class.py"), "w") as f:
    f.write(header + core + loader)
print("wrote paste_A_above_class.py")
