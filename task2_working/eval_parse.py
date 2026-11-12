"""How well does a classifier read the REAL renderer's images of the public levels?
Compares predicted cells with the true cells from the environment (the agent never sees the truth)."""
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))          # project root: utils.py, public_levels/
sys.path.insert(0, HERE)

import utils
from utils import create_env
from public_levels.catalog import PUBLIC_BOSS_LEVEL_BUILDERS, PUBLIC_GAMEPLAY_LEVEL_BUILDERS
import parser_core as pc


def truth(state):
    W, H = state.width, state.height
    t = [set() for _ in range(W * H)]
    for eid, p in state.position.items():
        nm = state.appearance[eid].name
        if nm == "door":
            nm = "locked" if eid in state.locked else "opened"
        if nm in pc.NAMES:
            t[p.y * W + p.x].add(nm)
    return t


def run(weights, asset_root, seeds=(0, 1, 2), verbose=True):
    """Draw every public level with the looks in asset_root and score the parse."""
    utils.DEFAULT_ASSET_ROOT = str(Path(asset_root).resolve())   # create_env reads this when it renders
    builders = {**PUBLIC_GAMEPLAY_LEVEL_BUILDERS, **PUBLIC_BOSS_LEVEL_BUILDERS}
    cells = good_cells = levels = good_levels = 0
    bad = []
    for label, builder in builders.items():
        for s in seeds:
            env = create_env(builder, observation_type="image", seed=s)
            obs, _ = env.reset()
            cfg = obs["info"]["config"]
            pred = pc.classify_grid(obs["image"], cfg["width"], cfg["height"], weights)
            real = truth(env.state)
            wrong = [(i, real[i], pred[i]) for i in range(len(real)) if real[i] != pred[i]]
            cells += len(real); good_cells += len(real) - len(wrong)
            levels += 1; good_levels += (not wrong)
            if wrong:
                bad.append((label, s, wrong[:4]))
    print(f"cells right {good_cells}/{cells} = {good_cells / cells:.4f} | "
          f"levels read perfectly {good_levels}/{levels} = {good_levels / levels:.3f}")
    if verbose:
        for b in bad[:25]:
            print("  wrong:", b)
    return good_levels / levels
