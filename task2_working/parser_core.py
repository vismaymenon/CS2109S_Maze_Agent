"""Image -> per-cell entity names (numpy + PIL only). train_final.py pastes this file into the Agent cell."""
from itertools import product

import numpy as np
from PIL import Image

NAMES = ["boots", "box", "coin", "exit", "gem", "ghost", "human", "key", "lava", "locked", "opened", "shield", "wall"]
NAMES14 = NAMES + ["floor"]
F = 16


def cell_features(image, W, H):
    """Cut the RGBA image into W*H cells and shrink each to F x F RGB in [0, 1]."""
    ih, iw = image.shape[:2]
    out = np.empty((H * W, F, F, 3), dtype=np.float32)
    k = 0
    for y in range(H):
        y0, y1 = y * ih // H, (y + 1) * ih // H
        for x in range(W):
            x0, x1 = x * iw // W, (x + 1) * iw // W
            cell = Image.fromarray(np.ascontiguousarray(image[y0:y1, x0:x1, :3]))
            out[k] = np.asarray(cell.resize((F, F), Image.BOX), dtype=np.float32) / 255.0
            k += 1
    return out


def texture_features(X):
    """Edge-direction energy on a 4x4 grid: brick has straight edges, lava has curvy ones (colour-free)."""
    n, f = X.shape[0], X.shape[1]
    g = X.mean(-1)
    dx = np.abs(g[:, :, 1:] - g[:, :, :-1])[:, :-1, :]
    dy = np.abs(g[:, 1:, :] - g[:, :-1, :])[:, :, :-1]
    d1 = np.abs(g[:, 1:, 1:] - g[:, :-1, :-1])
    d2 = np.abs(g[:, 1:, :-1] - g[:, :-1, 1:])
    c = ((f - 1) // 4) * 4
    feats = [d[:, :c, :c].reshape(n, 4, c // 4, 4, c // 4).mean((2, 4)).reshape(n, -1) for d in (dx, dy, d1, d2)]
    return np.concatenate(feats, 1) * 4.0


def all_features(X):
    return np.concatenate([X.reshape(len(X), -1), texture_features(X)], 1)


def mlp_probs(X, w):
    h = np.maximum(all_features(X) @ w["W1"] + w["b1"], 0)
    z = h @ w["W2"] + w["b2"]
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def cluster_by_look(X, tol=0.03):
    """Group cells that look (almost) identical. Returns a list of index lists."""
    flat = X.reshape(len(X), -1)
    reps, members = [], []
    for i in range(len(X)):
        for r, rep in enumerate(reps):
            if np.abs(flat[i] - flat[rep]).mean() < tol:
                members[r].append(i)
                break
        else:
            reps.append(i)
            members.append([i])
    return members


def decide(P, members, first_turn=True, max_uncertain=9):
    """Probabilities (n, 14) -> a set of entity names per cell.

    Uses what the task guarantees: exactly one agent, exactly one exit, equal numbers of keys and
    locked doors, and every door locked at turn 0. Cells that look alike share one label.
    """
    n = len(P)
    P = np.clip(P, 1e-6, 1.0)
    idx = {c: NAMES14.index(c) for c in NAMES14}
    out = [None] * n
    taken = set()
    for name in ("human", "exit"):                       # the one agent and the one exit, cell by cell
        best = max((i for i in range(n) if i not in taken), key=lambda i: P[i, idx[name]])
        out[best] = {name}
        taken.add(best)
    clusters = []                                        # (cells, log-belief, belief) per look-cluster
    for m in members:
        m = [i for i in m if i not in taken]
        if not m:
            continue
        p = P[m].mean(0).copy()
        p[idx["human"]] = p[idx["exit"]] = 0.0
        if first_turn:
            p[idx["locked"]] += p[idx["opened"]]
            p[idx["opened"]] = 0.0
        p = p / p.sum()
        clusters.append((m, np.log(np.clip(p, 1e-9, 1.0)) * len(m), p))
    choice = [int(np.argmax(c[2])) for c in clusters]

    def counts(ch):
        keys = sum(len(c[0]) for c, k in zip(clusters, ch) if k == idx["key"])
        doors = sum(len(c[0]) for c, k in zip(clusters, ch) if k in (idx["locked"], idx["opened"]))
        return keys, doors

    keys, doors = counts(choice)
    if keys != doors:                                    # impossible: re-label the shakiest clusters
        shaky = sorted(range(len(clusters)), key=lambda j: clusters[j][2][choice[j]])[:max_uncertain]
        options = [[int(k) for k in np.argsort(-clusters[j][2])[:3] if clusters[j][2][k] > 0.01] for j in shaky]
        best, best_score = None, -1e18
        for combo in product(*options):
            ch = list(choice)
            for j, k in zip(shaky, combo):
                ch[j] = k
            kk, dd = counts(ch)
            if kk != dd:
                continue
            score = sum(clusters[j][1][ch[j]] for j in range(len(clusters)))
            if score > best_score:
                best, best_score = ch, score
        if best is not None:
            choice = best
    for (m, _, _), k in zip(clusters, choice):
        c = NAMES14[k]
        for i in m:
            out[i] = set() if c == "floor" else {c}
    return out


def classify_grid(image, W, H, w, first_turn=True):
    X = cell_features(image, W, H)
    return decide(mlp_probs(X, w), cluster_by_look(X), first_turn=first_turn)
