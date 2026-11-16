"""Pilot of the latent space on bottles (no event simulation: the outcome of the cut-and-move protocol is written in directly).

Training bottle -> entries:
  whole bottle   the candidate cut that gives exactly the cap as its own part: behaviour 'separates' = 1 (the cap was seen to come away).
                 Every other candidate of the whole bottle stays unobserved (nothing moved relative to it): not stored.
  body alone     once the cap is gone the body is dragged: all its candidates were carried whole: 'separates' = 0 (weight 0.7)
  cap alone      the cap is dragged: all its candidates carried whole: 0 (weight 0.7)
Testing: a new bottle, nothing moved: score every candidate cut with the memory and see what comes out as 'separates'.

    PYTHONPATH=. .venv/bin/python experiments/latent_pilot.py [train_n] [probe_n] [hard_n]
"""
import json
import os
import sys
import time

import numpy as np

from experiments.bottles import make_bottle
from objectlearn.latent import LatentMemory
from objectlearn.paint import PaintApp
from objectlearn.traits import candidates, to_lab

TRAIN_N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
PROBE_N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
HARD_N = int(sys.argv[3]) if len(sys.argv) > 3 else 10
CHECK = [c for c in (10, 25, 50, 100, 200, 400, 800, 1600) if c <= TRAIN_N]
S = int(os.environ.get("S", 5))
RELEVANCE = os.environ.get("REL", "1") == "1"
app = PaintApp(s=S, cap=1100 if S >= 5 else 3000, memory=None, mode="patch", refresh_every=999999)


def scene(img, ycap, rng):
    app.clear()
    h, w = img.shape[:2]
    cx = 160 + rng.uniform(-25, 25)
    app.world.add_image(np.ascontiguousarray(img), np.full(img.shape[:2], 255, np.uint8), x=cx)
    y0 = int(app.world.H / 2 - h / 2)
    fr = app.world.frame()[0]
    v = np.flatnonzero(fr.vis)
    return fr.pos[v, :2].astype(float), to_lab(fr.patch_rgb[v]), y0 + ycap


def cands(pos, sig):
    return candidates(pos, sig, float(S))


def truth(pos, masks, yc):
    cap = pos[:, 1] < yc
    eq = np.array([(m == cap).all() for m in masks])
    iou = np.array([(m & cap).sum() / max((m | cap).sum(), 1) for m in masks])
    return eq, iou


def train_one(mem, i, rng):
    img, yc = make_bottle(np.random.default_rng(i))
    pos, sig, yc = scene(img, yc, np.random.default_rng(10 + i))
    X, masks, meta = cands(pos, sig)
    eq, _ = truth(pos, masks, yc)
    rows, y, w = [], [], []
    for k in np.flatnonzero(eq)[:1]:
        rows.append(X[k]); y.append(1.0); w.append(1.0)
    for sel in (pos[:, 1] >= yc, pos[:, 1] < yc):
        if sel.sum() < 12:
            continue
        Xs, _, _ = cands(pos[sel], sig[sel])
        take = rng.choice(len(Xs), min(len(Xs), 40), replace=False)
        for k in take:
            rows.append(Xs[k]); y.append(0.0); w.append(0.7)
    if rows:
        mem.add(np.array(rows), np.array(y), np.array(w))
    return bool(eq.any())


def probe(mem, img, ycap, rng):
    pos, sig, yc = scene(img, ycap, rng)
    X, masks, meta = cands(pos, sig)
    eq, iou = truth(pos, masks, yc)
    q = mem.query(X)
    if not eq.any():
        return dict(found=False)
    t = np.flatnonzero(eq)[0]
    order = np.argsort(-q["p"])
    rank = int(np.flatnonzero(order == t)[0]) + 1
    near = (iou > 0.5) & ~eq
    wrong = (q["statement"] == 1) & (iou <= 0.5)
    # one decision per cluster of overlapping cuts
    keep = []
    for k in np.argsort(-q["p"]):
        if q["statement"][k] != 1:
            break
        if all(((masks[k] & masks[j]).sum() / max((masks[k] | masks[j]).sum(), 1)) < 0.5 for j in keep):
            keep.append(k)
    wrong_clusters = [k for k in keep if iou[k] <= 0.5]
    return dict(found=True, p_true=float(q["p"][t]), rank=rank, stated=bool(q["statement"][t] == 1 or (q["statement"][near] == 1).any()),
                exact_stated=bool(q["statement"][t] == 1), top1=bool(rank == 1), n_cand=len(X), wrong=int(wrong_clusters.__len__()),
                nearest=float(q["nearest"][t]))


def evaluate(mem, tag, hard):
    n = HARD_N if hard else PROBE_N
    base = 2000 if hard else 1000
    rs = [probe(mem, *make_bottle(np.random.default_rng(base + i), hard=hard), np.random.default_rng(5 + i)) for i in range(n)]
    rs = [r for r in rs if r["found"]]
    f = lambda k: float(np.mean([r[k] for r in rs]))
    return dict(top1=round(f("top1"), 2), stated=round(f("stated"), 2), exact_stated=round(f("exact_stated"), 2),
                mean_rank=round(f("rank"), 1), wrong_per_bottle=round(f("wrong"), 2), p_true=round(f("p_true"), 2), n=len(rs))


if __name__ == "__main__":
    mem = LatentMemory(relevance=RELEVANCE)
    rng = np.random.default_rng(0)
    t0 = time.time()
    curve = {}
    for i in range(1, TRAIN_N + 1):
        train_one(mem, i, rng)
        if i in CHECK:
            r = dict(unseen=evaluate(mem, "unseen", False), hard=evaluate(mem, "hard", True), rows=len(mem))
            curve[i] = r
            print(f"[{(time.time() - t0) / 60:4.1f} min] {i:4d} bottles | unseen {r['unseen']} | hard {r['hard']} | rows {len(mem)}", flush=True)
            json.dump(curve, open(f"experiments/out/latent_pilot_{os.environ.get('TAG', 'rel' if RELEVANCE else 'norel')}.json", "w"), indent=1)
    mem.save(f"experiments/out/latent_{os.environ.get('TAG', 'rel' if RELEVANCE else 'norel')}.npz")
