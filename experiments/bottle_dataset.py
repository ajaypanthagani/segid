"""Teach the model on many different bottles (cut the cap, drag it away), and test on bottles it has never seen.

At checkpoints it is probed, frozen (nothing moves, so nothing is learned), on
  - 'unseen': new bottles from the same generator
  - 'hard'  : new bottles with colours, proportions and cap styles outside the training range
Scores (against where the cap really ends, which the model never sees):
  recall   share of pairs straddling the cap/body line that the model marks 'distinct'
  false    share of other pairs it marks 'distinct'
  split    the predicted objects contain the cap as its own object, without any move

    PYTHONPATH=. .venv/bin/python experiments/bottle_dataset.py [train_n] [probe_n] [hard_n]
"""
import json
import sys

import numpy as np

from experiments.bottles import make_bottle
from objectlearn.paint import PaintApp

TRAIN_N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
PROBE_N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
HARD_N = int(sys.argv[3]) if len(sys.argv) > 3 else 10
CHECK = [c for c in (0, 5, 10, 20, 30, 40, 60, 80, 100, 150, 200, 300, 400, 500, 600, 750, 1000, 1500, 2000, 3000, 4000, 5000) if c <= TRAIN_N]

import os
if os.environ.get("CHECKS"):
    CHECK = [int(c) for c in os.environ["CHECKS"].split(",")]
MODE = os.environ.get("MODE", "image")
TAG = os.environ.get("TAG", MODE)
S = int(os.environ.get("S", 5))
app = PaintApp(s=S, cap=(1100 if S >= 5 else 3000) if MODE == "patch" else 2500, memory=None, mode=MODE,
               refresh_every=999999 if MODE == "patch" else 3)
T = [0.0]


def tick(*a):
    T[0] += 0.1
    return app.tick(list(a), T[0], render=False)


def place(img, ycap, rng):
    h, w = img.shape[:2]
    cx = 160 + rng.uniform(-25, 25)
    pid = app.world.add_image(np.ascontiguousarray(img), np.full(img.shape[:2], 255, np.uint8), x=cx)
    y0 = int(app.world.H / 2 - h / 2)
    x0 = int(np.clip(int(cx - w / 2), 0, app.world.W - w))
    ys, xs = np.nonzero(app.world.pieces[pid].mask)
    return pid, y0 + ycap, xs.min(), xs.max(), ys.min()


def probe(img, ycap, rng):
    app.clear()
    pid, yc, xa, xb, ytop = place(img, ycap, rng)
    for _ in range(5):
        r = tick()
    ag = app.session.agent
    fr = app.world.frame()[0]
    ag.refresh(fr)
    r = tick()
    ii, jj, rows = ag.interfaces()
    vis = fr.vis[ii] & fr.vis[jj]
    capside = lambda k: fr.pos[k, 1] < yc
    straddle = np.array([capside(a) != capside(b) for a, b in zip(ii, jj)]) & vis
    other = ~straddle & vis
    dist = ag.stat_now[rows] == 1
    pts = np.array(r["pts"])
    cap = pts[:, 1] < yc
    out = dict(recall=float(dist[straddle].mean()) if straddle.any() else np.nan,
               false=float(dist[other].mean()) if other.any() else np.nan, n_straddle=int(straddle.sum()),
               objects=len(r["predicted"]))
    if cap.any():
        lab = pts[:, 3]
        v, c = np.unique(lab[cap], return_counts=True)
        main = v[np.argmax(c)]
        in_main = lab == main
        out["split"] = bool(c.max() >= 0.7 * cap.sum() and (in_main & ~cap).sum() <= 0.2 * in_main.sum() and (in_main & ~cap).sum() < 0.3 * max(cap.sum(), 1))
    else:
        out["split"] = False
    return out


def summarize(rs):
    return dict(recall=round(float(np.nanmean([r["recall"] for r in rs])), 2), false=round(float(np.nanmean([r["false"] for r in rs])), 3),
                split=f"{sum(r['split'] for r in rs)}/{len(rs)}", objects=round(float(np.mean([r["objects"] for r in rs])), 1))


def evaluate(k):
    rng = np.random.default_rng(5)
    res = {}
    for name, n, hard, base in (("unseen", PROBE_N, False, 1000), ("hard", HARD_N, True, 2000)):
        rs = []
        for i in range(n):
            img, yc = make_bottle(np.random.default_rng(base + i), hard=hard)
            rs.append(probe(img, yc, rng))
        res[name] = summarize(rs)
    print(f"[{(time.time() - T0) / 60:5.1f} min] after {k:4d} bottles | unseen {res['unseen']} | hard {res['hard']} | memory {len(app.session.agent.mem)}", flush=True)
    return res


def train_one(i):
    rng = np.random.default_rng(10 + i)
    img, yc = make_bottle(np.random.default_rng(i))
    app.clear()
    pid, ycut, xa, xb, ytop = place(img, yc, rng)
    for _ in range(3):
        tick()
    app.world.cut([[xa - 4, ycut], [xb + 4, ycut]], mark=False)
    for _ in range(4):
        tick()
    sign = 1 if (xa + xb) / 2 < 160 else -1
    tick({"t": "grab", "x": (xa + xb) / 2, "y": (ytop + ycut) / 2})
    for _ in range(16):
        r = tick({"t": "drag", "dx": 5 * sign, "dy": -1})
    tick({"t": "release"})
    for _ in range(12):
        r = tick()
    return len(r["formal"])


import time
T0 = time.time()
curve = {0: evaluate(0)}
for i in range(1, TRAIN_N + 1):
    n = train_one(i)
    if n != 2:
        print(f"  bottle {i}: {n} objects after the move", flush=True)
    if i in CHECK:
        curve[i] = evaluate(i)
        json.dump(curve, open(f"experiments/out/bottles_curve_{TAG}.json", "w"), indent=1)
        app.session.agent.save(f"experiments/out/bottles_{TAG}.npz")

for sp in [float(v) for v in os.environ.get("SPREAD_SWEEP", "").split(",") if v]:
    import objectlearn.agent as _A
    _A.SPREAD = sp
    print(f"spread {sp}:", end=" ")
    curve[f"spread{sp}"] = evaluate(TRAIN_N)
json.dump(curve, open(f"experiments/out/bottles_curve_{TAG}.json", "w"), indent=1)
