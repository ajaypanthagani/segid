"""The latent space inside the app, trained only through events (cut the cap, drag it away, drag the bottle), tested on unseen bottles
with nothing moved.  Same bottles and same 'split' score as experiments/bottle_dataset.py, so the numbers compare.

    PYTHONPATH=. .venv/bin/python experiments/latent_events.py [train_n] [probe_n] [hard_n]
"""
import json
import os
import sys
import time

import numpy as np

from experiments.bottles import make_bottle
from objectlearn.paint import PaintApp

TRAIN_N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
PROBE_N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
HARD_N = int(sys.argv[3]) if len(sys.argv) > 3 else 10
CHECK = [c for c in (0, 5, 10, 25, 50, 100, 200, 400, 800) if c <= TRAIN_N]
TAG = os.environ.get("TAG", "events")
DRAG_FIRST = os.environ.get("DRAG_FIRST", "0") == "1"
app = PaintApp(s=5, cap=1100, memory=None, mode="patch", refresh_every=999999, latent=True)
T = [0.0]


def tick(*a):
    T[0] += 0.1
    return app.tick(list(a), T[0], render=False)


def place(img, ycap, rng):
    h, w = img.shape[:2]
    cx = 160 + rng.uniform(-25, 25)
    pid = app.world.add_image(np.ascontiguousarray(img), np.full(img.shape[:2], 255, np.uint8), x=cx)
    y0 = int(app.world.H / 2 - h / 2)
    ys, xs = np.nonzero(app.world.pieces[pid].mask)
    return pid, y0 + ycap, xs.min(), xs.max(), ys.min(), ys.max()


def probe(img, ycap, rng):
    app.clear()
    pid, yc, xa, xb, ytop, ybot = place(img, ycap, rng)
    for _ in range(5):
        tick()
    app.session._lat_due = True
    r = tick()
    pts = np.array(r["pts"])
    cap = pts[:, 1] < yc
    lab = pts[:, 3]
    out = dict(objects=len(r["predicted"]), split=False)
    v, c = np.unique(lab[cap], return_counts=True)
    main = v[np.argmax(c)]
    in_main = lab == main
    out["split"] = bool(c.max() >= 0.7 * cap.sum() and (in_main & ~cap).sum() <= 0.2 * in_main.sum() and (in_main & ~cap).sum() < 0.3 * max(cap.sum(), 1))
    out["exact"] = bool(out["split"] and out["objects"] == 2)
    # tolerant: the cap is one object, and nothing more than two patch rows below the cap edge is in it (where the line was drawn is uncertain by about a row)
    below = in_main & (pts[:, 1] >= yc + 2 * app.s)
    out["near"] = bool((in_main & cap).sum() >= 0.85 * cap.sum() and below.sum() == 0 and (lab != main).any())
    out["near_exact"] = bool(out["near"] and out["objects"] == 2)
    return out


def evaluate(k):
    rng = np.random.default_rng(5)
    res = {}
    for name, n, hard, base in (("unseen", PROBE_N, False, 1000), ("hard", HARD_N, True, 2000)):
        rs = [probe(*make_bottle(np.random.default_rng(base + i), hard=hard), rng) for i in range(n)]
        res[name] = dict(split=f"{sum(r['split'] for r in rs)}/{n}", exact=f"{sum(r['exact'] for r in rs)}/{n}",
                         near=f"{sum(r['near'] for r in rs)}/{n}", near_exact=f"{sum(r['near_exact'] for r in rs)}/{n}",
                         objects=round(float(np.mean([r["objects"] for r in rs])), 2))
    print(f"[{(time.time() - T0) / 60:5.1f} min] after {k:4d} bottles | unseen {res['unseen']} | hard {res['hard']} | latent entries {len(app.session.latent)}", flush=True)
    return res


def train_one(i):
    rng = np.random.default_rng(10 + i)
    img, yc = make_bottle(np.random.default_rng(i))
    app.clear()
    pid, ycut, xa, xb, ytop, ybot = place(img, yc, rng)
    for _ in range(3):
        tick()
    sign = 1 if (xa + xb) / 2 < 160 else -1
    if DRAG_FIRST:                                   # carry the whole bottle first: every cut of it was carried
        tick({"t": "grab", "x": (xa + xb) / 2, "y": (ycut + ybot) / 2})
        for _ in range(12):
            tick({"t": "drag", "dx": 5 * -sign, "dy": 0})
        tick({"t": "release"})
        for _ in range(12):
            tick()
        xa, xb, ytop, ycut, ybot = xa - 60 * sign, xb - 60 * sign, ytop, ycut, ybot
    app.world.cut([[xa - 4, ycut], [xb + 4, ycut]], mark=False)
    for _ in range(4):
        tick()
    tick({"t": "grab", "x": (xa + xb) / 2, "y": (ytop + ycut) / 2})
    for _ in range(16):
        tick({"t": "drag", "dx": 5 * sign, "dy": -1})
    tick({"t": "release"})
    for _ in range(12):
        tick()
    tick({"t": "grab", "x": (xa + xb) / 2, "y": (ycut + ybot) / 2 + 5})        # then carry the bottle itself
    for _ in range(14):
        tick({"t": "drag", "dx": -5 * sign, "dy": 0})
    tick({"t": "release"})
    r = None
    for _ in range(12):
        r = tick()
    return len(r["formal"])


if __name__ == "__main__":
    T0 = time.time()
    curve = {0: evaluate(0)}
    for i in range(1, TRAIN_N + 1):
        n = train_one(i)
        if n != 2:
            print(f"  bottle {i}: {n} objects after the moves", flush=True)
        if i in CHECK:
            curve[i] = evaluate(i)
            json.dump(curve, open(f"experiments/out/latent_events_{TAG}.json", "w"), indent=1)
            app.session.latent.save(f"experiments/out/latent_events_{TAG}.npz")
