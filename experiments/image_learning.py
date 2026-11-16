"""Repeat: add the bottle picture, cut the cap, drag it away.  After each round, probe with a fresh copy of the picture
(not moved, not cut) and see whether the model marks the cap/body seam and splits the cap off by itself.

    PYTHONPATH=. .venv/bin/python experiments/image_learning.py IMAGE.png [rounds] [cap_bottom_fraction] [vary]
"""
import sys

import cv2
import numpy as np

from objectlearn.paint import PaintApp

path = sys.argv[1]
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 12
cap_frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.135      # where the cap ends, as a fraction of image height
vary = (sys.argv[4] == "vary") if len(sys.argv) > 4 else False
src = cv2.imread(path)
rng = np.random.default_rng(0)
MARK = len(sys.argv) > 6 and sys.argv[6] == "mark"
S = int(sys.argv[5]) if len(sys.argv) > 5 else 10
import os
app = PaintApp(s=S, cap=2500 if S < 10 else 2000, memory=os.environ.get('MEMORY'))
T = [0.0]


def tick(*a):
    T[0] += 0.1
    return app.tick(list(a), T[0])


def place():
    img = src
    if vary:
        k = rng.uniform(0.8, 1.0)
        img = cv2.resize(src, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(int)
        hsv[..., 0] = (hsv[..., 0] + rng.integers(-8, 9)) % 180
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        if rng.random() < .5:
            img = img[:, ::-1].copy()
    x = app.W / 2 + (rng.uniform(-30, 30) if vary else 0)
    pid = app.world.add_image(np.ascontiguousarray(img), np.full(img.shape[:2], 255, np.uint8), x=x)
    return pid


def cap_geometry(pid):
    m = app.world.pieces[pid].mask
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    return y0, y1, x0, x1, y0 + cap_frac * (y1 - y0)


def probe(i):
    app.clear()
    pid = place()
    y0, y1, x0, x1, ycut = cap_geometry(pid)
    for _ in range(4):
        r = tick()
    seams = r["seams"]
    near = [q for q in seams if abs((q[1] + q[3]) / 2 - ycut) < 8]
    far = len(seams) - len(near)
    pred = r["predicted"]
    capsep = any(o["box"][3] < ycut + 8 and o["box"][1] < y0 + 12 for o in pred) and len(pred) > 1
    print(f"after {i:2d} rounds: seam ticks at cap line {len(near):2d}, elsewhere {far:2d}, predicted objects {len(pred)}"
          f"{'  <- cap split off before any move' if capsep else ''}", flush=True)


def round_(i):
    app.clear()
    pid = place()
    y0, y1, x0, x1, ycut = cap_geometry(pid)
    for _ in range(3):
        r = tick()
    n0 = len(r["formal"])
    app.world.cut([[x0 - 4, ycut], [x1 + 4, ycut]], mark=MARK)
    tick()
    for _ in range(4):
        r = tick()
    tick({"t": "grab", "x": (x0 + x1) / 2, "y": (y0 + ycut) / 2 - 2})
    for _ in range(16):
        r = tick({"t": "drag", "dx": 5, "dy": -1})
    tick({"t": "release"})
    for _ in range(12):
        r = tick()
    return n0, len(r["formal"]), r["stats"]["samples"]


probe(0)
for i in range(1, rounds + 1):
    n0, n1, samples = round_(i)
    print(f"round {i}: objects before cut {n0}, after moving cap {n1}, memory {samples} samples", flush=True)
    probe(i)
