"""Learn a small generic filter bank without supervision, from generic synthetic pictures
(random polygons, ellipses, stripes, gradients, noise textures on random backgrounds - nothing bottle-like, no labels).

Spherical k-means on contrast-normalised 5x5 Lab patches (Coates & Ng 2011).  The centroids are the filters.

    PYTHONPATH=. .venv/bin/python experiments/learn_filters.py [K]
"""
import sys

import cv2
import numpy as np

K = int(sys.argv[1]) if len(sys.argv) > 1 else 6
rng = np.random.default_rng(0)


def scene(n=96):
    img = np.empty((n, n, 3), np.float32)
    img[:] = rng.uniform(0, 255, 3)
    yy, xx = np.mgrid[:n, :n] / n
    if rng.random() < .5:
        g = rng.normal(0, 1, 2); img += (g[0] * xx + g[1] * yy)[..., None] * rng.uniform(20, 80)
    for _ in range(int(rng.integers(3, 9))):
        c = tuple(float(v) for v in rng.uniform(0, 255, 3))
        kind = rng.integers(4)
        m = np.zeros((n, n), np.uint8)
        if kind == 0:
            cv2.fillPoly(m, [rng.uniform(0, n, (int(rng.integers(3, 7)), 2)).astype(np.int32)], 1)
        elif kind == 1:
            cv2.ellipse(m, tuple(int(v) for v in rng.uniform(10, n - 10, 2)), tuple(int(v) for v in rng.uniform(4, 30, 2)), float(rng.uniform(0, 180)), 0, 360, 1, -1)
        elif kind == 2:
            x0, y0 = rng.integers(0, n - 8, 2); cv2.rectangle(m, (int(x0), int(y0)), (int(x0 + rng.integers(4, 50)), int(y0 + rng.integers(3, 40))), 1, -1)
        else:
            m[:] = (np.sin((xx * rng.uniform(5, 40) + yy * rng.uniform(-30, 30)) * 6.28) > 0.3)
            m &= (np.abs(xx - rng.random()) < rng.uniform(.1, .6)).astype(np.uint8)
        img[m > 0] = c
        if rng.random() < .3:
            img[m > 0] += rng.normal(0, rng.uniform(3, 25), (int(m.sum()), 1))
    img += rng.normal(0, rng.uniform(0, 5), img.shape)
    img = cv2.GaussianBlur(np.clip(img, 0, 255), (0, 0), rng.uniform(0.3, 1.0))
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2Lab).astype(np.float32)


X = []
for _ in range(300):
    im = scene()
    for _ in range(300):
        y, x = rng.integers(0, 91, 2)
        p = im[y:y + 5, x:x + 5].reshape(-1)
        X.append(p)
X = np.array(X, np.float32)
X = np.concatenate([X, X.reshape(-1, 5, 5, 3)[:, :, ::-1].reshape(len(X), -1)])         # mirror-augmented
X = X - X.mean(1, keepdims=True)
nrm = np.linalg.norm(X, axis=1)
X = X[nrm > 25]; X /= np.linalg.norm(X, axis=1, keepdims=True)
print(len(X), "patches")
C = X[rng.choice(len(X), K, replace=False)]
for it in range(50):
    a = np.argmax(X @ C.T, 1)
    for k in range(K):
        if (a == k).any():
            v = X[a == k].sum(0); C[k] = v / np.linalg.norm(v)
F = C.reshape(K, 5, 5, 3)
np.savez("objectlearn/filters_generic.npz", F=F)
print("saved", F.shape, "cluster sizes", np.bincount(a, minlength=K))
cv2.imwrite("experiments/out/filters_generic.png", cv2.resize(np.hstack([np.clip(128 + 400 * (f[..., 0:1].repeat(3, 2)), 0, 255).astype(np.uint8) for f in F]), None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST))
