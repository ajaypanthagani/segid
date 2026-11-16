"""A small 3-D world used to exercise the model.

Rigid axis-aligned boxes sit on a floor (optionally stacked).  Every surface is
tiled into small patches (the "patches"); patches keep their identity as bodies
move, so patch correspondence across time is given.  A fixed camera sees only
front-facing, unoccluded patches, and the sensed positions are noisy.

Ground truth (`patch_body`) is exposed only for evaluation.  The model never reads it.
Bodies are rigid and translate only.  A body that is moved carries whatever rests
on it (friction), which is exactly the situation that produces weak "carried" evidence
for pairs that are in fact distinct objects.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    pos: np.ndarray   # (n, 3) sensed positions, valid where vis
    nrm: np.ndarray   # (n, 3) sensed unit normals
    col: np.ndarray   # (n,)   sensed intensity
    vis: np.ndarray   # (n,)   bool
    t: int = 0
    img: object = None   # optional BGR image (paint screen)
    patch_rgb: object = None   # optional (n, 3) uint8 BGR colour each patch sampled


class Body:
    def __init__(self, bid, origin, cells, color, floor=False):
        self.id, self.origin, self.cells, self.color, self.floor = bid, np.array(origin, float), tuple(cells), color, floor


def _patches(cells, s):
    nx, ny, nz = cells
    P, N = [], []
    for i in range(nx):
        for j in range(ny):
            P.append(((i + .5) * s, (j + .5) * s, nz * s)); N.append((0, 0, 1))
    for j in range(ny):
        for k in range(nz):
            P.append((nx * s, (j + .5) * s, (k + .5) * s)); N.append((1, 0, 0))
            P.append((0.0, (j + .5) * s, (k + .5) * s)); N.append((-1, 0, 0))
    for i in range(nx):
        for k in range(nz):
            P.append(((i + .5) * s, ny * s, (k + .5) * s)); N.append((0, 1, 0))
            P.append(((i + .5) * s, 0.0, (k + .5) * s)); N.append((0, -1, 0))
    return np.array(P, float).reshape(-1, 3), np.array(N, float).reshape(-1, 3)


class World:
    def __init__(self, bodies, s, rng, floor_cells, sigma_pos=0.001, sigma_col=0.01):
        self.s, self.rng = s, rng
        self.bodies = bodies
        self.sigma_pos, self.sigma_col = sigma_pos, sigma_col
        loc, nrm, pb, base = [], [], [], []
        for b in bodies:
            p, n = _patches(b.cells, s)
            loc.append(p); nrm.append(n); pb += [b.id] * len(p)
            base.append(np.full(len(p), b.color))
        self.local = np.concatenate(loc)
        self.nrm = np.concatenate(nrm)
        self.patch_body = np.array(pb)
        self.base_col = np.concatenate(base)
        self.texture = rng.normal(0, 0.03, len(self.local))
        self.n = len(self.local)
        self.origin = np.array([b.origin for b in bodies])
        self.offset = np.zeros((len(bodies), 3))
        w = floor_cells * s
        self.cam = np.array([w / 2, -0.9, 1.0])
        self.t = 0

    # ---------------------------------------------------------------- geometry
    def _lo(self, b):
        return self.origin[b] + self.offset[b]

    def _size(self, b):
        return np.array(self.bodies[b].cells, float) * self.s

    def top(self, b):
        return self._lo(b)[2] + self._size(b)[2]

    def riders(self, bid):
        out, stack = [], [bid]
        while stack:
            a = stack.pop()
            lo_a, sz_a = self._lo(a), self._size(a)
            for b in self.bodies:
                if b.floor or b.id == bid or b.id in out or b.id == a:
                    continue
                lo_b, sz_b = self._lo(b.id), self._size(b.id)
                on_top = abs(lo_b[2] - (lo_a[2] + sz_a[2])) < 1e-6
                ov = np.all(np.minimum(lo_a[:2] + sz_a[:2], lo_b[:2] + sz_b[:2]) - np.maximum(lo_a[:2], lo_b[:2]) > 1e-6)
                if on_top and ov:
                    out.append(b.id); stack.append(b.id)
        return out

    def translate(self, bid, d):
        for b in [bid] + self.riders(bid):
            self.offset[b] += d

    # ------------------------------------------------------------------ sensing
    def frame(self):
        self.t += 1
        true = self.local + self.offset[self.patch_body] + self.origin[self.patch_body]
        vec = self.cam - true
        facing = (self.nrm * vec).sum(1) > 0
        p0 = true + 1e-3 * self.nrm
        d = self.cam - p0
        d = np.where(np.abs(d) < 1e-12, 1e-12, d)
        occl = np.zeros(self.n, bool)
        for b in self.bodies:
            if b.floor:
                continue
            lo = self._lo(b.id); hi = lo + self._size(b.id)
            t1 = (lo - p0) / d; t2 = (hi - p0) / d
            tn = np.minimum(t1, t2).max(1); tf = np.maximum(t1, t2).min(1)
            hit = (tn <= tf) & (tf > 0) & (tn < 1) & (self.patch_body != b.id)
            occl |= hit
        vis = facing & ~occl
        pos = true + self.rng.normal(0, self.sigma_pos, true.shape)
        nr = self.nrm + self.rng.normal(0, 0.02, true.shape)
        nr /= np.linalg.norm(nr, axis=1, keepdims=True)
        col = self.base_col[...] + self.texture + self.rng.normal(0, self.sigma_col, self.n)
        return Frame(pos=pos, nrm=nr, col=col, vis=vis, t=self.t)

    def static(self, m):
        for _ in range(m):
            yield self.frame()

    # ------------------------------------------------------------------ actions
    def lift(self, bid, dz, steps=8, hold=3):
        for _ in range(steps):
            self.translate(bid, np.array([0, 0, dz / steps])); yield self.frame()
        for _ in range(hold):
            yield self.frame()

    def shift(self, bid, dxy, steps=6, hold=2):
        for _ in range(steps):
            self.translate(bid, np.array([dxy[0] / steps, dxy[1] / steps, 0.0])); yield self.frame()
        for _ in range(hold):
            yield self.frame()

    def lower(self, bid, steps=8, hold=3):
        group = [bid] + self.riders(bid)
        lo, sz = self._lo(bid), self._size(bid)
        floor_z = 0.0
        for b in self.bodies:
            if b.floor or b.id in group:
                continue
            lb, sb = self._lo(b.id), self._size(b.id)
            ov = np.all(np.minimum(lo[:2] + sz[:2], lb[:2] + sb[:2]) - np.maximum(lo[:2], lb[:2]) > 1e-6)
            if ov and lb[2] + sb[2] <= lo[2] + 1e-9:
                floor_z = max(floor_z, lb[2] + sb[2])
        dz = lo[2] - floor_z
        for _ in range(steps):
            self.translate(bid, np.array([0, 0, -dz / steps])); yield self.frame()
        for _ in range(hold):
            yield self.frame()

    def pick_and_place(self, bid, dz, dxy):
        yield from self.lift(bid, dz)
        yield from self.shift(bid, dxy)
        yield from self.lower(bid)

    def drag(self, bid, dxy, steps=14, hold=4):
        yield from self.shift(bid, dxy, steps=steps, hold=hold)


# ---------------------------------------------------------------------- scenes
def make_scene(rng, kind="train", s=0.05, W=16):
    """Random scene.  kind='train': low wide boxes with stacked children.
    kind='bottle': tall thin bodies with a cap on top (unseen shapes), plus one low box."""
    bodies = [Body(0, (0, 0, 0), (W, W, 0), 0.5 + rng.normal(0, 0.02), floor=True)]
    placed = []        # (x0, y0, nx, ny, top_cells, id)

    def free(x0, y0, nx, ny, gap=2):
        if x0 < 1 or y0 < 1 or x0 + nx > W - 1 or y0 + ny > W - 1:
            return False
        for (px, py, pnx, pny, _, _) in placed:
            if x0 < px + pnx + gap and px < x0 + nx + gap and y0 < py + pny + gap and py < y0 + ny + gap:
                return False
        return True

    def add(x0, y0, z0, nx, ny, nz, color):
        bid = len(bodies)
        bodies.append(Body(bid, (x0 * s, y0 * s, z0 * s), (nx, ny, nz), color))
        return bid

    def place_base(nx, ny, nz, tries=200):
        for _ in range(tries):
            x0, y0 = int(rng.integers(1, W - nx)), int(rng.integers(1, W - ny))
            if free(x0, y0, nx, ny):
                bid = add(x0, y0, 0, nx, ny, nz, float(rng.uniform(0.15, 0.95)))
                placed.append((x0, y0, nx, ny, nz, bid))
                return bid, x0, y0
        return None

    def stack(base, cx, cy, cnz, nz_base, flush_prob=0.4):
        bid, x0, y0 = base
        _, _, nx, ny, _, _ = placed[[p[5] for p in placed].index(bid)]
        cnx = int(rng.integers(2, max(3, nx)))
        cny = int(rng.integers(2, max(3, ny)))
        if rng.random() < flush_prob:
            cnx, cny, ox, oy = nx, ny, 0, 0
        else:
            cnx, cny = min(cnx, nx), min(cny, ny)
            ox, oy = int(rng.integers(0, nx - cnx + 1)), int(rng.integers(0, ny - cny + 1))
        add(x0 + ox, y0 + oy, nz_base, cnx, cny, cnz, float(rng.uniform(0.15, 0.95)))

    if kind == "train":
        for _ in range(int(rng.integers(3, 5))):
            nx, ny, nz = int(rng.integers(3, 6)), int(rng.integers(3, 6)), int(rng.integers(2, 4))
            b = place_base(nx, ny, nz)
            if b and rng.random() < 0.45:
                stack(b, 0, 0, int(rng.integers(1, 3)), nz)
    elif kind == "bottle":
        for _ in range(int(rng.integers(2, 4))):
            n = int(rng.integers(2, 4))
            nz = int(rng.integers(5, 8))
            b = place_base(n, n, nz)
            if b:
                if rng.random() < 0.5:
                    add(b[1], b[2], nz, n, n, 1, float(rng.uniform(0.15, 0.95)))      # flush cap
                else:
                    add(b[1] + (1 if n > 2 else 0), b[2] + (1 if n > 2 else 0), nz, max(2, n - 2), max(2, n - 2), 1,
                        float(rng.uniform(0.15, 0.95)))
        place_base(4, 4, 2)
    else:
        raise ValueError(kind)
    return World(bodies, s, rng, floor_cells=W)
