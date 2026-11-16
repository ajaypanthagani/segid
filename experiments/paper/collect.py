"""Run the real system (paint world -> patches -> formal layer -> latent learning) on every scene, and cache what it produced.

Training scenes: the scene is placed on blank paper and handled by a scripted hand:
    1. carry the whole scene a little (everything in it was carried: weak 'stays joined' entries)
    2. take away up to 3 of its units, one at a time (a cap, a book, a drawer, a cup from a table; a cap off a bottle only after
       the bottle has been taken off the table): the unit is freed along its outline and dragged to a free place on the paper
    3. carry what is left
  Things that never come apart are carried twice.  The formal layer decides what came apart; the session turns that into latent
  entries.  Each scene runs with an empty memory and its own scene state, so the entries of any training set are the union of its
  scenes' entries and can be assembled afterwards without rerunning the world.

Probe scenes: placed and left still.  For every formal object we cache every candidate part (traits and patch mask) and the
object's contact graph, and for every patch its ground-truth segment, so any memory can be scored afterwards exactly as the
session would predict.

    PYTHONPATH=. .venv/bin/python experiments/paper/collect.py train <family> <n> <seed0>
    PYTHONPATH=. .venv/bin/python experiments/paper/collect.py probe <family> <n> <seed0> <transform>
"""
import sys
import time

import numpy as np

from experiments.paper.scenes import GROUP, make, partition, transform
from objectlearn.latent import LatentMemory
from objectlearn.paint import PaintApp
from objectlearn.regions import ObjectGeom, candidate_masks, describe
from objectlearn.traits import to_lab

S, W, H, CAP = 5, 480, 320, 2800
OUT = "experiments/paper/out"
SCENE_AT = (140, 165)
SLOTS = [(345, 70), (430, 70), (345, 175), (430, 175), (345, 270), (430, 270)]


class Rig:
    """The paint world and the session, sized for one scene: the formal layer keeps n x n pair state, so n is the scene's own patch
    count rather than the canvas's capacity (this changes nothing but speed)."""

    def __init__(self):
        self.app = None
        self.t = 0.0

    def fresh(self, img):
        from objectlearn.paint import PaintWorld
        tmp = PaintWorld(W, H, S, CAP)
        h, wd = img.shape[:2]
        tmp.add_image(np.ascontiguousarray(img), np.full((h, wd), 255, np.uint8), x=SCENE_AT[0], y=SCENE_AT[1])
        self.app = PaintApp(W=W, H=H, s=S, cap=max(tmp.n, 16), memory=None, mode="patch", refresh_every=10 ** 9, latent=True)

    def tick(self, *a):
        self.t += 0.1
        return self.app.tick(list(a), self.t, render=False)

    def place(self, img):
        w = self.app.world
        h, wd = img.shape[:2]
        pid = w.add_image(np.ascontiguousarray(img), np.full((h, wd), 255, np.uint8), x=SCENE_AT[0], y=SCENE_AT[1])
        x0 = int(np.clip(int(SCENE_AT[0] - wd / 2), 0, w.W - wd))
        y0 = int(np.clip(int(SCENE_AT[1] - h / 2), 0, w.H - h))
        return pid, x0, y0

    def owner(self):
        return self.app.world.render()[1]

    def grab_point(self, canvas_mask, pid):
        """A pixel of canvas_mask, near its middle, where the piece pid is the one on top."""
        own = self.owner() == pid
        m = canvas_mask & own
        if not m.any():
            return None
        ys, xs = np.nonzero(m)
        k = np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2)
        return float(xs[k]), float(ys[k])

    def drag(self, at, D, steps):
        d = np.round(np.asarray(D, float) / steps).astype(int)
        self.tick({"t": "grab", "x": at[0], "y": at[1]})
        for _ in range(steps):
            self.tick({"t": "drag", "dx": int(d[0]), "dy": int(d[1])})
        self.tick({"t": "release"})
        r = None
        for _ in range(12):
            r = self.tick()
        return r, d * steps


def _on_canvas(mask, off):
    """Image-space mask placed on the canvas at offset off = (x0, y0)."""
    out = np.zeros((H, W), bool)
    h, w = mask.shape
    x0, y0 = int(off[0]), int(off[1])
    ys, xs = np.nonzero(mask)
    ys, xs = ys + y0, xs + x0
    ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
    out[ys[ok], xs[ok]] = True
    return out


def train_scene(rig, family, seed):
    sc = make(family, seed)
    rng = np.random.default_rng(seed + 7)
    rig.fresh(sc["img"])
    app = rig.app
    pid, x0, y0 = rig.place(sc["img"])
    for _ in range(3):
        rig.tick()
    obj = _on_canvas(sc["img"].min(2) < 245, (x0, y0))
    shift = {None: np.array([x0, y0])}
    side = 1 if rng.random() < 0.5 else -1
    r, D = rig.drag(rig.grab_point(obj, pid), (25 * side, 0), 10)                 # 1. carry the whole scene
    shift[None] = shift[None] + D
    units = sc["units"]
    # 2. which units to take away: parents before their own units, at most 3
    top = [i for i, u in enumerate(units) if u["parent"] is None]
    if family == "book_stack":
        top = top[::-1]                      # a stack is taken from the top down
    else:
        rng.shuffle(top)
    order = []
    for i in top:
        order.append(i)
        kids = [j for j, u in enumerate(units) if u["parent"] == i]
        if kids and rng.random() < 0.7:
            order.append(kids[0])
        if len(order) >= 3:
            break
    order = order[:3]
    for i in range(len(units)):
        shift.setdefault(i, None)
    taken, slot = [], 0
    for i in order:
        u = units[i]
        base = shift[u["parent"]] if u["parent"] is not None else shift[None]
        if base is None:
            continue
        cm = _on_canvas(u["mask"], base)
        nid = app.world.detach(cm)
        if nid is None:
            continue
        for _ in range(3):
            rig.tick()
        at = rig.grab_point(cm, nid)
        if at is None:
            continue
        ys, xs = np.nonzero(cm)
        target = np.array(SLOTS[slot % len(SLOTS)], float); slot += 1
        r, D = rig.drag(at, target - np.array([xs.mean(), ys.mean()]), 18)
        shift[i] = base + D
        for j, v in enumerate(units):                                             # its own units moved with it
            if v["parent"] == i:
                shift[j] = shift[i]
        taken.append(i)
    # 3. carry what is left (twice for things that never come apart)
    rest = obj.copy()
    rest_img = sc["img"].min(2) < 245
    for i in taken:
        if units[i]["parent"] is None:
            rest_img = rest_img & ~units[i]["mask"]
    rest = _on_canvas(rest_img, shift[None])
    at = rig.grab_point(rest, pid)
    if at is not None:
        r, D = rig.drag(at, (-25 * side, 0), 10)
        shift[None] = shift[None] + D
    if not units:
        at = rig.grab_point(_on_canvas(rest_img, shift[None]), pid)
        if at is not None:
            r, D = rig.drag(at, (0, -20), 10)
    mem = app.session.latent
    keep = mem.w > 0
    return dict(X=mem.X[keep] if mem.X is not None else np.zeros((0, 26), np.float32), y=mem.Y[keep, 0], w=mem.w[keep],
                formal=len(r["formal"]), formal_big=sum(o["n"] >= 5 for o in r["formal"]), want=1 + len(taken), taken=len(taken), events=app.session.n_events)


def probe_scene(rig, family, seed, kind):
    sc = transform(make(family, seed), "none" if kind == "val" else kind)      # "val": unseen scenes kept apart for choosing settings
    rig.fresh(sc["img"])
    app = rig.app
    pid, x0, y0 = rig.place(sc["img"])
    for _ in range(5):
        rig.tick()
    fr, _ = app.world.frame()
    lf = app.session._raw_f
    vis = np.flatnonzero(fr.vis & (lf >= 0))
    pos = fr.pos[vis, :2]
    lab = partition(sc)
    ix = np.clip(np.rint(pos[:, 0] - x0).astype(int), 0, lab.shape[1] - 1)
    iy = np.clip(np.rint(pos[:, 1] - y0).astype(int), 0, lab.shape[0] - 1)
    gt = lab[iy, ix]
    formal = lf[vis]
    objects = []
    for o in np.unique(formal):
        loc = np.flatnonzero(formal == o)
        if len(loc) < 12:
            objects.append(dict(loc=loc, X=np.zeros((0, 26), np.float32), masks=np.zeros((0, len(loc)), bool), kind=np.zeros(0, "<U6"),
                                i4=np.zeros(0, int), j4=np.zeros(0, int)))
            continue
        geo = ObjectGeom(pos[loc], to_lab(fr.patch_rgb[vis[loc]]), S)
        masks, kd = candidate_masks(geo)
        objects.append(dict(loc=loc, X=describe(geo, masks), masks=masks, kind=kd, i4=geo.i4, j4=geo.j4))
    return dict(family=family, kind=kind, pos=pos, gt=gt, formal=formal, objects=objects, n_units=len(sc["units"]), group=GROUP[family])


def main():
    mode, family, n, seed0 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    rig = Rig()
    t0 = time.time()
    if mode == "train":
        recs = []
        for i in range(n):
            recs.append(train_scene(rig, family, seed0 + i))
        lens = np.array([len(r["y"]) for r in recs])
        np.savez_compressed(f"{OUT}/train_{family}.npz", X=np.concatenate([r["X"] for r in recs]), y=np.concatenate([r["y"] for r in recs]),
                            w=np.concatenate([r["w"] for r in recs]), obj=np.repeat(np.arange(n), lens),
                            formal=np.array([r["formal"] for r in recs]), formal_big=np.array([r["formal_big"] for r in recs]), want=np.array([r["want"] for r in recs]),
                            taken=np.array([r["taken"] for r in recs]), events=np.array([r["events"] for r in recs]), seeds=seed0 + np.arange(n))
        ok = np.mean([r["formal_big"] == r["want"] for r in recs])
        print(f"train {family}: {n} scenes, {lens.sum()} entries, {sum(r['y'].sum() for r in recs):.0f} positive, "
              f"{np.mean([r['taken'] for r in recs]):.1f} units taken per scene, formal outcome as intended (ignoring fragments < 5 patches) {ok:.0%}, {time.time() - t0:.0f}s", flush=True)
    else:
        kind = sys.argv[5]
        recs = [probe_scene(rig, family, seed0 + i, kind) for i in range(n)]
        np.save(f"{OUT}/probe_{family}_{kind}.npy", np.array(recs, dtype=object), allow_pickle=True)
        print(f"probe {family} {kind}: {n} scenes, {np.mean([sum(len(o['X']) for o in r['objects']) for r in recs]):.0f} candidates each, "
              f"{np.mean([len(r['objects']) for r in recs]):.2f} formal objects, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
