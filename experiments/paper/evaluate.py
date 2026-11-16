"""Scoring a memory (or a baseline) on cached probe scenes.

A *method* maps candidate traits X (m, T) to a dict with p (m,) and statement (m,) in {0 unknown, 1 separates, 2 stays joined}.
Prediction on a probe scene is exactly the session's: for every formal object, select_cuts over its candidates (strongest
'separates' first, none overlapping a kept one by IoU >= 0.5), then every contact that crosses the boundary of a kept part is cut,
and the predicted objects are the connected components that remain.

The prediction is a partition of the scene's visible patches; so is the ground truth (the finest partition the units induce).
Segments of fewer than 3 patches in the ground truth are void (too small for the sensing to resolve) and are left out of scoring.
"""
import os

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from objectlearn.latent import LatentMemory
from objectlearn.regions import TRAIT_NAMES
from objectlearn.session import select_cuts

OUT = "experiments/paper/out"
SINGLE = ["bottle", "jar", "pen", "tube", "spray", "cup", "blocks", "lipstick"]
ARRANGED = ["table", "bookshelf", "cabinet", "fruit_bowl", "pencil_cup", "book_stack"]
WHOLE = ["hammer", "screwdriver", "vase", "goblet", "chair", "mug", "desk_lamp"]
FAMILIES = SINGLE + ARRANGED + WHOLE
GROUP = {**{f: "single" for f in SINGLE}, **{f: "arranged" for f in ARRANGED}, **{f: "whole" for f in WHOLE}}
GROUPS = dict(shape=list(range(0, 7)), contact=list(range(7, 15)), signal=list(range(15, 23)), where=list(range(23, 26)))
assert TRAIT_NAMES[7] == "contact_share" and TRAIT_NAMES[15] == "edge_med" and TRAIT_NAMES[23] == "above" and len(TRAIT_NAMES) == 26
VOID = 3

# The memory's settings: the app's defaults unless out/chosen.json (picked on validation scenes, validate.py) says otherwise.
SETTINGS = dict(w_carried=None, k_nn=24, theta=0.65, relevance=True)
DEFAULTS = dict(w_carried=None, k_nn=24, theta=0.65, relevance=True)
if os.path.exists(f"{OUT}/chosen.json"):
    import json as _json
    SETTINGS.update(_json.load(open(f"{OUT}/chosen.json")))

# ------------------------------------------------------------------------------------------------------------ data
_TRAIN, _PROBE = {}, {}


def train(family):
    if family not in _TRAIN:
        z = np.load(f"{OUT}/train_{family}.npz")
        _TRAIN[family] = {k: z[k] for k in z.files}
    return _TRAIN[family]


def probes(family, kind="none"):
    key = (family, kind)
    if key not in _PROBE:
        recs = list(np.load(f"{OUT}/probe_{family}_{kind}.npy", allow_pickle=True))
        for r in recs:
            _prepare(r)
        _PROBE[key] = recs
    return _PROBE[key]


def entries(spec, carried="settings"):
    """spec: {family: list of scene indices}.  Returns X, y, w of the union of those scenes' entries.  The weight of carried-whole
    entries is SETTINGS['w_carried'] (None: as recorded, 0.7), or `carried` if given."""
    Xs, ys, ws = [], [], []
    for f, objs in spec.items():
        t = train(f)
        m = np.isin(t["obj"], np.asarray(objs))
        Xs.append(t["X"][m]); ys.append(t["y"][m]); ws.append(t["w"][m])
    X, y, w = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)
    wc = SETTINGS["w_carried"] if carried == "settings" else carried
    if wc is not None:
        w = np.where(y > 0.5, w, wc).astype(np.float32)
    return X, y, w


def n_train(family):
    return len(train(family)["formal"])


def _prepare(r):
    """Ground truth per probe: void patches, the unit segments, and each candidate's best overlap with a unit segment."""
    gt = r["gt"].copy()
    sizes = np.bincount(gt)
    gt[sizes[gt] < VOID] = 0                                           # void
    r["gt_v"] = gt
    segs = [g for g in np.unique(gt) if g > 0]
    big = max(segs, key=lambda g: sizes[g]) if segs else None
    r["units"] = [g for g in segs if g != big]                         # everything except the largest segment (what is left)
    for o in r["objects"]:
        loc = o["loc"]
        if len(o["masks"]) == 0 or not r["units"]:
            o["best_iou"] = np.zeros(len(o["masks"]))
            continue
        lab = gt[loc]
        ious = []
        for g in r["units"]:
            gm = gt == g
            inter = (o["masks"] & (lab == g)[None, :]).sum(1)
            union = o["masks"].sum(1) + gm.sum() - inter
            ious.append(inter / np.maximum(union, 1))
        o["best_iou"] = np.max(ious, 0)


# --------------------------------------------------------------------------------------------------------- methods
class Latent:
    """The latent memory as used in the app.  cols restricts the traits; kinds restricts the candidate generators (ablations)."""

    def __init__(self, X, y, w, cols=None, abstain=True, kinds=None, **kw):
        self.cols = np.arange(X.shape[1]) if cols is None else np.asarray(cols)
        kw.setdefault("k_nn", SETTINGS["k_nn"])
        kw.setdefault("theta_hi", SETTINGS["theta"])
        kw.setdefault("theta_lo", 1 - SETTINGS["theta"])
        kw.setdefault("relevance", SETTINGS.get("relevance", True))
        self.mem = LatentMemory(**kw)
        self.mem.add(X[:, self.cols], y, w)
        self.abstain, self.kinds = abstain, kinds

    def __call__(self, X):
        q = self.mem.query(X[:, self.cols])
        if not self.abstain:                       # point estimate only: no credible bounds, no support requirement
            q["statement"] = np.where(q["p"] > 0.5, 1, 2).astype(np.int8)
        return q


class Logistic:
    """L2-regularised logistic regression on the same standardised traits; entry weights w; classes balanced."""

    def __init__(self, X, y, w, cols=None, lam=1e-2, kinds=None):
        self.cols = np.arange(X.shape[1]) if cols is None else np.asarray(cols)
        self.kinds = kinds
        X = X[:, self.cols].astype(float)
        self.mu, self.sd = X.mean(0), np.maximum(X.std(0), 0.05)
        Z = np.c_[(X - self.mu) / self.sd, np.ones(len(X))]
        pos = y > 0.5
        cw = np.where(pos, 0.5 / max(w[pos].sum(), 1e-9), 0.5 / max(w[~pos].sum(), 1e-9)) * w

        def f(b):
            s = Z @ b
            l = np.logaddexp(0, -s) * pos + np.logaddexp(0, s) * ~pos
            g = Z.T @ (cw * (1 / (1 + np.exp(-s)) - pos))
            return (cw * l).sum() + 0.5 * lam * (b[:-1] ** 2).sum(), g + np.r_[lam * b[:-1], 0]
        self.b = minimize(f, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B").x

    def __call__(self, X):
        Z = np.c_[(X[:, self.cols] - self.mu) / self.sd, np.ones(len(X))]
        p = 1 / (1 + np.exp(-(Z @ self.b)))
        return dict(p=p, statement=np.where(p > 0.5, 1, 2).astype(np.int8))


class EdgeRule:
    """Classical boundary-contrast segmentation as a decision rule: a candidate is a separate thing when the sensed signal changes
    strongly all along its contact with the rest.  Score = edge_med + edge_strong; the threshold is fitted on the training entries
    to maximise F1 (weighted by w)."""

    def __init__(self, X, y, w, kinds=None):
        self.kinds = kinds
        self.c = [TRAIT_NAMES.index("edge_med"), TRAIT_NAMES.index("edge_strong")]
        s = X[:, self.c].sum(1)
        best = (-1.0, 0.0)
        for th in np.quantile(s, np.linspace(0.3, 0.999, 300)):
            pred = s > th
            tp = (w * (pred & (y > 0.5))).sum(); fp = (w * (pred & (y <= 0.5))).sum(); fn = (w * (~pred & (y > 0.5))).sum()
            f1 = 2 * tp / max(2 * tp + fp + fn, 1e-9)
            if f1 > best[0]:
                best = (f1, th)
        self.theta = best[1]

    def __call__(self, X):
        s = X[:, self.c].sum(1)
        return dict(p=1 / (1 + np.exp(-(s - self.theta) * 4)), statement=np.where(s > self.theta, 1, 2).astype(np.int8))


class Never:
    """Predict that nothing separates (the observational definition on its own, before anything has moved)."""
    kinds = None

    def __call__(self, X):
        return dict(p=np.zeros(len(X)), statement=np.full(len(X), 2, np.int8))


# --------------------------------------------------------------------------------------------------------- scoring
def predict(method, r):
    """Predicted partition of the probe's visible patches, and the parts that were kept."""
    pred = np.zeros(len(r["pos"]), int)
    nxt = 0
    for o in r["objects"]:
        loc = o["loc"]
        if len(o["masks"]) == 0:
            pred[loc] = nxt; nxt += 1
            continue
        X, masks = o["X"], o["masks"]
        if method.kinds is not None:
            ok = np.isin(o["kind"], method.kinds)
            X, masks = X[ok], masks[ok]
        if len(X) == 0:
            pred[loc] = nxt; nxt += 1
            continue
        q = method(X)
        sel = select_cuts(q, masks)
        keep = np.ones(len(o["i4"]), bool)
        for k in sel:
            keep &= masks[k][o["i4"]] == masks[k][o["j4"]]
        n = len(loc)
        A = coo_matrix((np.ones(int(keep.sum())), (o["i4"][keep], o["j4"][keep])), shape=(n, n))
        lab = connected_components(A, directed=False)[1]
        pred[loc] = nxt + lab
        nxt += lab.max() + 1
    return pred


def match(pred, gt):
    """Panoptic matching (IoU > 0.5 is unique) over non-void patches."""
    ok = gt > 0
    p, g = pred[ok], gt[ok]
    ps, pc = np.unique(p, return_counts=True)
    gs, gc = np.unique(g, return_counts=True)
    pairs, inter = np.unique(np.stack([p, g], 1), axis=0, return_counts=True)
    psize, gsize = dict(zip(ps, pc)), dict(zip(gs, gc))
    tp, iou_sum, matched_p, matched_g = 0, 0.0, set(), set()
    for (a, b), c in zip(pairs, inter):
        u = psize[a] + gsize[b] - c
        if c / u > 0.5:
            tp += 1; iou_sum += c / u; matched_p.add(a); matched_g.add(b)
    fp = sum(1 for a in ps if a not in matched_p and psize[a] >= VOID)
    fn = sum(1 for b in gs if b not in matched_g)
    return tp, fp, fn, iou_sum, matched_g


def score_probe(method, r):
    pred = predict(method, r)
    tp, fp, fn, iou_sum, mg = match(pred, r["gt_v"])
    out = dict(group=r["group"], tp=tp, fp=fp, fn=fn, iou=iou_sum, correct=(fp == 0 and fn == 0),
               pq=iou_sum / max(tp + 0.5 * fp + 0.5 * fn, 1e-9), n_units=len(r["units"]),
               units_found=sum(1 for g in r["units"] if g in mg))
    # threshold-free: rank of candidates that are a true unit (IoU >= 0.75) against all others, pooled over the scene's objects
    ps, ys, sts = [], [], []
    for o in r["objects"]:
        if len(o["X"]) == 0:
            continue
        X, keep = o["X"], np.ones(len(o["X"]), bool)
        if method.kinds is not None:
            keep = np.isin(o["kind"], method.kinds)
        if not keep.any():
            continue
        q = method(X if method.kinds is None else X[keep])
        ps.append(q["p"]); ys.append(o["best_iou"][keep] >= 0.75); sts.append(q["statement"])
    if ps:
        p, y, st = np.concatenate(ps), np.concatenate(ys), np.concatenate(sts)
        if y.any() and (~y).any():
            a, b = p[y][:, None], p[~y][None, :]
            out["auc"] = float(((a > b) + 0.5 * (a == b)).mean())
        else:
            out["auc"] = np.nan
        out["unknown_true"] = float((st[y] == 0).mean()) if y.any() else np.nan
        out["unknown_all"] = float((st == 0).mean())
        out["p"], out["y"] = p, y
    else:
        out.update(auc=np.nan, unknown_true=np.nan, unknown_all=np.nan, p=np.zeros(0), y=np.zeros(0, bool))
    return out


def score(method, families=FAMILIES, kind="none"):
    return {f: [score_probe(method, r) for r in probes(f, kind)] for f in families if os.path.exists(f"{OUT}/probe_{f}_{kind}.npy")}


def summary(res, groups=("single", "arranged", "whole")):
    allr = [x for f in res for x in res[f] if x["group"] in groups]
    if not allr:
        return {}
    m = lambda xs, k: float(np.nanmean([x[k] for x in xs])) if xs else np.nan
    units = sum(x["n_units"] for x in allr)
    out = dict(accuracy=m(allr, "correct"), pq=m(allr, "pq"),
               unit_recall=float(sum(x["units_found"] for x in allr) / units) if units else np.nan,
               false_per_scene=m(allr, "fp"), split_whole=float(np.mean([x["fp"] > 0 for x in allr if x["group"] == "whole"])) if any(x["group"] == "whole" for x in allr) else np.nan,
               auc=m(allr, "auc"), unknown_true=m(allr, "unknown_true"), n=len(allr))
    return out


def by_group(res):
    return {g: summary(res, (g,)) for g in ("single", "arranged", "whole")}


def per_family(res, key="correct"):
    return {f: float(np.nanmean([x[key] for x in res[f]])) for f in res}


def bootstrap(res, key="accuracy", B=1000, seed=0, groups=("single", "arranged", "whole")):
    """95% interval, resampling scenes within each family."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        rr = {f: [res[f][i] for i in rng.integers(0, len(res[f]), len(res[f]))] for f in res}
        vals.append(summary(rr, groups).get(key, np.nan))
    return [float(v) for v in np.nanpercentile(vals, [2.5, 97.5])]


def strip(res):
    """Results without the per-candidate arrays (for saving)."""
    return {f: [{k: v for k, v in x.items() if k not in ("p", "y")} for x in res[f]] for f in res}
