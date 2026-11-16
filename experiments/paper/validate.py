"""Choose the memory's settings on validation scenes (seeds 60000+, never used for testing).

Grid: weight of carried-whole entries w0, neighbourhood size k, statement threshold theta_+ (theta_- = 1 - theta_+).
Distances do not depend on any of these, and with every carried-whole entry at the same weight w0 the evidence against is
w0 times a fixed sum, so each candidate's neighbourhood is computed once per k and every setting is read from it.  The reading
is exactly LatentMemory.query (checked against it below).  The best setting by exact-partition accuracy goes to out/chosen.json.

    PYTHONPATH=. .venv/bin/python experiments/paper/validate.py
"""
import itertools
import json

import numpy as np
from scipy.stats import beta as beta_dist

from experiments.paper import evaluate as E
from objectlearn.latent import LatentMemory

import os as _os
EXTEND = _os.environ.get("EXTEND") == "1"          # a second, finer grid past the edges of the first
KS = (3, 4, 6, 8) if EXTEND else (6, 12, 24, 48, 96)
W0 = (0.005, 0.01, 0.02, 0.05) if EXTEND else (0.02, 0.05, 0.1, 0.2, 0.35, 0.7, 1.0)
THETAS = (0.5, 0.525, 0.55, 0.6) if EXTEND else (0.55, 0.6, 0.65, 0.7, 0.75)
REL = _os.environ.get("RELEVANCE", "1") == "1"        # per-trait relevance weighting on (the design) or off (plain distance)
TAG = ("_ext" if EXTEND else "") + ("" if REL else "_norel")


class Grid:
    kinds = None

    def __init__(self, X, y, w):
        self.mem = LatentMemory(relevance=REL)
        self.mem.add(X, y, w)                                  # weights only matter through w > 0 for the relevance
        self.ypos = (y > 0.5).astype(np.float64)
        self.wpos = np.where(y > 0.5, w, 0).astype(np.float64)
        self.cache = {}
        self.cfg = None

    def precompute(self, X):
        mem = self.mem
        rel = mem.relevance(0)
        D = mem._dist(X, rel).astype(np.float64)
        n_live = len(mem)
        dmin = D.min(1)
        rho = np.exp(-dmin / mem.t_abs)
        kk = sorted({int(min(k, max(n_live - 1, 1))) for k in KS})
        P = np.partition(D, kk, axis=1)
        out = {}
        for k in KS:
            ke = int(min(k, max(n_live - 1, 1)))
            tau = P[:, ke]
            K = np.exp(-np.maximum(D - tau[:, None], 0) / (mem.c * np.maximum(tau, 1e-3)[:, None]))
            out[k] = (rho * (K @ self.wpos), rho * (K @ (1 - self.ypos)))
        self.cache[id(X)] = out

    def __call__(self, X):
        k, w0, th = self.cfg
        A, B = self.cache[id(X)][k]
        n1, n0 = A, w0 * B
        al, be = 1.0 + n1, 1.0 + n0
        p = al / (al + be)
        lo, hi = beta_dist.ppf(0.1, al, be), beta_dist.ppf(0.9, al, be)
        st = np.zeros(len(p), np.int8)
        ok = (n1 + n0) >= 1.0
        st[ok & (lo >= th)] = 1
        st[ok & (hi <= 1 - th) & (st == 0)] = 2
        return dict(p=p, statement=st)


if __name__ == "__main__":
    X, y, w = E.entries({f: np.arange(E.n_train(f)) for f in E.FAMILIES}, carried=None)
    g = Grid(X, y, w)
    import os
    import pickle
    probes = {f: E.probes(f, "val") for f in E.FAMILIES}
    path = f"{E.OUT}/val_neighbourhoods{TAG}.pkl"
    saved = pickle.load(open(path, "rb")) if os.path.exists(path) else {}
    for f in E.FAMILIES:
        for i, r in enumerate(probes[f]):
            for j, o in enumerate(r["objects"]):
                if len(o["X"]):
                    key = (f, i, j)
                    if key not in saved:
                        g.precompute(o["X"])
                        saved[key] = g.cache[id(o["X"])]
                    g.cache[id(o["X"])] = saved[key]
        pickle.dump(saved, open(path, "wb"))
        print("precomputed", f, flush=True)
    # the grid reading must equal LatentMemory.query for the same setting
    r0 = next(o for o in probes["bookshelf"][0]["objects"] if len(o["X"]))
    for k, w0, th in ((KS[-1], 0.7, 0.65), (KS[0], W0[0], THETAS[0])):
        g.cfg = (k, w0, th)
        mem = LatentMemory(k_nn=k, theta_hi=th, theta_lo=1 - th, relevance=REL)
        mem.add(X, y, np.where(y > 0.5, w, w0))
        q = mem.query(r0["X"]); qg = g(r0["X"])
        assert np.allclose(q["p"], qg["p"], atol=1e-5) and (q["statement"] == qg["statement"]).all(), "grid != LatentMemory.query"
    print("grid reading equals LatentMemory.query", flush=True)
    out = {}
    for k, w0, th in itertools.product(KS, W0, THETAS):
        g.cfg = (k, w0, th)
        res = {f: [E.score_probe(g, r) for r in probes[f]] for f in E.FAMILIES}
        s = E.summary(res)
        s["groups"] = E.by_group(res)
        out[f"rel={int(REL)} k={k} w0={w0} theta={th}"] = dict(relevance=REL, k_nn=k, w_carried=w0, theta=th, **{a: b for a, b in s.items()})
        print(f"k={k:3d} w0={w0:<5} theta={th}: accuracy {s['accuracy']:.3f}  pq {s['pq']:.3f}  "
              + "  ".join(f"{a} {b['accuracy']:.2f}" for a, b in s["groups"].items()), flush=True)
    json.dump(out, open(f"{E.OUT}/res_validation{TAG}.json", "w"), indent=1, default=float)
    import glob                                         # choose over every grid run so far
    for fpath in glob.glob(f"{E.OUT}/res_validation*.json"):
        for key, v in json.load(open(fpath)).items():
            v.setdefault("relevance", True)
            out.setdefault(f"rel={int(v['relevance'])} k={v['k_nn']} w0={v['w_carried']} theta={v['theta']}", v)
    # the top of the surface is flat: among settings within one validation scene of the best accuracy, take the highest PQ
    n_val = next(iter(out.values()))["n"]
    top = max(v["accuracy"] for v in out.values())
    best = max((v for v in out.values() if v["accuracy"] >= top - 1.0 / n_val - 1e-9), key=lambda v: v["pq"])
    chosen = dict(w_carried=best["w_carried"], k_nn=best["k_nn"], theta=best["theta"], relevance=bool(best["relevance"]))
    json.dump(chosen, open(f"{E.OUT}/chosen.json", "w"), indent=1)
    print("chosen on validation:", chosen, "accuracy", round(best["accuracy"], 3), "pq", round(best["pq"], 3))
