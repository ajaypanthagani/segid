"""The full model: formal layer (Part I) + online interface learner (Part II).

Information flows formal -> learned as labels, and learned -> formal only through the
preference score pi (used when interfaces are created).  The learned layer never writes
into A, N, I or O.
"""
import os

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from . import descriptor as _geom
from . import descriptor_image as _img
from . import descriptor_patch as _patch
from .descriptor import D
from .formal import FormalLayer
from .learned import Config, Estimator, SeamMemory, make_experts


SPREAD = float(os.environ.get("OL_SPREAD", 0))       # 0: judge each pair alone; m>0: pool evidence along the seam, up to m pairs' worth
SPREAD_LEN = float(os.environ.get("OL_SPREAD_LEN", 8))   # how far along the seam, in patch spacings


class Agent:
    def __init__(self, cfg=None, seed=0, mode="geometry"):
        self.cfg = cfg or Config()
        self.mode = mode                      # 'geometry' (depth-like patches) or 'image' (tracked image points)
        self.rng = np.random.default_rng(seed)
        self.experts = make_experts(self.cfg)
        self.mem = SeamMemory(self.cfg)
        self.est = Estimator(self.cfg, self.experts)
        self.log = []                     # prequential log, one entry per learning event
        self.learn = True
        self.new_scene(1)

    def _desc(self):
        return {"image": _img, "patch": _patch}.get(self.mode, _geom).pair_descriptors

    # ------------------------------------------------------------------ scene
    def new_scene(self, n_patches):
        E = len(self.experts)
        self.formal = FormalLayer(n_patches, self.cfg.eps, self.cfg.k)
        self.index = {}
        self.ti = np.zeros(0, int); self.tj = np.zeros(0, int)
        self.phi = np.zeros((0, D), np.float32); self.phiB = np.zeros((0, D), np.float32)
        self.cnt = np.zeros((0, E, 2), np.float32)
        self.stat = np.zeros(0, np.int8); self.stat_now = np.zeros(0, np.int8); self.phat = np.zeros(0); self.sup = np.zeros(0)
        self.run = np.zeros(0, int); self.maxd = np.zeros(0)
        self.anca = np.zeros((0, 3)); self.ancb = np.zeros((0, 3))
        self.wr = np.zeros(0, np.int8); self.row = np.zeros(0, int)
        self.frame = None

    # ------------------------------------------------------------- observation
    def observe(self, frame):
        self.frame = frame
        info = self.formal.step(frame.pos, frame.vis,
                                pi_fn=lambda ci, cj: self._on_candidates(ci, cj, frame))
        self._update_carry(info, frame)
        return info

    def _on_candidates(self, ci, cj, frame):
        c = self.cfg
        desc = self._desc()
        phiA, phiB = desc(frame, np.stack([ci, cj], 1), c.s)
        cnt = self.mem.query(phiA, phiB, self.experts)
        est = self.est.mix(cnt)
        st = self.est.statement(est)
        lam = est["s"] / (est["s"] + c.pi_support0)
        if self.mode == "patch":
            cos = _patch.default_pi(frame, ci, cj)                             # colour continuity between patches
        elif self.mode == "image":
            cos = _img.default_pi(frame, ci, cj)                               # colour continuity
        else:
            cos = np.clip((frame.nrm[ci] * frame.nrm[cj]).sum(1), 0, 1)       # normal continuity
        pi = lam * (1 - est["p"]) + (1 - lam) * cos
        base = len(self.ti)
        for q, (a, b) in enumerate(zip(ci, cj)):
            self.index[(int(a), int(b))] = base + q
        m = len(ci)
        self.ti = np.concatenate([self.ti, ci]); self.tj = np.concatenate([self.tj, cj])
        self.phi = np.concatenate([self.phi, phiA]); self.phiB = np.concatenate([self.phiB, phiB])
        self.cnt = np.concatenate([self.cnt, cnt])
        self.stat_now = np.concatenate([self.stat_now, st])
        self.stat = np.concatenate([self.stat, st]); self.phat = np.concatenate([self.phat, est["p"]])
        self.sup = np.concatenate([self.sup, est["s"]])
        self.run = np.concatenate([self.run, np.zeros(m, int)]); self.maxd = np.concatenate([self.maxd, np.zeros(m)])
        self.anca = np.concatenate([self.anca, np.zeros((m, 3))]); self.ancb = np.concatenate([self.ancb, np.zeros((m, 3))])
        self.wr = np.concatenate([self.wr, np.zeros(m, np.int8)]); self.row = np.concatenate([self.row, -np.ones(m, int)])
        return pi

    def _update_carry(self, info, frame):
        if len(self.ti) == 0:
            return
        i, j, vis = self.ti, self.tj, frame.vis
        ok = info.adj[i, j] & vis[i] & vis[j] & ~self.formal.opened[i, j]
        self.run = np.where(ok, self.run + 1, 0)
        fresh = self.run <= 1
        self.anca[fresh] = frame.pos[i][fresh]
        self.ancb[fresh] = frame.pos[j][fresh]
        carried = np.minimum(np.linalg.norm(frame.pos[i] - self.anca, axis=1),
                             np.linalg.norm(frame.pos[j] - self.ancb, axis=1))
        self.maxd = np.where(self.run >= self.cfg.k, np.maximum(self.maxd, carried), self.maxd)

    def _redescribe(self, rows, frame):
        """Recompute phi for table rows from `frame`, where both patches are visible.  The descriptor is then what the seam
        looks like at that moment, so something drawn after the pair was first formed (a cut line) is seen."""
        rows = np.asarray(rows, int)
        if rows.size == 0 or frame is None:
            return
        a, b = self.ti[rows], self.tj[rows]
        ok = frame.vis[a] & frame.vis[b]
        if not ok.any():
            return
        desc = self._desc()
        A, B = desc(frame, np.stack([a[ok], b[ok]], 1), self.cfg.s)
        self.phi[rows[ok]], self.phiB[rows[ok]] = A, B

    # ------------------------------------------------------------------ learning
    def end_event(self, rest_frame=None):
        """Turn the outcomes of the event just observed into samples; update the experts.
        rest_frame: the last still frame before the motion; if given, samples describe the seams as they looked then."""
        c = self.cfg
        if not self.learn or len(self.ti) == 0:
            return None
        opened = self.formal.opened[self.ti, self.tj]
        w_held = c.w_max * (1 - np.exp(-self.maxd / c.delta))
        pos = np.flatnonzero(opened & (self.wr != 2))
        neg = np.flatnonzero((~opened) & (self.wr == 0) & (self.maxd >= c.delta) & (w_held >= c.w_min))
        if len(pos) + len(neg) == 0:
            return None
        if len(pos) + len(neg) < c.min_event_pairs:          # too little evidence to be an intervention: consume and ignore
            self.wr[pos] = 3
            self.wr[neg] = 3
            return None
        rng = self.rng
        sel_pos = rng.choice(pos, min(len(pos), c.max_rows_per_event), replace=False) if len(pos) else pos
        sel_neg = rng.choice(neg, min(len(neg), c.max_rows_per_event), replace=False) if len(neg) else neg
        w_pos = np.ones(len(sel_pos))
        w_neg = w_held[sel_neg]
        for w in (w_pos, w_neg):
            tot = w.sum()
            if tot > c.event_budget:
                w *= c.event_budget / tot
        idx = np.concatenate([sel_pos, sel_neg]).astype(int)
        y = np.concatenate([np.ones(len(sel_pos)), np.zeros(len(sel_neg))])
        w = np.concatenate([w_pos, w_neg])

        # prequential: score the predictions made at tau before learning from the outcomes
        mixp = self.est.mix(self.cnt[idx])["p"]
        pc = np.clip(mixp, 1e-3, 1 - 1e-3)
        ll = float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
        self.est.update(self.cnt[idx], y, w)

        # flip: a carried pair that has now opened replaces its weak y = 0 row
        for r in self.row[sel_pos[self.wr[sel_pos] == 1]]:
            self.mem.w[r] = 0.0
        if rest_frame is not None:
            self._redescribe(idx, rest_frame)
        rows = self.mem.add(self.phi[idx], y, w)
        self.row[idx] = rows
        self.wr[sel_pos] = 2
        self.wr[sel_neg] = 1
        self.wr[np.setdiff1d(pos, sel_pos)] = 2        # surplus outcomes of this event are not stored
        self.wr[np.setdiff1d(neg, sel_neg)] = 3
        entry = dict(n_open=int(len(pos)), n_held=int(len(neg)), logloss=ll, rows=len(self.mem),
                     group_w=self.est.group_weights())
        self.log.append(entry)
        return entry

    # ------------------------------------------------------------- persistence
    def save(self, path):
        m = self.mem
        np.savez_compressed(path, X=m.X, y=m.y, w=m.w, n=m.n, mean=m.mean, m2=m.m2, W=self.est.W)

    def load(self, path):
        z = np.load(path)
        m = self.mem
        m.X, m.y, m.w, m.n, m.mean, m.m2 = z["X"], z["y"], z["w"], int(z["n"]), z["mean"], z["m2"]
        self.est.W = z["W"]

    # ----------------------------------------------------------------- readouts
    def interfaces(self):
        """Admitted interfaces with the statement made about each when it was created."""
        ii, jj = np.nonzero(np.triu(self.formal.I, 1))
        rows = np.array([self.index[(int(a), int(b))] for a, b in zip(ii, jj)], int)
        return ii, jj, rows

    def refresh(self, frame=None):
        """Re-state every admitted interface from the memory as it is now.  `stat` keeps the statement made when the
        pair was created (that is what prequential scoring needs); `stat_now` is what the model would say today."""
        if len(self.mem) == 0 or len(self.ti) == 0:
            return
        ii, jj, rows = self.interfaces()
        if len(rows) == 0:
            return
        self._redescribe(rows, frame)
        cnt = self.mem.query(self.phi[rows], self.phiB[rows], self.experts)
        if SPREAD > 0 and frame is not None:
            cnt = self._along_seam(cnt, rows, frame)
        est = self.est.mix(cnt)
        self.stat_now[rows] = self.est.statement(est)

    def _along_seam(self, cnt, rows, frame):
        """Pool the evidence of interfaces that lie on the same seam (same direction, same line, close by): the average of their
        counts, scaled up to at most SPREAD pairs' worth.  A seam is then judged as a line, not one pair at a time."""
        a, b = self.ti[rows], self.tj[rows]
        ok = frame.vis[a] & frame.vis[b]
        pa, pb = frame.pos[a, :2], frame.pos[b, :2]
        mid, d = 0.5 * (pa + pb), pb - pa
        d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
        s = self.cfg.s
        dm = mid[None, :, :] - mid[:, None, :]                       # (i, j): from i to j
        across = np.abs((dm * d[:, None, :]).sum(2))                 # distance along the pair direction (should be ~0)
        along = np.abs(dm[..., 0] * d[:, None, 1] - dm[..., 1] * d[:, None, 0])
        par = np.abs(d @ d.T)
        M = (par > 0.9) & (across <= 0.8 * s) & (along <= SPREAD_LEN * s) & ok[:, None] & ok[None, :]
        np.fill_diagonal(M, True)
        m = M.sum(1).astype(np.float32)
        flat = cnt.reshape(len(cnt), -1)
        pooled = (M.astype(np.float32) @ flat) / m[:, None] * np.minimum(m, SPREAD)[:, None]
        return pooled.reshape(cnt.shape)

    def predict_partition(self, current=False):
        """Predicted objects: components of (P, I minus interfaces predicted distinct). Hypotheses only.
        current=False uses the statements made at creation time; current=True the ones from `refresh`."""
        n = self.formal.n
        ii, jj, rows = self.interfaces()
        st = self.stat_now if current else self.stat
        keep = st[rows] != 1 if len(rows) else np.zeros(0, bool)
        g = csr_matrix((np.ones(keep.sum()), (ii[keep], jj[keep])), shape=(n, n))
        _, lab = connected_components(g, directed=False)
        lab = lab.astype(np.int64)
        lab[~self.formal.known] = -1
        return lab

    def choose_grasp(self, frame, policy="random"):
        s = self.cfg.s
        cand = np.flatnonzero(frame.vis & (frame.pos[:, 2] > 1.0 * s))
        if len(cand) == 0:
            return None
        if policy == "random" or len(self.ti) == 0:
            return int(self.rng.choice(cand))
        u = np.full(self.formal.n, 0.5)
        inI = self.formal.I[self.ti, self.tj]
        unc = 1 - np.abs(2 * self.phat - 1)
        cnt = np.zeros(self.formal.n); acc = np.zeros(self.formal.n)
        for arr in (self.ti, self.tj):
            np.add.at(acc, arr[inI], unc[inI]); np.add.at(cnt, arr[inI], 1)
        u = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.5)
        p = np.exp(4 * u[cand]); p /= p.sum()
        return int(self.rng.choice(cand, p=p))
