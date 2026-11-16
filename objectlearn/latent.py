"""The latent space of observations.

Every entry is something that was observed: a point described by its physical traits (see traits.py) together with the
behaviours recorded for it.  A behaviour is just another coordinate of the entry: 1 (seen), 0 (seen not to happen), NaN
(not observed).  Here the one behaviour is "separates": the part came away from the rest as an object of its own, which the
formal layer (the definition) is what decides.  Nothing in the memory says what a cap, a handle or a lid is.

To say something about a new thing, the memory looks at where its traits put it, finds the nearest entries, and reads off their
recorded behaviour.  Distance is not set by hand: how much each trait counts is taken from how that trait has co-varied with the
behaviour across everything stored, and is recomputed from the stored entries.  Beyond the stored entries the memory abstains.
"""
import numpy as np
from scipy.stats import beta as beta_dist


class LatentMemory:
    def __init__(self, behaviours=("separates",), k_nn=24, c=0.08, t_abs=3.0, gamma=0.1, alpha=1.0, beta=1.0,
                 theta_hi=0.65, theta_lo=0.35, relevance=True, rel_floor=0.15):
        self.behaviours = tuple(behaviours)
        self.k_nn, self.c, self.t_abs = k_nn, c, t_abs
        self.gamma, self.alpha, self.beta = gamma, alpha, beta
        self.theta_hi, self.theta_lo = theta_hi, theta_lo
        self.use_relevance, self.rel_floor = relevance, rel_floor
        self.X = None
        self.Y = np.zeros((0, len(self.behaviours)), np.float32)
        self.w = np.zeros(0, np.float32)
        self._cache = {}

    def __len__(self):
        return len(self.w)

    # ----------------------------------------------------------------- storing
    def add(self, X, Y, w=None):
        """X (m, T) traits; Y (m, B) or (m,) behaviours with NaN where not observed; w (m,) weight of the evidence."""
        X = np.asarray(X, np.float32).reshape(len(X), -1)
        Y = np.asarray(Y, np.float32).reshape(len(X), -1)
        w = np.ones(len(X), np.float32) if w is None else np.asarray(w, np.float32)
        self.X = X if self.X is None else np.concatenate([self.X, X])
        self.Y = np.concatenate([self.Y, Y])
        self.w = np.concatenate([self.w, w])
        self._cache = {}

    # --------------------------------------------------------------- the space
    def _scale(self):
        if "scale" not in self._cache:
            self._cache["scale"] = (self.X.mean(0), np.maximum(self.X.std(0), 0.05))
        return self._cache["scale"]

    def relevance(self, behaviour=0):
        """How much each trait counts for this behaviour: a trait counts in proportion to how far apart the entries that did
        and did not show the behaviour sit along it (in units of that trait's spread), plus a small floor so no trait is ruled out."""
        key = ("rel", behaviour)
        if key in self._cache:
            return self._cache[key]
        mu, sd = self._scale()
        Z = (self.X - mu) / sd
        y = self.Y[:, behaviour]
        ok = ~np.isnan(y) & (self.w > 0)
        pos, neg = ok & (y > 0.5), ok & (y <= 0.5)
        if not self.use_relevance or pos.sum() < 3 or neg.sum() < 3:
            rel = np.ones(Z.shape[1])
        else:
            m1, m0 = Z[pos].mean(0), Z[neg].mean(0)
            v = 0.5 * (Z[pos].var(0) + Z[neg].var(0)) + 0.05
            d2 = (m1 - m0) ** 2 / v
            rel = self.rel_floor + d2 / (d2.mean() + 1e-9)
        rel = rel / rel.mean()
        self._cache[key] = rel
        return rel

    def _dist(self, Q, rel):
        mu, sd = self._scale()
        A = ((self.X - mu) / sd) * np.sqrt(rel)
        B = ((Q - mu) / sd) * np.sqrt(rel)
        d2 = (B ** 2).sum(1)[:, None] + (A ** 2).sum(1)[None, :] - 2 * B @ A.T
        return np.maximum(d2, 0) / len(rel)

    # ------------------------------------------------------------ reading it
    def query(self, Q, behaviour=0, chunk=512):
        """For each query (traits), what the nearest entries say about the behaviour.

        Returns dict with p (share that showed it), lo/hi (credible bounds), support (effective number of entries), nearest
        (distance to the closest entry) and statement: 1 yes, 2 no, 0 unknown."""
        Q = np.asarray(Q, np.float32).reshape(-1, self.X.shape[1] if self.X is not None else 1)
        n = len(Q)
        out = dict(p=np.full(n, 0.5), lo=np.zeros(n), hi=np.ones(n), support=np.zeros(n), nearest=np.full(n, np.inf), statement=np.zeros(n, np.int8))
        if self.X is None or n == 0:
            return out
        y = self.Y[:, behaviour]
        live = ~np.isnan(y) & (self.w > 0)
        if live.sum() == 0:
            return out
        rel = self.relevance(behaviour)
        yy = np.nan_to_num(y)
        wy, wn = self.w * yy * live, self.w * (1 - yy) * live
        k = int(min(self.k_nn, max(live.sum() - 1, 1)))
        for a in range(0, n, chunk):
            D = self._dist(Q[a:a + chunk], rel)
            D[:, ~live] = np.inf
            dmin = D.min(1)
            tau = np.partition(D, k, axis=1)[:, k] if k < live.sum() else np.where(np.isfinite(D), D, 0).max(1)
            tau = np.where(np.isfinite(tau), tau, dmin)
            K = np.exp(-np.maximum(D - tau[:, None], 0) / (self.c * np.maximum(tau, 1e-3)[:, None]))
            rho = np.exp(-dmin / self.t_abs)
            n1, n0 = rho * (K @ wy), rho * (K @ wn)
            al, be = self.alpha + n1, self.beta + n0
            sl = slice(a, a + chunk)
            out["p"][sl] = al / (al + be)
            out["lo"][sl] = beta_dist.ppf(self.gamma, al, be)
            out["hi"][sl] = beta_dist.ppf(1 - self.gamma, al, be)
            out["support"][sl] = n1 + n0
            out["nearest"][sl] = dmin
        st = np.zeros(n, np.int8)
        ok = out["support"] >= 1.0
        st[ok & (out["lo"] >= self.theta_hi)] = 1
        st[ok & (out["hi"] <= self.theta_lo) & (st == 0)] = 2
        out["statement"] = st
        return out

    def explain(self, q, behaviour=0, top=6):
        """Everything behind the answer for one query: relevance of each trait, the kernel, and the nearest entries."""
        q = np.asarray(q, np.float32).reshape(1, -1)
        res = self.query(q, behaviour)
        rel = self.relevance(behaviour)
        y = self.Y[:, behaviour]
        live = ~np.isnan(y) & (self.w > 0)
        D = self._dist(q, rel)[0]
        D[~live] = np.inf
        k = int(min(self.k_nn, max(live.sum() - 1, 1)))
        tau = float(np.partition(D, k)[k]) if k < live.sum() else float(D[np.isfinite(D)].max())
        K = np.exp(-np.maximum(D - tau, 0) / (self.c * max(tau, 1e-3)))
        rho = float(np.exp(-D.min() / self.t_abs))
        mu, sd = self._scale()
        near = np.argsort(-(K * self.w * live))[:top]
        n1 = float(rho * (K * self.w * np.nan_to_num(y) * live).sum())
        n0 = float(rho * (K * self.w * (1 - np.nan_to_num(y)) * live).sum())
        return dict(p=float(res["p"][0]), lo=float(res["lo"][0]), hi=float(res["hi"][0]), support=float(res["support"][0]),
                    statement=int(res["statement"][0]), relevance=rel, tau=tau, rho=rho, dmin=float(D.min()), n1=n1, n0=n0,
                    neighbours=[dict(idx=int(j), y=float(y[j]), w=float(self.w[j]), D=float(D[j]), K=float(K[j])) for j in near],
                    cfg=dict(alpha=self.alpha, beta=self.beta, gamma=self.gamma, theta_hi=self.theta_hi, theta_lo=self.theta_lo,
                             k_nn=self.k_nn, c=self.c, t_abs=self.t_abs))

    # ------------------------------------------------------------ looking at it
    def _pca(self, behaviour=0):
        """The three directions along which the stored entries differ most, in the same weighted space distance is measured in."""
        key = ("pca", behaviour)
        if key in self._cache:
            return self._cache[key]
        mu, sd = self._scale()
        rel = self.relevance(behaviour)
        Z = ((self.X - mu) / sd) * np.sqrt(rel)
        y = self.Y[:, behaviour]
        live = ~np.isnan(y) & (self.w > 0)
        zm = Z[live].mean(0)
        C = np.cov((Z[live] - zm).T)
        ev, vec = np.linalg.eigh(C)
        order = np.argsort(-ev)[:3]
        V, ev3 = vec[:, order], ev[order]
        pos, neg = live & (y > 0.5), live & (y <= 0.5)
        if pos.sum() and neg.sum():                      # orient each axis so "came away" lies on its positive side
            sgn = np.sign((Z[pos].mean(0) - Z[neg].mean(0)) @ V)
            V = V * np.where(sgn == 0, 1, sgn)
        P = (Z[live] - zm) @ V
        scale = max(float(np.percentile(np.linalg.norm(P, axis=1), 95)), 1e-6)
        out = dict(zm=zm, V=V, scale=scale, share=ev3 / max(ev.sum(), 1e-9), rel=rel, sqrt_rel=np.sqrt(rel))
        self._cache[key] = out
        return out

    def project(self, X, behaviour=0):
        """Where traits X sit in the 3-D view: (m, 3)."""
        a = self._pca(behaviour)
        mu, sd = self._scale()
        Z = ((np.asarray(X, np.float32).reshape(-1, self.X.shape[1]) - mu) / sd) * a["sqrt_rel"]
        return ((Z - a["zm"]) @ a["V"]) / a["scale"]

    def cloud(self, behaviour=0, max_each=2500, names=None, seed=0):
        """Entries for drawing: 3-D positions, whether the behaviour was seen, and the weight of the evidence."""
        if self.X is None or len(self) == 0:
            return dict(n=0, pts=[], axes=[])
        y = self.Y[:, behaviour]
        live = np.flatnonzero(~np.isnan(y) & (self.w > 0))
        rng = np.random.default_rng(seed)
        pos, neg = live[y[live] > 0.5], live[y[live] <= 0.5]
        pos = pos if len(pos) <= max_each else rng.choice(pos, max_each, replace=False)
        neg = neg if len(neg) <= max_each else rng.choice(neg, max_each, replace=False)
        pick = np.concatenate([neg, pos]).astype(int)
        P = self.project(self.X[pick], behaviour)
        a = self._pca(behaviour)
        axes = []
        for k in range(3):
            load = a["V"][:, k] * 1.0
            top = np.argsort(-np.abs(load))[:3]
            axes.append(dict(share=float(a["share"][k]), traits=[names[i] if names else int(i) for i in top]))
        mu, sd = self._scale()
        Zt = np.clip((self.X[pick] - mu) / sd, -4, 4)
        return dict(n=int(len(live)), shown=int(len(pick)), n_pos=int((y[live] > 0.5).sum()), n_neg=int((y[live] <= 0.5).sum()),
                    pts=[[round(float(v), 3) for v in p] + [int(y[i] > 0.5), round(float(self.w[i]), 2), int(i)] for p, i in zip(P, pick)], axes=axes,
                    names=list(names) if names else None, tz=np.round(Zt * 10).astype(int).tolist())

    def activation(self, Q, behaviour=0, top=250):
        """Which stored entries a set of queries leans on: for each entry, the strongest kernel x absence weight over the queries.
        Returns (indices, strength in 0..1, n1, n0) where n1/n0 are the evidence totals for the strongest query."""
        Q = np.asarray(Q, np.float32).reshape(-1, self.X.shape[1])
        y = self.Y[:, behaviour]
        live = ~np.isnan(y) & (self.w > 0)
        rel = self.relevance(behaviour)
        D = self._dist(Q, rel)
        D[:, ~live] = np.inf
        k = int(min(self.k_nn, max(live.sum() - 1, 1)))
        dmin = D.min(1)
        tau = np.partition(D, k, axis=1)[:, k] if k < live.sum() else np.where(np.isfinite(D), D, 0).max(1)
        tau = np.where(np.isfinite(tau), tau, dmin)
        K = np.exp(-np.maximum(D - tau[:, None], 0) / (self.c * np.maximum(tau, 1e-3)[:, None])) * np.exp(-dmin / self.t_abs)[:, None]
        K[:, ~live] = 0
        act = K.max(0)
        order = np.argsort(-act)[:top]
        order = order[act[order] > 0.02]
        return order, act[order]

    def entry(self, i, behaviour=0, top=8, names=None):
        """One stored entry: what it is, and which other entries sit nearest to it."""
        i = int(i)
        mu, sd = self._scale()
        z = (self.X[i] - mu) / sd
        rel = self.relevance(behaviour)
        D = self._dist(self.X[i:i + 1], rel)[0]
        D[i] = np.inf
        y = self.Y[:, behaviour]
        D[np.isnan(y) | (self.w <= 0)] = np.inf
        near = np.argsort(D)[:top]
        P = self.project(self.X[[i] + [int(j) for j in near]], behaviour)
        return dict(idx=i, y=None if np.isnan(y[i]) else float(y[i]), w=float(self.w[i]), xyz=P[0].tolist(),
                    traits=[dict(name=(names[t] if names else t), value=float(self.X[i, t]), z=float(z[t]), rel=float(rel[t])) for t in range(self.X.shape[1])],
                    near=[dict(idx=int(j), y=float(y[j]), w=float(self.w[j]), D=float(D[j]), xyz=P[1 + a].tolist()) for a, j in enumerate(near)])

    # ------------------------------------------------------------ persistence
    def save(self, path):
        np.savez_compressed(path, X=self.X if self.X is not None else np.zeros((0, 0)), Y=self.Y, w=self.w, behaviours=np.array(self.behaviours))

    def load(self, path):
        z = np.load(path)
        self.X = z["X"] if z["X"].size else None
        self.Y, self.w = z["Y"], z["w"]
        self.behaviours = tuple(str(b) for b in z["behaviours"])
        self._cache = {}
