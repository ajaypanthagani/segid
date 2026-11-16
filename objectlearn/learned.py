"""Part II: the learned layer.  Instance-based, online, no training phase.

Memory holds samples (phi, y, w).  Similarity is an additive kernel over the feature
groups of section 9.  A small set of kernel "experts" (bandwidth x group weights) is
mixed by exponential weights on their running log-loss, which is how the group weights
and bandwidth adapt online.  Nothing here is trained by gradient descent.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import beta as beta_dist

from .descriptor import DIMS, SLICES, D

GROUPS = ("side", "seam", "app")


@dataclass
class Config:
    s: float = 0.05
    eps: float = 0.065 
    k: int = 3
    delta: float = 0.12           # carried displacement, delta >> eps
    w_max: float = 0.7            # weight ceiling for carried (y = 0) evidence
    w_min: float = 0.25           # below this, a carried pair stays unlabelled
    event_budget: float = 3.0     # max total weight per outcome class per event (see report)
    max_rows_per_event: int = 64
    min_event_pairs: int = 0      # an event with fewer outcomes than this is discarded (noise, not an intervention)
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.1            # credibility 1 - gamma for the bounds
    theta_d: float = 0.65
    theta_c: float = 0.35
    s_min: float = 1.0
    pi_support0: float = 1.0
    eta: float = 10.0             # exponential-weights learning rate
    share: float = 0.02           # fixed-share mixing so no expert dies
    std_floor: float = 0.15
    cs: tuple = (0.04, 0.08)      # relative kernel widths
    k_nn: int = 96                # neighbourhood holding about two events of evidence
    t_abs: float = 3.0            # out-of-distribution fall-off of support
    omegas: dict = field(default_factory=lambda: {
        "uniform": (1 / 3, 1 / 3, 1 / 3), "side": (.6, .2, .2), "seam": (.2, .6, .2),
        "app": (.2, .2, .6), "seam_heavy": (.1, .8, .1), "seam_only": (0.0, 1.0, 0.0)})


def make_experts(cfg):
    return [(c, name, w) for c in cfg.cs for name, w in cfg.omegas.items()]


class SeamMemory:
    def __init__(self, cfg):
        self.cfg = cfg
        self.X = np.zeros((0, D), np.float32)
        self.y = np.zeros(0, np.float32)
        self.w = np.zeros(0, np.float32)
        self.n = 0
        self.mean = np.zeros(D)
        self.m2 = np.zeros(D)

    def __len__(self):
        return len(self.y)

    def add(self, phi, y, w):
        start = len(self.y)
        self.X = np.concatenate([self.X, phi.astype(np.float32)])
        self.y = np.concatenate([self.y, np.asarray(y, np.float32)])
        self.w = np.concatenate([self.w, np.asarray(w, np.float32)])
        for row in phi:                                   # Welford running statistics
            self.n += 1
            d = row - self.mean
            self.mean += d / self.n
            self.m2 += d * (row - self.mean)
        return np.arange(start, len(self.y))

    def std(self):
        if self.n < 2:
            return np.ones(D)
        return np.maximum(np.sqrt(self.m2 / self.n), self.cfg.std_floor)

    def query(self, phiA, phiB, experts, chunk=256):
        """Effective counts (n1, n0) per expert.

        For each expert the combined squared distance Dc = sum_g omega_g * d2_g / dim_g is
        taken (minimum over the two pair orderings).  The kernel is flat out to the k_nn-th
        nearest stored sample and decays beyond it, on a scale proportional to that radius, so
        the neighbourhood adapts to local density whatever the dimensionality.  Support also
        falls off for queries far from everything stored (rho), which gives the abstention.
        """
        c0 = self.cfg
        Q, E = len(phiA), len(experts)
        out = np.zeros((Q, E, 2), np.float32)
        M = len(self)
        if M == 0 or Q == 0:
            return out
        sd = self.std()
        Xs = ((self.X - self.mean) / sd).astype(np.float32)
        xn = {g: (Xs[:, SLICES[g]] ** 2).sum(1) for g in GROUPS}
        live = self.w > 0
        wy = self.w * self.y
        wn = self.w * (1 - self.y)
        k = int(min(c0.k_nn, max(M - 1, 1)))
        for q0 in range(0, Q, chunk):
            dn = []
            for phi in (phiA, phiB):
                Qs = ((phi[q0:q0 + chunk] - self.mean) / sd).astype(np.float32)
                d = {}
                for g in GROUPS:
                    sl = SLICES[g]
                    qn = (Qs[:, sl] ** 2).sum(1)
                    d[g] = np.maximum(qn[:, None] + xn[g][None, :] - 2 * Qs[:, sl] @ Xs[:, sl].T, 0) / DIMS[g]
                dn.append(d)
            for e, (c, _, w) in enumerate(experts):
                Dc = np.minimum(*[sum(w[i] * d[g] for i, g in enumerate(GROUPS) if w[i] > 0) for d in dn])
                Dc[:, ~live] = np.inf
                dmin = Dc.min(1)
                tau = np.partition(Dc, k, axis=1)[:, k] if k < M else Dc.max(1)
                tau = np.where(np.isfinite(tau), tau, dmin)
                K = np.exp(-np.maximum(Dc - tau[:, None], 0) / (c * np.maximum(tau, 1e-3)[:, None]))
                rho = np.exp(-dmin / c0.t_abs)
                out[q0:q0 + chunk, e, 0] = rho * (K @ wy)
                out[q0:q0 + chunk, e, 1] = rho * (K @ wn)
        return out


class Estimator:
    """Maps per-expert counts to (p_hat, p_low, p_high, support) and a statement."""

    def __init__(self, cfg, experts):
        self.cfg, self.experts = cfg, experts
        self.W = np.full(len(experts), 1.0 / len(experts))

    def mix(self, counts):
        n = np.tensordot(counts, self.W, axes=(1, 0))              # (Q, 2)
        a, b = self.cfg.alpha + n[:, 0], self.cfg.beta + n[:, 1]
        s = n[:, 0] + n[:, 1]
        return dict(p=a / (a + b), lo=beta_dist.ppf(self.cfg.gamma, a, b),
                    hi=beta_dist.ppf(1 - self.cfg.gamma, a, b), s=s)

    def statement(self, est):
        c = self.cfg
        ok = est["s"] >= c.s_min
        st = np.zeros(len(est["p"]), np.int8)                      # 0 unknown
        st[ok & (est["lo"] >= c.theta_d)] = 1                      # distinct
        st[ok & (est["hi"] <= c.theta_c) & (st == 0)] = 2          # coherent
        return st

    def expert_p(self, counts):
        a = self.cfg.alpha + counts[..., 0]
        b = self.cfg.beta + counts[..., 1]
        return a / (a + b)                                         # (Q, E)

    def update(self, counts, y, w):
        """Exponential weights on the per-expert weighted log-loss of an event."""
        if len(y) == 0:
            return
        p = np.clip(self.expert_p(counts), 1e-3, 1 - 1e-3)
        y = np.asarray(y)[:, None]
        loss = -(w[:, None] * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum(0) / max(w.sum(), 1e-9)
        W = self.W * np.exp(-self.cfg.eta * (loss - loss.min()))
        W /= W.sum()
        self.W = (1 - self.cfg.share) * W + self.cfg.share / len(W)

    def group_weights(self):
        g = np.zeros(3)
        for We, (_, _, w) in zip(self.W, self.experts):
            g += We * np.array(w)
        return dict(zip(GROUPS, g))
