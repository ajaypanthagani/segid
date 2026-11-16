"""Part I of the paper: the formal layer.

This module implements sections 1-4 of the definition exactly.  It consumes only
sensed positions and visibility, and produces A_t, N_t, I_t and O_t.  Nothing
learned ever writes into it; the only hook is the preference score `pi_fn`
(section 1, "Preference"), which is used only when interfaces are created.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


class IdTracker:
    """Object ids over time (section 4).

    An object that does not divide keeps its id, even as it gains patches.  When an object divides it ceases to exist:
    every part of it is a new object with a new id and a reference to the object it came from.  (Objects never merge in
    the formal layer; in the predicted view a merge also gives the result a new id.)  Ids are never reused.
    """

    def __init__(self, n, first=0):
        self.prev = -np.ones(n, np.int64)
        self.next = first
        self.parent = {}                 # part id -> id of the object it divided from
        self.divisions = []              # [(old id, [new ids])] from the latest call

    def assign(self, comp):
        comp = np.asarray(comp)
        out = -np.ones(len(comp), np.int64)
        groups = {int(c): np.flatnonzero(comp == c) for c in np.unique(comp[comp >= 0])}
        held, spread = {}, {}
        for c, idx in groups.items():
            held[c] = {int(p) for p in np.unique(self.prev[idx]) if p >= 0}
            for p in held[c]:
                spread[p] = spread.get(p, 0) + 1
        children = {}
        for c, idx in sorted(groups.items(), key=lambda t: int(t[1][0])):
            ps = held[c]
            if len(ps) == 1 and spread[next(iter(ps))] == 1:
                out[idx] = next(iter(ps))                     # undivided: same object
                continue
            nid, self.next = self.next, self.next + 1
            out[idx] = nid
            if len(ps) == 1:                                  # one of several parts of a divided object
                p = next(iter(ps))
                self.parent[nid] = p
                children.setdefault(p, []).append(nid)
        self.divisions = list(children.items())
        self.prev = out
        return out


@dataclass
class StepInfo:
    t: int
    adj: np.ndarray            # A_t  (n x n bool, symmetric)
    nmask: np.ndarray          # N_t  (n x n bool, symmetric)
    cand_new: np.ndarray       # (m, 2) pairs with tau == t and Cand  (i < j)
    admitted_new: np.ndarray   # (m2, 2) of those admitted into I_t
    opened_now: np.ndarray     # (q, 2) candidate pairs first observed apart at t (i < j)
    visible: np.ndarray        # V_t (bool n)
    divisions: list = field(default_factory=list)   # [(parent_id, [piece ids])]


class FormalLayer:
    def __init__(self, n_patches, eps, k):
        n = n_patches
        self.n, self.eps, self.k = n, float(eps), int(k)
        self.t = 0
        self.known = np.zeros(n, bool)
        self.first_obs = -np.ones(n, np.int32)
        self.apart_run = np.zeros((n, n), np.int16)
        self.tau = -np.ones((n, n), np.int32)          # first determination
        self.cand = np.zeros((n, n), bool)             # Cand(a, b)
        self.opened = np.zeros((n, n), bool)           # candidate pair observed apart (Open)
        self.I = np.zeros((n, n), bool)                # admitted interfaces
        self.labels = -np.ones(n, np.int64)            # object id in O_t, -1 if unknown patch
        self._ids = IdTracker(n)
        self.parent = self._ids.parent                 # part id -> id of the object it divided from

    # ------------------------------------------------------------------ step
    def step(self, positions, visible, pi_fn=None):
        n = self.n
        self.t += 1
        t = self.t
        vis = np.asarray(visible, bool)
        newly = vis & ~self.known
        self.known |= vis
        self.first_obs[newly] = t

        idx = np.flatnonzero(vis)
        adj = np.zeros((n, n), bool)
        far = np.zeros((n, n), bool)
        if idx.size:
            p = np.asarray(positions, float)[idx]
            d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
            off = ~np.eye(idx.size, dtype=bool)
            ix = np.ix_(idx, idx)
            adj[ix] = (d <= self.eps) & off
            far[ix] = (d > self.eps) & off
        # Apart requires k consecutive observations; anything else resets the run.
        self.apart_run = np.where(far, self.apart_run + 1, 0).astype(np.int16)
        nmask = self.apart_run >= self.k

        # first determination
        newdet = (self.tau < 0) & (adj | nmask)
        self.tau[newdet] = t
        cand_new_mat = newdet & adj
        self.cand |= cand_new_mat
        iu = np.triu(np.ones((n, n), bool), 1)
        ci, cj = np.nonzero(cand_new_mat & iu)

        # Open: interfaces now observed apart are removed, permanently.
        newly_opened = self.cand & nmask & ~self.opened
        self.opened |= self.cand & nmask
        self.I &= ~nmask
        oi, oj = np.nonzero(newly_opened & iu)

        # Admit candidates in decreasing pi.
        prev = self.labels.copy()
        comp_of, comp_label, dsu = self._components_state(prev)
        if ci.size:
            pi = np.zeros(ci.size) if pi_fn is None else np.asarray(pi_fn(ci, cj), float)
            order = np.argsort(-pi, kind="stable")
        else:
            order = np.array([], int)
        admitted = []
        for o in order:
            i, j = int(ci[o]), int(cj[o])
            ri, rj = self._find(dsu, comp_of[i]), self._find(dsu, comp_of[j])
            if ri != rj:
                li, lj = comp_label[ri], comp_label[rj]
                if li != -1 and lj != -1 and li != lj:
                    continue                      # would join two objects of O_{t-1}
                dsu[rj] = ri
                comp_label[ri] = li if li != -1 else lj
            self.I[i, j] = self.I[j, i] = True
            admitted.append((i, j))

        divisions = self._relabel(prev)
        return StepInfo(
            t=t, adj=adj, nmask=nmask,
            cand_new=np.stack([ci, cj], 1) if ci.size else np.zeros((0, 2), int),
            admitted_new=np.array(admitted, int).reshape(-1, 2),
            opened_now=np.stack([oi, oj], 1) if oi.size else np.zeros((0, 2), int),
            visible=vis, divisions=divisions)

    # --------------------------------------------------------------- helpers
    def _components_state(self, prev):
        ncomp, comp = connected_components(csr_matrix(self.I), directed=False)
        label = -np.ones(ncomp, np.int64)
        kn = np.flatnonzero(self.known & (prev >= 0))
        label[comp[kn]] = prev[kn]
        return comp, label, list(range(ncomp))

    @staticmethod
    def _find(dsu, x):
        while dsu[x] != x:
            dsu[x] = dsu[dsu[x]]
            x = dsu[x]
        return x

    def _relabel(self, prev):
        """O_t = connected components of (P_t, I_t).  A divided object is replaced by new objects, one per part."""
        _, comp = connected_components(csr_matrix(self.I), directed=False)
        self.labels = self._ids.assign(np.where(self.known, comp, -1))
        return self._ids.divisions

    # ------------------------------------------------------------------ views
    def objects(self):
        out = {}
        for lid in np.unique(self.labels[self.labels >= 0]):
            out[int(lid)] = np.flatnonzero(self.labels == lid)
        return out

    def n_objects(self):
        return len(np.unique(self.labels[self.labels >= 0]))
