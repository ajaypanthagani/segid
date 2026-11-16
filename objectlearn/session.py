"""A session: a stream of frames in, objects out, with the model learning online.

The session does not care where frames come from.  It only needs, per frame, the position and visibility of
every patch (patches keep their identity over time) and the image the descriptors are computed from.  An intervention
is a burst of motion followed by about a second of stillness; the caller supplies timestamps, so it works the same
for a live stream and for a replay.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .agent import Agent
from .formal import IdTracker
from .latent import LatentMemory
from .learned import Config
from .sim import Frame
from .regions import ObjectGeom, candidate_masks, describe, part_traits, TRAIT_NAMES as REGION_TRAITS
from .traits import to_lab


# The latent memory's settings, chosen on validation scenes (experiments/paper/validate.py): neighbourhood size, statement
# threshold, and the weight of a 'carried whole' entry (being carried along with its support is weak evidence against coming apart).
LATENT_SETTINGS = dict(k_nn=4, theta_hi=0.5, theta_lo=0.5, relevance=False)
HELD_WEIGHT = 0.005


def select_cuts(q, masks, iou=0.5):
    """The cuts the latent space says separate, strongest first, dropping any that overlaps an already chosen one (IoU >= iou)."""
    keep = []
    for k in np.argsort(-q["p"], kind="stable"):
        if q["statement"][k] != 1:
            break
        if all((masks[k] & masks[j]).sum() / max((masks[k] | masks[j]).sum(), 1) < iou for j in keep):
            keep.append(int(k))
    return keep


@dataclass
class Result:
    xy: np.ndarray            # (m, 2) visible patches, pixels
    idx: np.ndarray           # (m,) patch indices
    formal: np.ndarray        # (m,) formal object id per visible patch
    predicted: np.ndarray     # (m,) predicted object id per visible patch
    events: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    formal_objects: list = field(default_factory=list)
    predicted_objects: list = field(default_factory=list)
    seams: list = field(default_factory=list)      # [x1, y1, x2, y2] for interfaces the model currently calls distinct


class Session:
    def __init__(self, patches=2000, s=10.0, eps=13.0, k=3, delta=40.0, min_event_pairs=8, event_budget=16.0, k_nn=24, memory=None,
                 min_moving_patches=6, still_s=1.0, min_active_s=0.0, refresh_every=3, mode="image", latent=False, latent_memory=None):
        self.mode = mode
        # latent: predicted objects come from the latent space of traits (latent.py) instead of the pair learner.  The formal layer
        # is unchanged; the pair learner is switched off and only the default colour-continuity preference admits interfaces.
        self.latent_on, self.latent_path = bool(latent), latent_memory
        self.latent = LatentMemory(**LATENT_SETTINGS) if self.latent_on else None
        self._carried = {}                                    # memory row -> (patch ids of a carried part, row) in this scene
        self.held_per_piece = 40
        self.patches = patches
        self.cfg = Config(s=s, eps=eps, k=k, delta=delta, min_event_pairs=min_event_pairs, event_budget=event_budget, k_nn=k_nn)
        self.min_moving_patches, self.still_s, self.min_active_s, self.refresh_every = min_moving_patches, still_s, min_active_s, refresh_every
        self.memory_path = memory
        self.agent = Agent(self.cfg, seed=0, mode=self.mode)
        self.agent.learn = not self.latent_on
        self.n_events = 0
        if memory and not self.latent_on:
            try:
                self.agent.load(memory)
            except OSError:
                pass
        if self.latent_on and latent_memory:
            try:
                self.latent.load(latent_memory)
            except OSError:
                pass
        self._new_scene()

    # ------------------------------------------------------------------ control
    def _new_scene(self):
        self.agent.new_scene(self.patches)
        self.prev_pos = np.zeros((self.patches, 2), np.float32)
        self.prev_vis = np.zeros(self.patches, bool)
        self.active_since = self.quiet_since = None
        self.frames, self._rest, self._settle = 0, None, 0
        self._raw_f = self._rest_f = None
        self._cutmasks, self._lat_due = [], False
        self._carried = {}                                    # patch ids are reused in a new scene, so held entries are keyed per scene
        self._fids, self._pids = IdTracker(self.patches, 1), IdTracker(self.patches, 1)

    def reset_scene(self):
        """New scene, same memory: what has been learned is kept."""
        self._new_scene()

    def forget(self):
        """Empty the memory as well."""
        self.agent = Agent(self.cfg, seed=0, mode=self.mode)
        self.agent.learn = not self.latent_on
        if self.latent_on:
            self.latent = LatentMemory(**LATENT_SETTINGS)
            self._carried = {}
        self.n_events = 0
        self._new_scene()

    def save(self):
        if self.latent_on:
            if self.latent_path and len(self.latent):
                self.latent.save(self.latent_path)
        elif self.memory_path:
            self.agent.save(self.memory_path)

    # ------------------------------------------------------------------- ingest
    def ingest(self, frame, t):
        events = []
        self.agent.observe(frame)
        self._last = frame
        self.frames += 1
        vis = frame.vis
        idx = np.flatnonzero(vis)
        xy = frame.pos[idx, :2]

        changed = bool((vis != self.prev_vis).any())
        both = vis & self.prev_vis
        moving_n = int((np.linalg.norm(frame.pos[both, :2] - self.prev_pos[both], axis=1) > 1.0).sum()) if both.any() else 0
        changed = changed or moving_n > 0
        self.prev_pos[vis] = frame.pos[vis, :2]
        self.prev_vis = vis.copy()
        if moving_n >= self.min_moving_patches:
            if self.active_since is None:
                self.active_since = t
            self.quiet_since = None
        elif self.active_since is not None:
            if self.quiet_since is None:
                self.quiet_since = t
            if t - self.quiet_since > self.still_s and self.quiet_since - self.active_since > self.min_active_s:
                e = self._learn_latent(frame) if self.latent_on else self.agent.end_event(self._rest)
                self.active_since = self.quiet_since = None
                if e:
                    self.n_events += 1
                    if self.latent_on:
                        self._lat_due = True
                        events.append(f"learned from that move: {e['n_open']} parts came away, {e['n_held']} carried whole "
                                      f"(latent space {e['rows']} entries)")
                    else:
                        self.agent.refresh(frame)
                        events.append(f"learned from that move: {e['n_open']} openings, {e['n_held']} carried pairs "
                                      f"(memory {e['rows']} samples)")
                    self.save()

        if moving_n < self.min_moving_patches and self.active_since is None:
            self._rest = Frame(pos=frame.pos.copy(), nrm=frame.nrm, col=frame.col, vis=frame.vis.copy(),
                               img=None if frame.img is None else frame.img.copy(),
                               patch_rgb=None if frame.patch_rgb is None else frame.patch_rgb.copy())
            self._rest_f = None if self._raw_f is None else self._raw_f.copy()
        if changed:
            self._settle = 15                       # keep predictions current while the picture changes, and once after
        due = (self._settle > 0 and self.frames % self.refresh_every == 0) or self._settle == 1
        self._settle = max(self._settle - 1, 0)
        if self.latent_on:
            self._lat_due = self._lat_due or due
        elif len(self.agent.mem) and due:
            self.agent.refresh(frame)
        lf, lp = self._object_labels(frame)
        fobj, pobj = self._boxes(xy, lf[idx], self._fids.parent), self._boxes(xy, lp[idx], self._pids.parent)
        seams = self._seams(frame)
        stats = dict(frames=self.frames, patches_visible=int(len(idx)), patches_total=int(self.agent.formal.known.sum()),
                     formal_objects=len(fobj), samples=int(len(self.latent) if self.latent_on else len(self.agent.mem)),
                     events=self.n_events, moving=moving_n, pairs_opened=int(np.triu(self.agent.formal.opened, 1).sum()),
                     group_weights={k: round(float(v), 2) for k, v in self.agent.est.group_weights().items()})
        return Result(xy=xy, idx=idx, formal=lf[idx], predicted=lp[idx], events=events,
                      stats=stats, formal_objects=fobj, predicted_objects=pobj, seams=seams)

    # ------------------------------------------------------------------ explanation
    def explain(self, x, y):
        """Every number behind the model's statement about the interface nearest to (x, y), in the order it is computed."""
        from .learned import GROUPS, make_experts
        from .descriptor import DIMS, SLICES
        ag, fr = self.agent, getattr(self, "_last", None)
        if fr is None:
            return None
        if self.latent_on:
            return self._explain_latent(fr, x, y)
        ii, jj, rows = ag.interfaces()
        if len(rows) == 0:
            return dict(found=False, why="no interfaces yet: add something to the canvas")
        ok = fr.vis[ii] & fr.vis[jj]
        if not ok.any():
            return dict(found=False, why="no visible interfaces")
        mid = 0.5 * (fr.pos[ii, :2] + fr.pos[jj, :2])
        d = np.where(ok, np.hypot(mid[:, 0] - x, mid[:, 1] - y), np.inf)
        q = int(np.argmin(d))
        if d[q] > 25:
            return dict(found=False, why="click closer to the picture")
        r = rows[q]
        cfg, mem, est = ag.cfg, ag.mem, ag.est
        ag._redescribe(rows[q:q + 1], fr)
        pa, pb = ag.phi[r:r + 1], ag.phiB[r:r + 1]
        counts = mem.query(pa, pb, ag.experts)                  # (1, E, 2)
        mixed = est.mix(counts)
        out = dict(found=True, x=float(mid[q, 0]), y=float(mid[q, 1]), memory=len(mem),
                   formal_open=bool(ag.formal.opened[ii[q], jj[q]]),
                   cfg=dict(alpha=cfg.alpha, beta=cfg.beta, gamma=cfg.gamma, theta_d=cfg.theta_d, theta_c=cfg.theta_c,
                            k_nn=cfg.k_nn, t_abs=cfg.t_abs, eta=cfg.eta, s_min=cfg.s_min),
                   dims=dict(DIMS), groups=list(GROUPS),
                   experts=[dict(c=float(c), name=nm, omega=[float(v) for v in w], W=float(est.W[e]),
                                 n1=float(counts[0, e, 0]), n0=float(counts[0, e, 1]),
                                 p=float((cfg.alpha + counts[0, e, 0]) / (cfg.alpha + cfg.beta + counts[0, e, 0] + counts[0, e, 1])))
                            for e, (c, nm, w) in enumerate(ag.experts)],
                   group_weights={k: float(v) for k, v in est.group_weights().items()},
                   mixed=dict(n1=float(np.tensordot(counts[0, :, 0], est.W, 1)), n0=float(np.tensordot(counts[0, :, 1], est.W, 1)),
                              p=float(mixed["p"][0]), lo=float(mixed["lo"][0]), hi=float(mixed["hi"][0]), s=float(mixed["s"][0])),
                   statement=["unknown", "distinct", "coherent"][int(est.statement(mixed)[0])], neighbours=[])
        if len(mem):
            e = int(np.argmax(est.W))
            c, nm, w = ag.experts[e]
            sd = mem.std()
            Xs = (mem.X - mem.mean) / sd
            dd = []
            for phi in (pa, pb):
                Qs = (phi - mem.mean) / sd
                dd.append({g: ((Qs[0, SLICES[g]] - Xs[:, SLICES[g]]) ** 2).sum(1) / DIMS[g] for g in GROUPS})
            per = [sum(w[i] * dv[g] for i, g in enumerate(GROUPS)) for dv in dd]
            use = np.argmin(np.stack(per), 0)
            Dc = np.where(mem.w > 0, np.minimum(*per), np.inf)
            k = int(min(cfg.k_nn, max(len(mem) - 1, 1)))
            tau = np.partition(Dc, k)[k] if k < len(mem) else Dc[np.isfinite(Dc)].max()
            K = np.exp(-np.maximum(Dc - tau, 0) / (c * max(tau, 1e-3)))
            rho = float(np.exp(-np.min(Dc) / cfg.t_abs))
            out["kernel"] = dict(expert=nm, c=float(c), tau=float(tau), rho=rho, dmin=float(np.min(Dc)))
            for j in np.argsort(-(K * mem.w))[:6]:
                g = {gn: float(dd[use[j]][gn][j]) for gn in GROUPS}
                out["neighbours"].append(dict(y=float(mem.y[j]), w=float(mem.w[j]), D=float(Dc[j]), K=float(K[j]), **g))
        return out

    # ------------------------------------------------------------------ latent space
    def _signal(self, frame_or_rest, idx):
        return to_lab(frame_or_rest.patch_rgb[idx])

    def _learn_latent(self, frame):
        """Turn what the definition decided in this event into entries of the latent space.

        An object that came apart: every piece except the largest (the rest) is a part that came away, and its traits are a
        'separates' = 1 entry -- the traits of exactly the patches that the definition put in that piece.  A piece that moved as a
        whole: each of its candidate parts was carried, a weak 'separates' = 0 entry; if that part later comes away in this scene
        (overlap >= 0.8), the weak entry is withdrawn."""
        rest, lab0, lab1 = self._rest, self._rest_f, self._raw_f
        if rest is None or lab0 is None or lab1 is None or rest.patch_rgb is None:
            return None
        c = self.cfg
        s = c.s
        rng = np.random.default_rng(self.n_events)
        ok = rest.vis & frame.vis & (lab0 >= 0) & (lab1 >= 0)
        disp = frame.pos[:, :2] - rest.pos[:, :2]
        rows, y, w, n_open, n_held = [], [], [], 0, 0
        came = []
        for obj in np.unique(lab0[ok]):
            idx = np.flatnonzero(ok & (lab0 == obj))
            if len(idx) < 12:
                continue
            groups = [idx[lab1[idx] == g] for g in np.unique(lab1[idx])]
            groups = [g for g in groups if len(g) >= 3]
            if len(groups) >= 2:                                         # the object came apart: positives
                geo = ObjectGeom(rest.pos[idx, :2], self._signal(rest, idx), s)
                big = int(np.argmax([len(g) for g in groups]))
                for gi, g in enumerate(groups):
                    if gi == big:
                        continue
                    rows.append(part_traits(geo, np.isin(idx, g))); y.append(1.0); w.append(1.0); n_open += 1
                    came.append(set(g.tolist()))
            for g in groups:                                              # pieces that moved as a whole: held entries
                if len(g) < 12:
                    continue
                d = disp[g]
                med = np.median(d, axis=0)
                if np.linalg.norm(med) < c.delta or np.percentile(np.linalg.norm(d - med, axis=1), 90) > 0.5 * c.delta:
                    continue
                geo = ObjectGeom(rest.pos[g, :2], self._signal(rest, g), s)
                masks, _ = candidate_masks(geo)
                if len(masks) == 0:
                    continue
                take = rng.choice(len(masks), min(len(masks), self.held_per_piece), replace=False)
                for k in take:
                    rows.append((part_traits(geo, masks[k]), g[masks[k]])); y.append(0.0); w.append(HELD_WEIGHT)
                n_held += 1
        if not y:
            return None
        base = len(self.latent)
        for part in came:                                                 # a part that has now come away replaces its weak 'held' entry
            for key, (pids, r) in list(self._carried.items()):
                inter = len(part & pids)
                if inter and inter / len(part | pids) >= 0.8:
                    self.latent.w[r] = 0.0
                    del self._carried[key]
        Xs = np.array([r[0] if isinstance(r, tuple) else r for r in rows], np.float32)
        self.latent.add(Xs, np.array(y), np.array(w))
        for j, r in enumerate(rows):
            if isinstance(r, tuple):
                self._carried[base + j] = (set(r[1].tolist()), base + j)
        return dict(n_open=n_open, n_held=n_held, rows=len(self.latent))

    def _latent_predict(self, frame, lf):
        """For every object the definition currently has, read every candidate part from the latent space and keep the ones it says
        separate (strongest first, none overlapping a kept one).  Hypotheses only: the formal layer is not touched."""
        self._cutmasks = []
        self._act = None
        self._act_id = getattr(self, "_act_id", 0) + 1
        if self.latent is None or len(self.latent) == 0 or frame.patch_rgb is None:
            return
        s = self.cfg.s
        vis = frame.vis & (lf >= 0)
        lit, called = [], 0
        for obj in np.unique(lf[vis]):
            idx = np.flatnonzero(vis & (lf == obj))
            if len(idx) < 12:
                continue
            geo = ObjectGeom(frame.pos[idx, :2], self._signal(frame, idx), s)
            masks, _ = candidate_masks(geo)
            if len(masks) == 0:
                continue
            X = describe(geo, masks)
            q = self.latent.query(X)
            keep = select_cuts(q, masks)
            for k in keep:
                m = np.zeros(len(lf), bool)
                m[idx[masks[k]]] = True
                self._cutmasks.append(m)
            called += len(keep)
            lit.append(X[list(dict.fromkeys(keep + [int(a) for a in np.argsort(-q["p"])[:3]]))])      # the parts it called, and the best few it considered
        if lit:
            order, st = self.latent.activation(np.concatenate(lit))
            self._act = dict(idx=order, strength=st, cuts=called, considered=int(sum(len(a) for a in lit)))

    def _explain_latent(self, fr, x, y):
        """The candidate part containing the clicked spot that the latent space is most sure about, with all the numbers behind it."""
        if len(self.latent) == 0:
            return dict(found=False, why="the latent space is empty: cut a part off something and drag it away, a few times")
        s = self.cfg.s
        lf = self._raw_f
        if lf is None:
            return dict(found=False, why="nothing on the canvas yet")
        vis = fr.vis & (lf >= 0)
        near = np.flatnonzero(vis)
        if len(near) == 0:
            return dict(found=False, why="nothing visible")
        k0 = near[np.argmin(np.hypot(fr.pos[near, 0] - x, fr.pos[near, 1] - y))]
        if np.hypot(fr.pos[k0, 0] - x, fr.pos[k0, 1] - y) > 2.5 * s:
            return dict(found=False, why="click on the picture")
        idx = np.flatnonzero(vis & (lf == lf[k0]))
        geo = ObjectGeom(fr.pos[idx, :2], self._signal(fr, idx), s)
        masks, kind = candidate_masks(geo)
        on = np.flatnonzero(masks[:, int(np.flatnonzero(idx == k0)[0])]) if len(masks) else []
        if len(on) == 0:
            return dict(found=False, why="no candidate part here")
        X = describe(geo, masks[on])
        q = self.latent.query(X)
        kk = int(np.argmax(q["p"]))
        k = int(on[kk])
        e = self.latent.explain(X[kk])
        q3 = self.latent.project(X[kk])[0].tolist()
        nn3 = self.latent.project(self.latent.X[[n["idx"] for n in e["neighbours"]]]).tolist()
        part = idx[masks[k]]
        mid = fr.pos[part, :2].mean(0)
        return dict(found=True, latent=True, x=float(mid[0]), y=float(mid[1]), memory=len(self.latent), q3=q3, nn3=nn3, part=int(masks[k].sum()), of=int(len(idx)),
                    kind=str(kind[k]), outline=[[float(a), float(b)] for a, b in fr.pos[part, :2]],
                    traits=[dict(name=n, value=float(v), rel=float(r)) for n, v, r in zip(REGION_TRAITS, X[kk], e["relevance"])], **{kk2: e[kk2] for kk2 in e if kk2 not in ("relevance", "statement")},
                    statement=["unknown", "separates", "stays joined"][e["statement"]])

    def latent_active(self):
        """The stored entries the latest reading leaned on, placed in the 3-D view (None when there is nothing to show)."""
        a = getattr(self, "_act", None)
        if not self.latent_on or a is None or len(a["idx"]) == 0:
            return None
        P = self.latent.project(self.latent.X[a["idx"]])
        y = self.latent.Y[a["idx"], 0]
        return dict(cuts=a["cuts"], considered=a["considered"], n_pos=int((y > 0.5).sum()), n_neg=int((y <= 0.5).sum()),
                    pts=[[round(float(v), 3) for v in p] + [int(yy > 0.5), round(float(self.latent.w[i]), 2), int(i), round(float(sv), 3)]
                         for p, yy, i, sv in zip(P, y, a["idx"], a["strength"])])

    def latent_entry(self, i):
        if not self.latent_on or not (0 <= int(i) < len(self.latent)):
            return None
        return self.latent.entry(i, names=REGION_TRAITS)

    def latent_cloud(self):
        """3-D view of the latent memory for the front end (None when not in latent mode)."""
        if not self.latent_on:
            return None
        return self.latent.cloud(names=REGION_TRAITS)

    def _latent_cut(self, ii, jj):
        cut = np.zeros(len(ii), bool)
        for m in self._cutmasks:
            cut |= m[ii] != m[jj]
        return cut

    # ------------------------------------------------------------------ readout
    def _seams(self, frame, limit=700):
        """Admitted interfaces the model currently predicts distinct (and that are both visible), as line segments."""
        ag = self.agent
        ii, jj, rows = ag.interfaces()
        if len(rows) == 0:
            return []
        keep = (self._latent_cut(ii, jj) if self.latent_on else ag.stat_now[rows] == 1) & frame.vis[ii] & frame.vis[jj]
        a, b = ii[keep][:limit], jj[keep][:limit]
        return [[int(frame.pos[i, 0]), int(frame.pos[i, 1]), int(frame.pos[j, 0]), int(frame.pos[j, 1])] for i, j in zip(a, b)]

    def _object_labels(self, frame):
        """An object id for every known patch (-1 for patches not yet seen).

        Only drawn things have patches, so there is no background to exclude.  Formal objects are the connected
        components of the admitted interfaces (Part I): things that touch are one object, and a piece becomes its own
        object when the interfaces joining it to the rest have been seen to open.  Predicted objects are the components
        after also cutting the interfaces the model currently calls distinct.
        """
        ag = self.agent
        f = ag.formal
        known = f.known

        def components(adj):
            _, lab = connected_components(csr_matrix(adj), directed=False)
            return np.where(known, lab, -1)
        lf = components(f.I)
        self._raw_f = lf.copy()
        if self.latent_on and self._lat_due:
            self._latent_predict(frame, lf)
            self._lat_due = False
        P = f.I.copy()
        ii, jj, rows = ag.interfaces()
        if len(rows):
            cut = self._latent_cut(ii, jj) if self.latent_on else ag.stat_now[rows] == 1
            P[ii[cut], jj[cut]] = False
            P[jj[cut], ii[cut]] = False
        return self._fids.assign(lf), self._pids.assign(components(P))

    @staticmethod
    def _boxes(xy, labels, parent, min_patches=1):
        out = []
        for i in np.unique(labels[labels >= 0]):
            m = labels == i
            if m.sum() < min_patches:
                continue
            q = xy[m]
            out.append(dict(id=int(i), n=int(m.sum()), background=False, parent=parent.get(int(i)),
                            box=[float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())]))
        return out
