"""Candidate parts of any shape, and the physical traits of a part.

The straight-cut candidates of traits.py can only describe a part that lies on one side of a line (a cap, a lid).  A book standing
among other books, one cup among several on a table, a drawer in a cabinet, an apple in a bowl: each of these is a region bounded
on several sides by the rest of the object.  Here a candidate part is any set of patches of one object, from two generators:

  cuts     the smaller side of a straight cut across the object, at 4 directions and every patch spacing (as in traits.py)
  regions  connected regions of the object whose neighbouring patches differ by less than a threshold in the sensed signal, at
           several thresholds relative to the object's own typical neighbour difference; holes are filled (a label on a book,
           a knob on a drawer front belong to the region that surrounds them)

Every candidate, whatever produced it, is described by the same physical traits (TRAIT_NAMES): ratios and relations of the part
to the rest of its object -- size, shape, how and where it touches the rest, and how the sensed signal changes across that contact.
No absolute colour, position or size enters, and nothing names a kind of object.  The one directional trait is `above` (is the part
on top of the rest?), because gravity is a physical fact of the scene; it is the only trait that changes when a picture is turned.

All functions take an object's patches: pos (n, 2) positions, sig (n, c) the signal each patch sensed, s the patch spacing.
"""
import cv2
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .traits import DIRS

NOISE = 3.0                       # signal units below which a change counts as nothing
THRESHOLDS = (1.5, 3.0, 6.0, 12.0)
MAX_REGIONS = 40                  # largest regions kept per threshold

TRAIT_NAMES = (
    ["share", "elong_part", "elong_object", "fill_part", "solid_part", "solid_object", "extent_part"]          # size and shape
    + ["contact_share", "contact_len", "contact_sectors", "free_sectors", "contact_straight",                # how it touches
       "contact_vs_part", "contact_vs_rest", "rest_pieces"]
    + ["edge_med", "edge_lo", "edge_hi", "edge_strong", "inner_part", "inner_ring", "step_inside", "mean_step"]   # signal
    + ["above", "central", "at_end"])                                                                       # where
T = len(TRAIT_NAMES)


class ObjectGeom:
    """Everything about one object that does not depend on the candidate."""

    def __init__(self, pos, sig, s):
        self.pos = np.asarray(pos, float)
        self.sig = np.asarray(sig, float).reshape(len(self.pos), -1)
        self.s = float(s)
        n = self.n = len(self.pos)
        g = np.round((self.pos - self.pos.min(0)) / self.s).astype(int)      # grid cell of each patch
        self.g = g
        self.shape = (g[:, 1].max() + 3, g[:, 0].max() + 3)
        self.cell = -np.ones(self.shape, int)
        self.cell[g[:, 1] + 1, g[:, 0] + 1] = np.arange(n)
        tree = cKDTree(self.pos)
        pr = tree.query_pairs(1.05 * self.s, output_type="ndarray")           # 4-neighbours: the contacts
        self.i4, self.j4 = (pr[:, 0], pr[:, 1]) if len(pr) else (np.zeros(0, int), np.zeros(0, int))
        diff = np.linalg.norm(self.sig[self.i4] - self.sig[self.j4], axis=1) if len(pr) else np.zeros(0)
        self.med = float(np.median(diff)) if len(diff) else 0.0
        self.d4 = diff / (self.med + NOISE)                                   # neighbour change, in units of the object's own
        free = np.zeros(n, int)                                               # 4-neighbour cells that are not part of the object
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            free += self.cell[g[:, 1] + 1 + dy, g[:, 0] + 1 + dx] < 0
        self.free = free
        self.c0 = self.pos.mean(0)
        ev, vec = np.linalg.eigh(np.cov((self.pos - self.c0).T) + 1e-9 * np.eye(2)) if n >= 3 else (np.ones(2), np.eye(2))
        self.axis = vec[:, 1]
        u = (self.pos - self.c0) @ self.axis
        self.half = max(float(np.abs(u).max()), self.s)
        self.size = np.sqrt(n) * self.s
        self.elong = _elong(self.pos, self.s)
        self.solid = _solidity(g, np.ones(n, bool))
        self.extent = _extent(self.pos) + self.s

    def components(self, keep_pairs, mask=None):
        """Connected components over 4-neighbour contacts that are kept (restricted to `mask` if given)."""
        n = self.n
        k = keep_pairs if mask is None else keep_pairs & mask[self.i4] & mask[self.j4]
        A = coo_matrix((np.ones(int(k.sum())), (self.i4[k], self.j4[k])), shape=(n, n))
        return connected_components(A, directed=False)[1]


def _elong(p, s):
    if len(p) < 3:
        return 0.0
    ev = np.linalg.eigvalsh(np.cov((p - p.mean(0)).T))
    f = (0.25 * s) ** 2
    return float(np.log((max(ev[1], 0) + f) / (max(ev[0], 0) + f)))


def _extent(p):
    if len(p) < 2:
        return 0.0
    c = p - p.mean(0)
    ev, vec = np.linalg.eigh(np.cov(c.T) + 1e-9 * np.eye(2))
    u = c @ vec[:, 1]
    return float(u.max() - u.min())


def _solidity(g, m):
    """Share of the lattice cells inside the convex hull of the part that the part occupies."""
    q = g[m]
    if len(q) < 3:
        return 1.0
    hull = cv2.convexHull(q.astype(np.int32).reshape(-1, 1, 2))
    img = np.zeros((q[:, 1].max() + 2, q[:, 0].max() + 2), np.uint8)
    cv2.fillConvexPoly(img, hull, 1)
    return float(min(1.0, len(q) / max(img.sum(), 1)))


def _fill(g, m):
    """Share of the part's minimum-area bounding rectangle (in cells) that it occupies."""
    q = g[m]
    if len(q) < 3:
        return 1.0
    (cx, cy), (w, h), _ = cv2.minAreaRect(q.astype(np.float32))
    return float(min(1.0, len(q) / max((w + 1) * (h + 1), 1)))


# ------------------------------------------------------------------------------------------------------- candidates
def cut_masks(geo, min_part=3):
    pos, s, n = geo.pos, geo.s, geo.n
    out = []
    for th in DIRS:
        u = pos @ np.array([np.cos(th), np.sin(th)])
        b = np.floor((u - u.min()) / s + 1e-6).astype(int)
        cnt = np.bincount(b)
        cum = np.cumsum(cnt)
        for j in range(1, len(cnt)):
            if cnt[j - 1] == 0 or cnt[j] == 0:
                continue
            lo = cum[j - 1]
            if min(lo, n - lo) < min_part:
                continue
            out.append((b < j) if lo <= n - lo else (b >= j))
    return out


def region_masks(geo, min_part=3, max_share=0.8):
    out = []
    for th in THRESHOLDS:
        lab = geo.components(geo.d4 <= th)
        sizes = np.bincount(lab)
        order = np.argsort(-sizes)
        taken = 0
        for r in order:
            if sizes[r] < min_part or taken >= MAX_REGIONS:
                break
            if sizes[r] > max_share * geo.n:
                continue
            m = lab == r
            img = np.zeros(geo.shape, bool)                       # fill holes: what the region surrounds belongs to it
            img[geo.g[m, 1] + 1, geo.g[m, 0] + 1] = True
            filled = ndimage.binary_fill_holes(img)
            if filled.sum() > img.sum():
                m = filled[geo.g[:, 1] + 1, geo.g[:, 0] + 1]
                if m.sum() > max_share * geo.n:
                    continue
            out.append(m)
            taken += 1
    return out


def candidate_masks(geo, kinds=("cut", "region")):
    """All candidate parts of the object, without duplicates.  Returns (masks (m, n) bool, kind (m,) str)."""
    seen, masks, kind = set(), [], []
    for k in kinds:
        for m in (cut_masks(geo) if k == "cut" else region_masks(geo)):
            key = np.packbits(m).tobytes()
            if key in seen:
                continue
            seen.add(key)
            masks.append(m)
            kind.append(k)
    if not masks:
        return np.zeros((0, geo.n), bool), np.zeros(0, "<U6")
    return np.array(masks), np.array(kind)


# ----------------------------------------------------------------------------------------------------------- traits
def part_traits(geo, m):
    """The physical traits of part m (bool over the object's patches) relative to the rest of its object."""
    s, n = geo.s, geo.n
    m = np.asarray(m, bool)
    nP = int(m.sum())
    P, Rm = geo.pos[m], ~m
    i4, j4, d4 = geo.i4, geo.j4, geo.d4
    a, b = m[i4], m[j4]
    cross = a != b                                    # contacts between the part and the rest
    inP, inR = a & b, ~a & ~b
    # patches of the part that touch the rest / touch nothing (the object's own outline)
    touchR = np.zeros(n, bool)
    touchR[i4[cross & a]] = True
    touchR[j4[cross & b]] = True
    touchR &= m
    rim = touchR | (m & (geo.free > 0))
    n_contact, n_rim = int(touchR.sum()), int(rim.sum())
    cP = P.mean(0)
    # in which of 8 directions around the part's centre the rest touches it, and in which the part is free
    def sectors(sel):
        if not sel.any():
            return 0.0
        v = geo.pos[sel] - cP
        ang = np.floor(((np.arctan2(v[:, 1], v[:, 0]) + np.pi) / (2 * np.pi)) * 8).astype(int) % 8
        return len(np.unique(ang)) / 8.0
    cs = sectors(touchR)
    fs = sectors(m & (geo.free > 0))
    # the contact as a curve: how straight it is and how long, against the part's and the rest's extent along it
    if n_contact >= 3:
        Q = geo.pos[touchR]
        ev, vec = np.linalg.eigh(np.cov((Q - Q.mean(0)).T) + 1e-9 * np.eye(2))
        straight = float(np.sqrt(max(ev[0], 0)) / (np.sqrt(max(ev[1], 0)) + 0.5 * s))
        dvec = vec[:, 1]
        clen = float(np.ptp(Q @ dvec)) + s
    else:
        straight, clen = 0.0, float(n_contact * s)
        dvec = np.array([1.0, 0.0])
    plen = float(np.ptp(P @ dvec)) + s
    rpos = geo.pos[Rm]
    near_rest = Rm & (np.linalg.norm(geo.pos - cP, axis=1) <= 0.5 * plen + 3 * s)
    rlen = float(np.ptp(geo.pos[near_rest] @ dvec)) + s if near_rest.sum() >= 2 else s
    # does taking the part away leave the rest in one piece?
    rest_lab = geo.components(np.ones(len(i4), bool), mask=Rm)
    rest_pieces = len(np.unique(rest_lab[Rm])) if Rm.any() else 0
    # how the signal changes across the contact, inside the part, and just beyond it in the rest
    dc = d4[cross]
    ring = np.zeros(n, bool)
    ring[i4[cross & ~a]] = True
    ring[j4[cross & ~b]] = True
    ring_pairs = inR & (ring[i4] | ring[j4])
    step_in = inP & (touchR[i4] | touchR[j4])
    lg = lambda v: float(np.log1p(v))
    ringmean = geo.sig[ring].mean(0) if ring.any() else geo.sig[m].mean(0)
    mean_step = np.linalg.norm(geo.sig[m].mean(0) - ringmean) / (geo.med + NOISE)
    # where the part is: above the rest (gravity), near the object's centre, at an end of its long axis
    above = float((rpos[:, 1].mean() - cP[1]) / geo.size) if len(rpos) else 0.0
    central = float(np.linalg.norm(cP - geo.c0) / geo.size)
    at_end = float(min(abs((cP - geo.c0) @ geo.axis) / geo.half, 1.0))
    return [np.log(nP / n), _elong(P, s), geo.elong, _fill(geo.g, m), _solidity(geo.g, m), geo.solid, np.log((_extent(P) + s) / geo.extent),
            n_contact / max(n_rim, 1), np.log((n_contact + 1) / np.sqrt(nP)), cs, fs, straight,
            np.log(clen / plen), np.log(clen / rlen), min(rest_pieces, 3) - 1,
            lg(np.median(dc)) if len(dc) else 0.0, lg(np.percentile(dc, 10)) if len(dc) else 0.0, lg(np.percentile(dc, 90)) if len(dc) else 0.0,
            float((dc > 2.0).mean()) if len(dc) else 0.0,
            lg(np.median(d4[inP])) if inP.any() else 0.0, lg(np.median(d4[ring_pairs])) if ring_pairs.any() else 0.0,
            lg(np.median(d4[step_in])) if step_in.any() else 0.0, lg(mean_step),
            above, central, at_end]


def describe(geo, masks):
    """Traits of many candidate parts: (m, T)."""
    if len(masks) == 0:
        return np.zeros((0, T), np.float32)
    return np.array([part_traits(geo, mk) for mk in masks], np.float32)


def candidates(pos, sig, s, kinds=("cut", "region")):
    """All candidate parts of one object with their traits: X (m, T), masks (m, n), kind (m,)."""
    geo = ObjectGeom(pos, sig, s)
    masks, kind = candidate_masks(geo, kinds)
    return describe(geo, masks), masks, kind
