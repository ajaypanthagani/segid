"""Physical traits of a candidate part.

A candidate is a way of dividing one object into a smaller part and the rest, by a straight cut across it (swept at four
directions, at every patch spacing).  Everything below is read from patch positions and from whatever signal the patches
carry, as ratios and relations: no absolute colour, position or size, and nothing that names a kind of object.  The same
number means the same thing for any object, so candidates from different objects can be compared.

Canonical order: the smaller side is always S ("the part") and the other is L ("the rest"), so a trait vector does not depend
on which way the cut was swept.

Traits (TRAIT_NAMES):
  profile      width of the object at 4 patch rows and 3 coarser steps outward from the cut, each side, relative to the width
               at the cut (a narrowing at the cut, a step, the end of the object)
  shift        how far the object's centre line moves at the first 3 rows outward, each side, relative to the cut width
  extent       how far each side extends beyond the cut, relative to the cut width, and the part's share of the whole
  material     size of the signal change at the cut and at 2 rows either side, against the usual change between rows;
               and how uniform each side is (signal-agnostic: only changes are used, never values)
  shape        cut width against the object's size and against its widest place; how near the cut is to an end; whether the cut
               runs across the object's long axis; how elongated the object and the part are
"""
import cv2
import numpy as np

DIRS = np.deg2rad([0.0, 45.0, 90.0, 135.0])
NFINE, NCOARSE, COARSE_W = 4, 3, 4
NOISE = 3.0                                            # signal units below which a change counts as nothing

TRAIT_NAMES = (
    [f"width_{side}{k}" for side in "SL" for k in range(1, NFINE + 1)]
    + [f"width_{side}c{k}" for side in "SL" for k in range(1, NCOARSE + 1)]
    + [f"shift_{side}{k}" for side in "SL" for k in range(1, 4)]
    + ["extent_S", "extent_L", "mass_S"]
    + ["step_cut", "step_S1", "step_S2", "step_L1", "step_L2", "uniform_S", "uniform_L"]
    + ["cut_vs_size", "cut_vs_widest", "near_end", "across_axis", "elong_object", "elong_part"])
T = len(TRAIT_NAMES)


def to_lab(bgr):
    """(n, 3) uint8 BGR patch colours -> (n, 3) float Lab.  Any other signal can be passed as `sig` directly."""
    return cv2.cvtColor(np.asarray(bgr, np.uint8).reshape(-1, 1, 3), cv2.COLOR_BGR2Lab).astype(np.float32).reshape(-1, 3)


def _elong(p, s):
    """log of the ratio of spreads along the long and short axes (a floor of a quarter patch keeps one-row parts finite)."""
    if len(p) < 3:
        return 0.0
    ev = np.linalg.eigvalsh(np.cov((p - p.mean(0)).T))
    f = (0.25 * s) ** 2
    return float(np.log((max(ev[1], 0) + f) / (max(ev[0], 0) + f)))


def _axis(p):
    ev, vec = np.linalg.eigh(np.cov((p - p.mean(0)).T) + 1e-9 * np.eye(2))
    return vec[:, 1]


def _side_bins(j, nb, small_low, k):
    """Bin index k rows outward from the cut on the part side (S) and on the other side (L); -1 when off the object."""
    s_idx = j - k if small_low else j + k - 1
    l_idx = j + k - 1 if small_low else j - k
    return (s_idx if 0 <= s_idx < nb else -1), (l_idx if 0 <= l_idx < nb else -1)


def candidates(pos, sig, s, min_part=3, dirs=DIRS):
    """Candidate cuts of one object.

    pos (n, 2) patch positions; sig (n, c) the signal each patch sensed; s the patch spacing.
    Returns X (m, T) traits, masks (m, n) bool for the smaller part of each cut, and meta (m, 2) = (direction index, cut offset).
    """
    pos = np.asarray(pos, float)
    sig = np.asarray(sig, float).reshape(len(pos), -1)
    n = len(pos)
    obj_axis = _axis(pos) if n >= 3 else np.array([0.0, 1.0])
    obj_elong = _elong(pos, s)
    size = np.sqrt(n * s * s)
    X, M, meta = [], [], []
    for di, th in enumerate(dirs):
        d = np.array([np.cos(th), np.sin(th)])
        u = pos @ d
        v = pos @ np.array([-np.sin(th), np.cos(th)])
        u0 = u.min()
        b = np.floor((u - u0) / s + 1e-6).astype(int)
        nb = int(b.max()) + 1
        cnt = np.bincount(b, minlength=nb).astype(float)
        vmin = np.full(nb, np.inf); vmax = np.full(nb, -np.inf)
        np.minimum.at(vmin, b, v); np.maximum.at(vmax, b, v)
        occ = cnt > 0
        w = np.where(occ, vmax - vmin + s, 0.0)
        ctr = np.where(occ, 0.5 * (vmax + vmin), 0.0)
        m = np.stack([np.bincount(b, weights=sig[:, c], minlength=nb) / np.maximum(cnt, 1) for c in range(sig.shape[1])], 1)
        var = sum(np.bincount(b, weights=sig[:, c] ** 2, minlength=nb) / np.maximum(cnt, 1) - m[:, c] ** 2 for c in range(sig.shape[1]))
        sd = np.sqrt(np.maximum(var, 0))
        g = np.full(nb, np.nan)
        both = occ[1:] & occ[:-1]
        g[1:][both] = np.linalg.norm(m[1:] - m[:-1], axis=1)[both]
        med = np.nanmedian(g) if np.isfinite(g).any() else 0.0
        cum = np.concatenate([[0.0], np.cumsum(cnt)])
        wmax = max(w.max(), s)
        umax = u.max()
        for j in range(1, nb):
            if not (occ[j - 1] and occ[j]):
                continue
            n_low = cum[j]
            n_high = n - n_low
            if min(n_low, n_high) < min_part:
                continue
            small_low = n_low <= n_high
            c = u0 + j * s
            w0 = max(min(w[j - 1], w[j]), s)
            c0 = 0.5 * (ctr[j - 1] + ctr[j])
            f = []
            for side in (0, 1):                                   # fine width profile, S then L
                for k in range(1, NFINE + 1):
                    i = _side_bins(j, nb, small_low, k)[side]
                    f.append(np.log((w[i] if i >= 0 else 0.0) + s) - np.log(w0 + s))
            for side in (0, 1):                                   # coarse width profile
                for q in range(NCOARSE):
                    ws = [w[_side_bins(j, nb, small_low, NFINE + 1 + q * COARSE_W + k)[side]] if _side_bins(j, nb, small_low, NFINE + 1 + q * COARSE_W + k)[side] >= 0 else 0.0 for k in range(COARSE_W)]
                    f.append(np.log(np.mean(ws) + s) - np.log(w0 + s))
            for side in (0, 1):                                   # centre-line shift
                for k in range(1, 4):
                    i = _side_bins(j, nb, small_low, k)[side]
                    f.append(min(abs(ctr[i] - c0) / (w0 + s), 2.0) if i >= 0 and occ[i] else 0.0)
            ext_low, ext_high = c - u0, umax - c
            ext_s, ext_l = (ext_low, ext_high) if small_low else (ext_high, ext_low)
            f += [np.log((ext_s + s) / (w0 + s)), np.log((ext_l + s) / (w0 + s)), min(n_low, n_high) / n]
            gs = lambda i: 0.0 if not (0 < i < nb and np.isfinite(g[i])) else np.log1p(g[i] / (med + NOISE))
            f += [gs(j)]
            f += [gs(j - 1), gs(j - 2), gs(j + 1), gs(j + 2)] if small_low else [gs(j + 1), gs(j + 2), gs(j - 1), gs(j - 2)]
            lows = [i for i in range(max(0, j - 4), j) if occ[i]]
            highs = [i for i in range(j, min(nb, j + 4)) if occ[i]]
            hs = lambda ii: float(np.mean(sd[ii]) / 20.0) if ii else 0.0
            f += [hs(lows), hs(highs)] if small_low else [hs(highs), hs(lows)]
            mask = (b < j) if small_low else (b >= j)
            part = pos[mask]
            frac = (c - u0) / (umax - u0 + s)
            f += [np.log(w0 / size), np.log(w0 / wmax), min(frac, 1 - frac), abs(float(d @ obj_axis)), obj_elong, _elong(part, s)]
            X.append(f); M.append(mask); meta.append((di, c))
    if not X:
        return np.zeros((0, T), np.float32), np.zeros((0, n), bool), np.zeros((0, 2))
    return np.array(X, np.float32), np.array(M), np.array(meta)
