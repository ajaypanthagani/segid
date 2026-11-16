"""Pair descriptor computed from patches alone (no image).

Each patch carries the colour it sampled (`frame.patch_rgb`).  The patches that are currently visible are put on a small
grid, one cell per patch spacing; cells with no patch are paper-coloured.  A pair's descriptor is read from that small grid in
a frame turned to the pair's own direction, with the same layout as `descriptor_image.py` (side / seam / app, three radii,
4 x 4 grids, five channels) so the learner is unchanged.  Everything is done for all pairs at once.
"""
import os

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .descriptor import D, G, H, NCH, NCTX, RADII           # noqa: F401  (shared layout)

FEAT = os.environ.get("OL_FEAT", "hand")                # 'hand': five hand-written channels; 'learned': generic learned filter bank
PHYS = os.environ.get("OL_PHYS", "0") == "1"            # keep the physical window size when patches are finer (radii are in 5 px units)
CTX_PX = 48.0                                           # context radius in pixels for the silhouette profile
_F = None

P = 16
PAPER_BGR = np.array([[[236, 242, 244]]], np.uint8)


def occupancy_grid(frame, s):
    h, w = frame.img.shape[:2] if frame.img is not None else (int(frame.pos[:, 1].max() / s) * s + s, int(frame.pos[:, 0].max() / s) * s + s)
    gh, gw = int(np.ceil(h / s)), int(np.ceil(w / s))
    occ = np.zeros((gh, gw), np.float32)
    idx = np.flatnonzero(frame.vis)
    occ[np.clip((frame.pos[idx, 1] / s).astype(int), 0, gh - 1), np.clip((frame.pos[idx, 0] / s).astype(int), 0, gw - 1)] = 1.0
    return occ


def patch_grid(frame, s):
    """(gh, gw, 3) Lab grid of the visible patches' colours; paper elsewhere."""
    h, w = frame.img.shape[:2] if frame.img is not None else (int(frame.pos[:, 1].max() / s) * s + s, int(frame.pos[:, 0].max() / s) * s + s)
    gh, gw = int(np.ceil(h / s)), int(np.ceil(w / s))
    bgr = np.repeat(PAPER_BGR, gh * gw, 1).reshape(gh, gw, 3).copy()
    idx = np.flatnonzero(frame.vis)
    cx = np.clip((frame.pos[idx, 0] / s).astype(int), 0, gw - 1)
    cy = np.clip((frame.pos[idx, 1] / s).astype(int), 0, gh - 1)
    bgr[cy, cx] = frame.patch_rgb[idx]
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float32)


def _windows(grid, mid, ang, r, s):
    """(N, P, P, 3) windows of half-width r cells, centred at mid (pixels), turned by ang."""
    n = len(mid)
    c, sn = np.cos(ang)[:, None, None], np.sin(ang)[:, None, None]
    t = (np.arange(P) + 0.5) / P * 2 * r - r
    u, v = t[None, None, :], t[None, :, None]            # u along the pair, v across
    cx = (mid[:, 0] / s - 0.5)[:, None, None]
    cy = (mid[:, 1] / s - 0.5)[:, None, None]
    mx = (cx + c * u - sn * v).astype(np.float32).reshape(n * P, P)
    my = (cy + sn * u + c * v).astype(np.float32).reshape(n * P, P)
    out = cv2.remap(grid, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(n, P, P, 3)


def _sobel(L):
    pad = np.pad(L, ((0, 0), (1, 1), (1, 1)), mode="edge")
    dx = pad[:, :, 2:] - pad[:, :, :-2]
    gx = dx[:, :-2] + 2 * dx[:, 1:-1] + dx[:, 2:]
    dy = pad[:, 2:] - pad[:, :-2]
    gy = dy[:, :, :-2] + 2 * dy[:, :, 1:-1] + dy[:, :, 2:]
    return gx, gy


def _bank():
    global _F
    if _F is None:
        f = np.load(os.path.join(os.path.dirname(__file__), "filters_generic.npz"))["F"].astype(np.float32)      # (K, 5, 5, 3)
        _F = np.concatenate([f, f[:, :, ::-1]]).reshape(2 * len(f), -1).T.copy()                                # (75, 2K): filters and their mirror images
    return _F


def _learned(pl, chunk=96):
    """(N, 2K+1, G, G): responses of the generic learned filters (and their mirror images) plus normalised lightness."""
    F = _bank()
    n = len(pl)
    out = np.zeros((n, F.shape[1] + 1, G, G), np.float32)
    k = P // G
    for a in range(0, n, chunk):
        w = pl[a:a + chunk]
        pad = np.pad(w, ((0, 0), (2, 2), (2, 2), (0, 0)), mode="edge")
        x = sliding_window_view(pad, (5, 5, 3), axis=(1, 2, 3))[:, :, :, 0].reshape(len(w), P * P, 75)
        x = x - x.mean(2, keepdims=True)
        r = np.maximum(x @ F / 25.0 - 0.3, 0).reshape(len(w), P, P, -1).transpose(0, 3, 1, 2)
        out[a:a + chunk, :-1] = r.reshape(len(w), -1, G, k, G, k).mean((3, 5))
        L = w[..., 0]
        Lz = (L - L.mean((1, 2), keepdims=True)) / (L.std((1, 2), keepdims=True) + 4.0)
        out[a:a + chunk, -1] = Lz.reshape(len(w), G, k, G, k).mean((2, 4))
    return out


def _channels(pl):
    """(N, 5, G, G) from windows (N, P, P, 3)."""
    if FEAT == "learned":
        return _learned(pl)
    L, a, b = pl[..., 0], pl[..., 1], pl[..., 2]
    Lz = (L - L.mean((1, 2), keepdims=True)) / (L.std((1, 2), keepdims=True) + 4.0)
    gx, gy = _sobel(L)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    sc = mag.mean((1, 2), keepdims=True) + 8.0
    chroma = np.sqrt((a - a.mean((1, 2), keepdims=True)) ** 2 + (b - b.mean((1, 2), keepdims=True)) ** 2) / 12.0
    ch = np.stack([Lz, gx / sc, gy / sc, mag / sc, chroma], 1)            # (N, 5, P, P)
    k = P // G
    return ch.reshape(len(ch), 5, G, k, G, k).mean((3, 5))


def _ctx(occ, mid, ang, s):
    """Silhouette profile across the pair: how wide the object is at 8 distances along the pair direction, relative to its width at
    the seam.  Same for any object; a step in width shows up as a step in the profile."""
    n = len(mid)
    R = CTX_PX / s
    Pc, nb = 32, 8
    c, sn = np.cos(ang)[:, None, None], np.sin(ang)[:, None, None]
    t = (np.arange(Pc) + 0.5) / Pc * 2 * R - R
    u, v = t[None, None, :], t[None, :, None]
    mx = ((mid[:, 0] / s - 0.5)[:, None, None] + c * u - sn * v).astype(np.float32).reshape(n * Pc, Pc)
    my = ((mid[:, 1] / s - 0.5)[:, None, None] + sn * u + c * v).astype(np.float32).reshape(n * Pc, Pc)
    w = cv2.remap(occ, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0).reshape(n, Pc, Pc)
    width = w.sum(1).reshape(n, nb, Pc // nb).mean(2)                  # occupied length across the seam direction, per u-bin
    ref = 0.5 * (width[:, nb // 2 - 1] + width[:, nb // 2]) + 1e-3
    return np.clip(width / ref[:, None] - 1.0, -1.0, 1.0)


def _one(grid, mid, ang, s, occ=None):
    side, seam, app = [], [], []
    n = len(mid)
    rs = 5.0 / s if PHYS else 1.0
    for rf in RADII:
        pl = _windows(grid, mid, ang, rf * rs, s)
        ch = _channels(pl)
        left, right = ch[:, :, :, :H], ch[:, :, :, H:]
        if FEAT == "learned":
            K = (ch.shape[1] - 1) // 2
            mir = right[:, list(range(K, 2 * K)) + list(range(K)) + [2 * K], :, ::-1].copy()
        else:
            mir = right[:, :, :, ::-1].copy()
            mir[:, 1] *= -1.0
        side += [left.reshape(n, -1), right.reshape(n, -1)]
        seam.append((left - mir).reshape(n, -1))
        lm, rm = pl[:, :, :P // 2].mean((1, 2)), pl[:, :, P // 2:].mean((1, 2))
        app.append((np.linalg.norm(lm - rm, axis=1) / 100.0)[:, None])
    if NCTX:
        app.append(_ctx(occ, mid, ang, s))
    return np.concatenate(side + seam + app, 1)


def pair_descriptors(frame, pairs, s):
    pairs = np.asarray(pairs, int).reshape(-1, 2)
    if len(pairs) == 0:
        return np.zeros((0, D), np.float32), np.zeros((0, D), np.float32)
    grid = patch_grid(frame, s)
    pa, pb = frame.pos[pairs[:, 0], :2], frame.pos[pairs[:, 1], :2]
    mid = 0.5 * (pa + pb)
    ang = np.arctan2(pb[:, 1] - pa[:, 1], pb[:, 0] - pa[:, 0])
    occ = occupancy_grid(frame, s) if NCTX else None
    A, B = [], []
    for a in range(0, len(mid), 800):                      # cv2.remap limits the height of its output
        sl = slice(a, a + 800)
        A.append(_one(grid, mid[sl], ang[sl], s, occ)); B.append(_one(grid, mid[sl], ang[sl] + np.pi, s, occ))
    return np.concatenate(A).astype(np.float32), np.concatenate(B).astype(np.float32)


def default_pi(frame, ci, cj, s=None):
    """Material-continuity default before anything is learned: similar patch colours => more likely joined."""
    lab = cv2.cvtColor(frame.patch_rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2Lab).astype(np.float32).reshape(-1, 3)
    return np.exp(-np.linalg.norm(lab[ci] - lab[cj], axis=-1) / 20.0)
