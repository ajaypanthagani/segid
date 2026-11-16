"""Pair descriptor phi(a, b): raw local geometry in a seam-aligned frame (section 9).

No shape features are hand-chosen.  For every candidate pair we express the visible
local surface (positions, normals) in a frame aligned to the seam's own normal and
tangent, and rasterise it onto a fixed grid at three radii.  The same code runs for
every pair of every object.
"""
import os

import numpy as np
from scipy.spatial import cKDTree

G = 4                       # grid cells per side
NCH = int(os.environ.get("OL_NCH", 5))   # channels per cell (geometry: occupancy, height, normal x/y/z; patch mode: see descriptor_patch)
NCTX = int(os.environ.get("OL_CTX", 0))  # extra context numbers (patch mode), kept with the appearance group
RADII = (2.0, 3.0, 5.0)     # in units of the patch spacing s
H = G // 2
D_SIDE = 2 * NCH * G * H * len(RADII)
D_SEAM = NCH * G * H * len(RADII)
D_APP = len(RADII) + NCTX
DIMS = {"side": D_SIDE, "seam": D_SEAM, "app": D_APP}
D = D_SIDE + D_SEAM + D_APP
SLICES = {"side": slice(0, D_SIDE), "seam": slice(D_SIDE, D_SIDE + D_SEAM), "app": slice(D_SIDE + D_SEAM, D)}


def _axes(pa, pb, na, nb):
    z = na + nb
    nz = np.linalg.norm(z)
    z = z / nz if nz > 0.2 else na
    d = pb - pa
    d = d - (d @ z) * z
    nd = np.linalg.norm(d)
    if nd < 1e-9:
        d = np.cross(z, np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0]))
        nd = np.linalg.norm(d)
    x = d / nd
    y = np.cross(z, x)
    return np.stack([x, y, z], 1)


def _grid(Q, Nn, col, r):
    u, v, w = Q[:, 0], Q[:, 1], Q[:, 2]
    m = (np.abs(u) < r) & (np.abs(v) < r) & (np.abs(w) < r)
    grid = np.zeros((NCH, G, G))
    if m.any():
        cu = np.clip(((u[m] / r + 1) * 0.5 * G).astype(int), 0, G - 1)
        cv = np.clip(((v[m] / r + 1) * 0.5 * G + 0.25).astype(int), 0, G - 1)   # quarter-cell offset: the seam axis sits mid-cell
        cell = cv * G + cu
        cnt = np.bincount(cell, minlength=G * G).astype(float)
        nz = cnt > 0
        chans = [w[m] / r, Nn[m, 0], Nn[m, 1], Nn[m, 2]]
        grid[0] = nz.reshape(G, G)
        for c, vals in enumerate(chans):
            s = np.bincount(cell, weights=vals, minlength=G * G)
            mean = np.zeros(G * G)
            mean[nz] = s[nz] / cnt[nz]
            grid[1 + c] = mean.reshape(G, G)
    left = m & (u < 0)
    right = m & (u >= 0)
    app = abs(col[left].mean() - col[right].mean()) if left.any() and right.any() else 0.0
    return grid, app


def _one(mid, pts, nrm, col, pa, pb, na, nb, s):
    ax = _axes(pa, pb, na, nb)
    side, seam, app = [], [], []
    for rf in RADII:
        r = rf * s
        grid, a = _grid((pts - mid) @ ax, nrm @ ax, col, r)
        left = grid[:, :, :H]
        right = grid[:, :, H:]
        mir = right[:, :, ::-1].copy()
        mir[2] *= -1.0                      # nx' flips under u -> -u
        side += [left.ravel(), right.ravel()]
        seam.append((left - mir).ravel())
        app.append(a)
    return np.concatenate(side + seam + [np.array(app)])


def pair_descriptors(frame, pairs, s):
    """Return (phi_ab, phi_ba): descriptors for both orderings of each pair, shape (m, D)."""
    pairs = np.asarray(pairs, int).reshape(-1, 2)
    m = len(pairs)
    A = np.zeros((m, D), np.float32)
    B = np.zeros((m, D), np.float32)
    if m == 0:
        return A, B
    vi = np.flatnonzero(frame.vis)
    pts, nrm, col = frame.pos[vi], frame.nrm[vi], frame.col[vi]
    tree = cKDTree(pts)
    rmax = max(RADII) * s * 1.8
    for q, (a, b) in enumerate(pairs):
        pa, pb, na, nb = frame.pos[a], frame.pos[b], frame.nrm[a], frame.nrm[b]
        mid = 0.5 * (pa + pb)
        idx = tree.query_ball_point(mid, rmax)
        P, Nn, C = pts[idx], nrm[idx], col[idx]
        A[q] = _one(mid, P, Nn, C, pa, pb, na, nb, s)
        B[q] = _one(mid, P, Nn, C, pb, pa, nb, na, s)
    return A, B
