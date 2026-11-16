"""Pair descriptor for the image sensor: raw local appearance in a seam-aligned frame.

Same layout and dimensionality as `descriptor.py` (side / seam / app groups, three radii,
4 x 4 grids), so the learner is unchanged.  With no depth there is no height or normal, so the
five channels are image structure instead: normalised lightness, the two gradient components
in the seam frame, gradient magnitude and chroma deviation.  Nothing names an object or a
seam type; the same code runs for every pair.
"""
import cv2
import numpy as np

from .descriptor import D, DIMS, G, H, NCH, RADII, SLICES   # noqa: F401  (shared layout)

P = 16          # side of the sampled patch in pixels


def _patch(lab, mid, ang, r):
    c, s = np.cos(ang), np.sin(ang)
    sc = 2 * r / P
    o = 0.5 * sc - r
    # output pixel (j, i) -> source = mid + R(ang) @ (sc*j + o, sc*i + o)
    M = np.array([[c * sc, -s * sc, mid[0] + (c - s) * o],
                  [s * sc, c * sc, mid[1] + (s + c) * o]], np.float32)
    return cv2.warpAffine(lab, M, (P, P), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)


def _channels(pl):
    L, a, b = pl[..., 0], pl[..., 1], pl[..., 2]
    Lz = (L - L.mean()) / (L.std() + 4.0)
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    sc = mag.mean() + 8.0
    chroma = np.sqrt((a - a.mean()) ** 2 + (b - b.mean()) ** 2) / 12.0
    ch = np.stack([Lz, gx / sc, gy / sc, mag / sc, chroma])
    return np.stack([cv2.resize(c, (G, G), interpolation=cv2.INTER_AREA) for c in ch])


def _one(lab, mid, ang, s):
    side, seam, app = [], [], []
    for rf in RADII:
        r = rf * s
        pl = _patch(lab, mid, ang, r)
        grid = _channels(pl)
        left, right = grid[:, :, :H], grid[:, :, H:]
        mir = right[:, :, ::-1].copy()
        mir[1] *= -1.0                              # gradient along the seam axis flips under mirroring
        side += [left.ravel(), right.ravel()]
        seam.append((left - mir).ravel())
        lm = pl[:, :P // 2].reshape(-1, 3).mean(0)
        rm = pl[:, P // 2:].reshape(-1, 3).mean(0)
        app.append(np.linalg.norm(lm - rm) / 100.0)
    return np.concatenate(side + seam + [np.array(app)])


def pair_descriptors(frame, pairs, s):
    pairs = np.asarray(pairs, int).reshape(-1, 2)
    A = np.zeros((len(pairs), D), np.float32)
    B = np.zeros((len(pairs), D), np.float32)
    if len(pairs) == 0:
        return A, B
    lab = cv2.cvtColor(frame.img, cv2.COLOR_BGR2Lab).astype(np.float32)
    for q, (a, b) in enumerate(pairs):
        pa, pb = frame.pos[a, :2], frame.pos[b, :2]
        mid = 0.5 * (pa + pb)
        ang = float(np.arctan2(pb[1] - pa[1], pb[0] - pa[0]))
        A[q] = _one(lab, mid, ang, s)
        B[q] = _one(lab, mid, ang + np.pi, s)
    return A, B


def default_pi(frame, ci, cj):
    """Material-continuity default used before anything is learned: similar colour => more likely joined."""
    blurred = cv2.blur(cv2.cvtColor(frame.img, cv2.COLOR_BGR2Lab).astype(np.float32), (5, 5))
    h, w = blurred.shape[:2]

    def pt(i):
        x = np.clip(frame.pos[i, 0].astype(int), 0, w - 1)
        y = np.clip(frame.pos[i, 1].astype(int), 0, h - 1)
        return blurred[y, x]
    return np.exp(-np.linalg.norm(pt(ci) - pt(cj), axis=-1) / 20.0)
