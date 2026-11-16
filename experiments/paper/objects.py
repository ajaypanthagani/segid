"""Procedural object families for the paper's experiments.

Every generator returns one object drawn on white paper, upright, plus what the experimenter (never the model) knows about it:

    img        (h, w, 3) uint8 BGR
    sep        True if the object has a part that comes away (the event script will cut it there and drag it off)
    cut        y coordinate (image rows) of the straight cut between the part (above) and the rest (below), or None
    family     name

Separable families have the part at the top end; the part is exactly the object's pixels above the cut.  Non-separable families
(hammer, screwdriver, vase, goblet) are moved only as a whole, so by the observational definition they stay one object, even though
several of them have a change of colour and of width that looks like a cap.

Colours, sizes and proportions are drawn at random per instance; train and test instances use disjoint seeds.
"""
import cv2
import numpy as np

from experiments.bottles import make_bottle

PAPER = 255


def _canvas(h, w):
    return np.full((h, w, 3), PAPER, np.uint8)


def _col(rng, lo=40, hi=225):
    """A random colour that is clearly not paper."""
    while True:
        c = rng.integers(lo, hi, 3)
        if c.min() < 200:
            return c.astype(float)


def _shade(img, mask, col, cx, half):
    xs = np.arange(img.shape[1])[None, :].repeat(img.shape[0], 0)
    t = np.clip((xs - cx) / (half + 1e-6), -1, 1)
    sh = 0.7 + 0.3 * np.sqrt(np.clip(1 - t ** 2, 0, 1))
    img[mask] = np.clip(np.asarray(col, float)[None] * sh[mask][:, None], 0, 255).astype(np.uint8)


def _rect(shape, x0, x1, y0, y1):
    m = np.zeros(shape[:2], bool)
    m[int(round(y0)):int(round(y1)), int(round(x0)):int(round(x1))] = True
    return m


def _poly(shape, pts):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(pts, float)).astype(np.int32)], 1)
    return m.astype(bool)


def _outline(img, mask, col):
    cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cnt, -1, tuple(float(v) for v in np.asarray(col) * 0.55), 1)


# ------------------------------------------------------------------------------------------------ separable families
def bottle(rng):
    img, yc = make_bottle(rng, hard=bool(rng.random() < 0.3))
    return dict(img=img, sep=True, cut=float(round(yc)))


def jar(rng):
    H, W = int(rng.uniform(120, 170)), 130
    img = _canvas(H, W); cx = W / 2
    bw = rng.uniform(60, 92); lw = bw * rng.uniform(0.94, 1.08); lh = rng.uniform(12, 24)
    top = 4.0; yc = round(top + lh)
    body, lid = _col(rng), _col(rng)
    bm = _poly(img.shape, [(cx - bw / 2, yc), (cx + bw / 2, yc), (cx + bw / 2, H - 10), (cx + bw / 2 - 6, H - 4), (cx - bw / 2 + 6, H - 4), (cx - bw / 2, H - 10)])
    _shade(img, bm, body, cx, bw / 2)
    if rng.random() < 0.6:
        y0 = int(yc + rng.uniform(0.2, 0.35) * (H - yc)); y1 = int(y0 + rng.uniform(0.25, 0.4) * (H - yc))
        lab = bm.copy(); lab[:y0] = False; lab[y1:] = False
        img[lab] = (0.5 * img[lab] + 0.5 * _col(rng)).astype(np.uint8)
    _outline(img, bm, body)
    lm = _rect(img.shape, cx - lw / 2, cx + lw / 2, top, yc)
    _shade(img, lm, lid, cx, lw / 2)
    for x in range(int(cx - lw / 2) + 2, int(cx + lw / 2) - 1, 4):
        img[int(top) + 1:int(yc) - 1, x] = np.clip(lid * 0.75, 0, 255).astype(np.uint8)
    return dict(img=img, sep=True, cut=float(yc))


def pen(rng):
    H, W = int(rng.uniform(150, 190)), 130
    img = _canvas(H, W); cx = W / 2
    bw = rng.uniform(12, 20); ch = rng.uniform(0.18, 0.32) * H; top = 4.0; yc = round(top + ch)
    c = _col(rng)
    same = rng.random() < 0.6                                      # most pens: the cap has the body's colour
    capc = c * rng.uniform(0.85, 1.0) if same else _col(rng)
    bm = _rect(img.shape, cx - bw / 2, cx + bw / 2, yc, H - 18)
    _shade(img, bm, c, cx, bw / 2)
    tip = _poly(img.shape, [(cx - bw / 2, H - 18), (cx + bw / 2, H - 18), (cx + 1.5, H - 4), (cx - 1.5, H - 4)])
    img[tip] = np.clip(c * 0.5, 0, 255).astype(np.uint8)
    cw = bw + rng.uniform(1, 5)
    cm = _rect(img.shape, cx - cw / 2, cx + cw / 2, top, yc)
    _shade(img, cm, capc, cx, cw / 2)
    clip = _rect(img.shape, cx + cw / 2 - 1, cx + cw / 2 + 2, top + 4, yc - 6)
    img[clip] = np.clip(capc * 0.6, 0, 255).astype(np.uint8)
    return dict(img=img, sep=True, cut=float(yc))


def tube(rng):
    H, W = int(rng.uniform(140, 185)), 130
    img = _canvas(H, W); cx = W / 2
    bw = rng.uniform(40, 62); cw = bw * rng.uniform(0.32, 0.55); ch = rng.uniform(9, 17); top = 4.0; yc = round(top + ch)
    body, cap = _col(rng), _col(rng)
    sh = rng.uniform(18, 40)
    bm = _poly(img.shape, [(cx - cw / 2, yc), (cx + cw / 2, yc), (cx + bw / 2, yc + sh), (cx + bw / 2 + 3, H - 6), (cx - bw / 2 - 3, H - 6), (cx - bw / 2, yc + sh)])
    _shade(img, bm, body, cx, bw / 2)
    img[H - 9:H - 5, :][bm[H - 9:H - 5, :]] = np.clip(body * 0.6, 0, 255).astype(np.uint8)     # crimped end
    _outline(img, bm, body)
    k = rng.uniform(1.0, 1.25)
    cm = _rect(img.shape, cx - cw / 2 * k, cx + cw / 2 * k, top, yc)
    _shade(img, cm, cap, cx, cw / 2 * k)
    return dict(img=img, sep=True, cut=float(yc))


def spray(rng):
    H, W = int(rng.uniform(150, 190)), 130
    img = _canvas(H, W); cx = W / 2
    bw = rng.uniform(34, 52); body, cap = _col(rng), _col(rng)
    nh = rng.uniform(6, 11); capw = bw * rng.uniform(0.6, 0.95); caph = rng.uniform(22, 36); top = 4.0
    yc = round(top + nh + caph)
    bm = _rect(img.shape, cx - bw / 2, cx + bw / 2, yc, H - 6)
    _shade(img, bm, body, cx, bw / 2)
    _outline(img, bm, body)
    cm = _rect(img.shape, cx - capw / 2, cx + capw / 2, top + nh, yc)
    _shade(img, cm, cap, cx, capw / 2)
    nz = bw * rng.uniform(0.18, 0.3)
    img[_rect(img.shape, cx - nz / 2, cx + nz / 2 + 4, top, top + nh)] = np.clip(cap * 0.7, 0, 255).astype(np.uint8)
    return dict(img=img, sep=True, cut=float(yc))


def cup(rng):
    """A takeaway cup with a lid."""
    H, W = int(rng.uniform(120, 165)), 130
    img = _canvas(H, W); cx = W / 2
    tw = rng.uniform(52, 80); bwid = tw * rng.uniform(0.62, 0.8); lh = rng.uniform(9, 16); top = 4.0; yc = round(top + lh)
    body, lid = _col(rng), _col(rng, 30, 120) if rng.random() < 0.5 else _col(rng)
    bm = _poly(img.shape, [(cx - tw / 2, yc), (cx + tw / 2, yc), (cx + bwid / 2, H - 4), (cx - bwid / 2, H - 4)])
    _shade(img, bm, body, cx, tw / 2)
    if rng.random() < 0.6:
        y0 = int(yc + 0.35 * (H - yc)); y1 = int(yc + 0.6 * (H - yc))
        sl = bm.copy(); sl[:y0] = False; sl[y1:] = False
        img[sl] = (0.4 * img[sl] + 0.6 * _col(rng)).astype(np.uint8)
    _outline(img, bm, body)
    lw = tw * rng.uniform(1.02, 1.12)
    lm = _poly(img.shape, [(cx - lw / 2, yc), (cx + lw / 2, yc), (cx + lw / 2 - 4, top + lh * 0.35), (cx + lw * 0.3, top), (cx - lw * 0.3, top), (cx - lw / 2 + 4, top + lh * 0.35)])
    _shade(img, lm, lid, cx, lw / 2)
    return dict(img=img, sep=True, cut=float(yc))


def blocks(rng):
    """Two stacked toy blocks: the top one lifts off."""
    H, W = int(rng.uniform(110, 170)), 130
    img = _canvas(H, W); cx = W / 2 + rng.uniform(-6, 6)
    bw = rng.uniform(40, 80); tw = bw * rng.uniform(0.5, 1.15); th = rng.uniform(0.25, 0.5) * H; top = 4.0; yc = round(top + th)
    a, b = _col(rng), _col(rng)
    off = rng.uniform(-0.15, 0.15) * bw
    bm = _rect(img.shape, W / 2 - bw / 2, W / 2 + bw / 2, yc, H - 4)
    img[bm] = b.astype(np.uint8); _outline(img, bm, b)
    tm = _rect(img.shape, cx + off - tw / 2, cx + off + tw / 2, top, yc)
    img[tm] = a.astype(np.uint8); _outline(img, tm, a)
    return dict(img=img, sep=True, cut=float(yc))


def lipstick(rng):
    H, W = int(rng.uniform(120, 170)), 130
    img = _canvas(H, W); cx = W / 2
    bw = rng.uniform(22, 34); ch = rng.uniform(0.35, 0.5) * H; top = 4.0; yc = round(top + ch)
    body, cap = _col(rng), _col(rng)
    bm = _rect(img.shape, cx - bw / 2, cx + bw / 2, yc, H - 4)
    _shade(img, bm, body, cx, bw / 2)
    ring = _rect(img.shape, cx - bw / 2, cx + bw / 2, yc, yc + 3)
    img[ring] = np.clip(body * 0.6 + 60, 0, 255).astype(np.uint8)
    cw = bw * rng.uniform(1.0, 1.08)
    cm = _rect(img.shape, cx - cw / 2, cx + cw / 2, top, yc)
    _shade(img, cm, cap, cx, cw / 2)
    return dict(img=img, sep=True, cut=float(yc))


# ------------------------------------------------------------------------------------------- non-separable families
def hammer(rng):
    H, W = int(rng.uniform(150, 190)), 130
    img = _canvas(H, W); cx = W / 2
    hw = rng.uniform(70, 110); hh = rng.uniform(16, 26); sw = rng.uniform(9, 15); top = 4.0
    head, handle = _col(rng, 60, 140), _col(rng)
    sm = _rect(img.shape, cx - sw / 2, cx + sw / 2, top + hh, H - 4)
    _shade(img, sm, handle, cx, sw / 2)
    grip = sm.copy(); grip[:int(H * 0.6)] = False
    img[grip] = np.clip(img[grip] * 0.7, 0, 255).astype(np.uint8)
    off = rng.uniform(-0.2, 0.2) * hw
    hm = _poly(img.shape, [(cx - hw / 2 + off, top + 3), (cx + hw / 2 + off - 8, top), (cx + hw / 2 + off, top + hh * 0.5), (cx + hw / 2 + off - 8, top + hh), (cx - hw / 2 + off, top + hh - 2)])
    img[hm] = head.astype(np.uint8); _outline(img, hm, head)
    return dict(img=img, sep=False, cut=None)


def screwdriver(rng):
    H, W = int(rng.uniform(150, 190)), 130
    img = _canvas(H, W); cx = W / 2
    gw = rng.uniform(18, 30); gh = rng.uniform(0.3, 0.45) * H; sw = rng.uniform(4, 8); top = 4.0
    grip, steel = _col(rng), np.array([170, 170, 175], float) * rng.uniform(0.8, 1.1)
    gm = _poly(img.shape, [(cx - gw / 2 + 3, top), (cx + gw / 2 - 3, top), (cx + gw / 2, top + 6), (cx + gw / 2, top + gh - 6), (cx + gw / 2 - 5, top + gh), (cx - gw / 2 + 5, top + gh), (cx - gw / 2, top + gh - 6), (cx - gw / 2, top + 6)])
    _shade(img, gm, grip, cx, gw / 2)
    for x in range(int(cx - gw / 2) + 3, int(cx + gw / 2) - 2, 4):
        img[int(top) + 6:int(top + gh) - 6, x] = np.clip(grip * 0.7, 0, 255).astype(np.uint8)
    sm = _rect(img.shape, cx - sw / 2, cx + sw / 2, top + gh, H - 10)
    _shade(img, sm, np.clip(steel, 0, 230), cx, sw / 2)
    img[_poly(img.shape, [(cx - sw / 2, H - 10), (cx + sw / 2, H - 10), (cx + 1, H - 4), (cx - 1, H - 4)])] = np.clip(steel * 0.6, 0, 255).astype(np.uint8)
    return dict(img=img, sep=False, cut=None)


def vase(rng):
    """A bottle-like vase in one material: a neck, a shoulder, often a painted band, no cap."""
    H, W = int(rng.uniform(140, 190)), 130
    img = _canvas(H, W); cx = W / 2
    bw = rng.uniform(50, 90); nw = bw * rng.uniform(0.25, 0.5); lip = nw * rng.uniform(1.1, 1.5)
    top = 4.0; lh = rng.uniform(4, 8); nh = rng.uniform(15, 35); sh = rng.uniform(0.15, 0.3) * H
    c = _col(rng)
    t = np.linspace(0, 1, 14)
    y1 = top + lh + nh
    sx = nw / 2 + (bw / 2 - nw / 2) * np.sin(t * np.pi / 2)
    left = [(cx - lip / 2, top), (cx - lip / 2, top + lh), (cx - nw / 2, top + lh), (cx - nw / 2, y1)] + [(cx - x, y1 + sh * tt) for x, tt in zip(sx, t)] + [(cx - bw / 2, H - 12), (cx - bw / 2 + 10, H - 4)]
    pts = left + [(2 * cx - x, y) for x, y in left[::-1]]
    m = _poly(img.shape, pts)
    _shade(img, m, c, cx, bw / 2)
    if rng.random() < 0.7:                                       # a painted band, sometimes right at the neck
        y0 = int(top + lh + (rng.uniform(0, 0.4) * nh if rng.random() < 0.5 else nh + sh * rng.uniform(0.6, 1.4)))
        band = m.copy(); band[:y0] = False; band[y0 + int(rng.uniform(5, 14)):] = False
        img[band] = (0.35 * img[band] + 0.65 * _col(rng)).astype(np.uint8)
    _outline(img, m, c)
    return dict(img=img, sep=False, cut=None)


def goblet(rng):
    H, W = int(rng.uniform(140, 185)), 130
    img = _canvas(H, W); cx = W / 2
    cw = rng.uniform(55, 90); ch = rng.uniform(0.35, 0.5) * H; stw = rng.uniform(6, 12); bw = cw * rng.uniform(0.6, 0.9); top = 4.0
    c = _col(rng)
    t = np.linspace(0, 1, 12)
    bowl = [(cx - cw / 2 * (1 - 0.85 * tt ** 2), top + ch * tt) for tt in t]
    left = bowl + [(cx - stw / 2, top + ch), (cx - stw / 2, H - 14), (cx - bw / 2, H - 6), (cx - bw / 2, H - 4)]
    pts = left + [(2 * cx - x, y) for x, y in left[::-1]]
    m = _poly(img.shape, pts)
    _shade(img, m, c, cx, cw / 2)
    if rng.random() < 0.5:                                       # two-tone: a coloured bowl on a clear-coloured stem
        two = m.copy(); two[int(top + ch):] = False
        img[two] = (0.3 * img[two] + 0.7 * _col(rng)).astype(np.uint8)
    _outline(img, m, c)
    return dict(img=img, sep=False, cut=None)


SEPARABLE = dict(bottle=bottle, jar=jar, pen=pen, tube=tube, spray=spray, cup=cup, blocks=blocks, lipstick=lipstick)
WHOLE = dict(hammer=hammer, screwdriver=screwdriver, vase=vase, goblet=goblet)
FAMILIES = dict(**SEPARABLE, **WHOLE)


def make(family, seed):
    rng = np.random.default_rng(seed)
    o = FAMILIES[family](rng)
    o["family"] = family
    return o


def transform(o, kind):
    """Test-time changes the model never trained on: rotation, scale, colour, sensor noise.  Returns a new object dict whose `cut`
    is replaced by `part_fn(x, y) -> bool` in image coordinates (or None for whole objects)."""
    img = o["img"].copy()
    h, w = img.shape[:2]
    cut = o["cut"]
    part = (lambda x, y: y < cut) if cut is not None else None
    if kind == "none":
        pass
    elif kind == "rot90":                     # part now on the left
        img = np.ascontiguousarray(np.rot90(img, 1))
        part = (lambda x, y: x < cut) if cut is not None else None
    elif kind == "rot180":                    # part now at the bottom
        img = np.ascontiguousarray(img[::-1, ::-1])
        part = (lambda x, y: y >= h - cut) if cut is not None else None
    elif kind.startswith("scale"):
        k = float(kind[5:])
        img = cv2.resize(img, (int(round(w * k)), int(round(h * k))), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
        part = (lambda x, y: y < cut * k) if cut is not None else None
    elif kind == "recolour":                  # every colour on the object turned by a fixed random hue rotation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        obj = img.min(2) < 235
        hsv[..., 0][obj] = ((hsv[..., 0][obj].astype(int) + 60) % 180).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    elif kind == "dark":                      # dimmer light: every object pixel scaled by 0.6
        obj = img.min(2) < 235
        img[obj] = (img[obj] * 0.6).astype(np.uint8)
    elif kind == "noise":                     # sensor noise on the object (sd 12 grey levels)
        obj = img.min(2) < 235
        n = np.random.default_rng(0).normal(0, 12, img.shape)
        img = np.where(obj[..., None], np.clip(img + n, 0, 234), img).astype(np.uint8)
    else:
        raise ValueError(kind)
    return dict(img=img, sep=o["sep"], part=part, family=o["family"])
