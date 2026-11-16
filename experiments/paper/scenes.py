"""Procedural scenes for the paper: single objects with a part that comes off, everyday arrangements (things on a table, books
in a shelf, drawers in a cabinet, fruit in a bowl, pencils in a cup, a stack of books) and things that never come apart.

Every generator returns
    img      (h, w, 3) uint8 BGR on white paper (the paper is not part of anything)
    units    list of dict(mask=(h, w) bool, parent=int or None): each unit is something that can be taken away from what it
             rests in or on.  parent is the index of the unit it belongs to (a cap on a bottle that stands on a table), None if
             it belongs to the scene itself.  Whole objects have no units.
    family   name

The model never sees any of this: it only gets the picture, and later the motions of the experimenter's hand.  Ground truth for
a still picture is the finest partition the units induce (the scene with every unit taken away, and each unit minus its own units).
"""
import cv2
import numpy as np

from experiments.paper import objects as O

PAPER = 255


def _canvas(h, w):
    return np.full((h, w, 3), PAPER, np.uint8)


def _objmask(img):
    return img.min(2) < 245


def _draw(img, mask, col, shade_cx=None, half=None):
    if shade_cx is None:
        img[mask] = np.clip(col, 0, 255).astype(np.uint8)
    else:
        O._shade(img, mask, col, shade_cx, half)


def _wood(rng):
    base = np.array([60, 110, 160], float) * rng.uniform(0.6, 1.2)          # BGR browns
    return np.clip(base + rng.normal(0, 12, 3), 20, 220)


# -------------------------------------------------------------------------------------- single objects (from objects.py)
def _single(fn):
    def gen(rng):
        o = fn(rng)
        img = o["img"]
        units = []
        if o["sep"]:
            m = _objmask(img)
            m[int(o["cut"]):] = False
            units = [dict(mask=m, parent=None)]
        return dict(img=img, units=units)
    return gen


# ---------------------------------------------------------------------------------------------------------- scenes
def _item(rng, kind, w_avail):
    """One thing to stand on a table: returns (img, mask, units-inside) drawn in its own small canvas, bottom-aligned."""
    if kind == "box":
        w, h = rng.uniform(22, min(50, w_avail)), rng.uniform(20, 55)
        img = _canvas(int(h) + 2, int(w) + 2); m = O._rect(img.shape, 1, w + 1, 1, h + 1); c = O._col(rng)
        img[m] = c.astype(np.uint8)
        if rng.random() < 0.6:
            lab = O._rect(img.shape, 4, w - 2, h * 0.3, h * 0.6); img[lab & m] = O._col(rng).astype(np.uint8)
        O._outline(img, m, c)
        return img, m, []
    if kind == "can":
        w, h = rng.uniform(18, min(30, w_avail)), rng.uniform(26, 45)
        img = _canvas(int(h) + 2, int(w) + 2); m = O._rect(img.shape, 1, w + 1, 1, h + 1); c = O._col(rng)
        O._shade(img, m, c, w / 2 + 1, w / 2)
        top = O._rect(img.shape, 1, w + 1, 1, 4); img[top & m] = np.clip(c * 0.6 + 70, 0, 255).astype(np.uint8)
        return img, m, []
    if kind == "ball":
        r = rng.uniform(10, min(22, w_avail / 2))
        img = _canvas(int(2 * r) + 3, int(2 * r) + 3); m = np.zeros(img.shape[:2], np.uint8)
        cv2.circle(m, (int(r) + 1, int(r) + 1), int(r), 1, -1); m = m.astype(bool); c = O._col(rng)
        O._shade(img, m, c, r + 1, r)
        return img, m, []
    if kind == "book":
        w, h = rng.uniform(30, min(60, w_avail)), rng.uniform(7, 14)
        img = _canvas(int(h) + 2, int(w) + 2); m = O._rect(img.shape, 1, w + 1, 1, h + 1); c = O._col(rng)
        img[m] = c.astype(np.uint8)
        pages = O._rect(img.shape, 3, w - 1, 3, h - 1); img[pages & m] = (235, 235, 230)
        O._outline(img, m, c)
        return img, m, []
    if kind == "mug":
        w, h = rng.uniform(20, min(32, w_avail - 10)), rng.uniform(22, 34)
        img = _canvas(int(h) + 2, int(w) + 14); m = O._rect(img.shape, 1, w + 1, 1, h + 1); c = O._col(rng)
        O._shade(img, m, c, w / 2 + 1, w / 2)
        hm = np.zeros(img.shape[:2], np.uint8)
        cv2.ellipse(hm, (int(w + 1), int(h / 2 + 1)), (8, int(h / 4) + 2), 0, -90, 90, 1, 4)
        hm = hm.astype(bool) & ~m
        img[hm] = np.clip(c * 0.85, 0, 255).astype(np.uint8)
        return img, m | hm, []
    if kind == "bottle":
        b, yc = O.make_bottle(rng)
        k = rng.uniform(0.35, 0.5)
        b = cv2.resize(b, (int(b.shape[1] * k), int(b.shape[0] * k)), interpolation=cv2.INTER_AREA)
        m = _objmask(b)
        ys, xs = np.nonzero(m)
        b, m = b[ys.min():ys.max() + 1, xs.min():xs.max() + 1], m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        cap = m.copy(); cap[int(round(yc * k)) - ys.min():] = False
        return b, m, [cap]
    raise ValueError(kind)


def table(rng):
    W, H = int(rng.uniform(170, 220)), int(rng.uniform(150, 200))
    img = _canvas(H, W)
    top_y = int(H * rng.uniform(0.38, 0.5)); th = int(rng.uniform(7, 12)); lw = int(rng.uniform(7, 12)); wood = _wood(rng)
    x0, x1 = 4, W - 4
    tab = O._rect(img.shape, x0, x1, top_y, top_y + th)
    legs = O._rect(img.shape, x0 + 8, x0 + 8 + lw, top_y + th, H - 3) | O._rect(img.shape, x1 - 8 - lw, x1 - 8, top_y + th, H - 3)
    img[tab | legs] = wood.astype(np.uint8)
    img[O._rect(img.shape, x0, x1, top_y, top_y + 2)] = np.clip(wood * 1.2, 0, 255).astype(np.uint8)
    O._outline(img, tab | legs, wood)
    units = []
    n = int(rng.integers(1, 4))
    slots = np.linspace(x0 + 8, x1 - 8, n + 1)
    for k in range(n):
        kind = rng.choice(["box", "can", "ball", "book", "mug", "bottle"])
        avail = slots[k + 1] - slots[k] - 6
        it, m, inner = _item(rng, kind, avail)
        ih, iw = m.shape
        if iw > avail + 6 or ih > top_y - 2:
            continue
        ox = int(slots[k] + rng.uniform(0, max(avail - iw, 0)) + 3); oy = top_y - ih
        sub = img[oy:oy + ih, ox:ox + iw]
        sub[m] = it[m]
        full = np.zeros((H, W), bool); full[oy:oy + ih, ox:ox + iw] = m
        units.append(dict(mask=full, parent=None))
        me = len(units) - 1
        for im in inner:
            f2 = np.zeros((H, W), bool); f2[oy:oy + ih, ox:ox + iw] = im
            units.append(dict(mask=f2, parent=me))
    return dict(img=img, units=units)


def bookshelf(rng):
    W, H = int(rng.uniform(140, 210)), int(rng.uniform(170, 240))
    img = _canvas(H, W)
    wood = _wood(rng); sw = int(rng.uniform(7, 11)); bt = int(rng.uniform(6, 9))
    nshelf = int(rng.integers(2, 4))
    frame = O._rect(img.shape, 2, 2 + sw, 2, H - 2) | O._rect(img.shape, W - 2 - sw, W - 2, 2, H - 2)
    ys = np.linspace(2, H - 2 - bt, nshelf + 1).astype(int)
    for y in ys:
        frame |= O._rect(img.shape, 2, W - 2, y, y + bt)
    img[frame] = wood.astype(np.uint8)
    back = O._rect(img.shape, 2 + sw, W - 2 - sw, 2, H - 2) & ~frame        # a darker back panel: the shelf is one object with it
    img[back] = np.clip(wood * 0.55, 0, 255).astype(np.uint8)
    O._outline(img, frame | back, wood)
    units = []
    for si in range(nshelf):
        top, bot = ys[si] + bt, ys[si + 1]
        x = 2 + sw + int(rng.uniform(0, 6))
        while True:
            bw = int(rng.uniform(8, 18)); bh = int((bot - top) * rng.uniform(0.6, 0.95))
            if x + bw > W - 2 - sw - 1 or rng.random() < 0.08:
                break
            m = O._rect(img.shape, x, x + bw, bot - bh, bot)
            c = O._col(rng)
            img[m] = c.astype(np.uint8)
            if rng.random() < 0.7:                                          # a label on the spine
                ly = int(bot - bh * rng.uniform(0.5, 0.8))
                img[O._rect(img.shape, x + 2, x + bw - 2, ly, ly + int(rng.uniform(4, 9))) & m] = O._col(rng).astype(np.uint8)
            img[O._rect(img.shape, x, x + 1, bot - bh, bot)] = np.clip(c * 0.6, 0, 255).astype(np.uint8)
            units.append(dict(mask=m, parent=None))
            x += bw + int(rng.choice([0, 0, 0, 1, 2]))
    return dict(img=img, units=units)


def cabinet(rng):
    W, H = int(rng.uniform(110, 180), ), int(rng.uniform(130, 220))
    img = _canvas(H, W)
    wood = _wood(rng); fr = int(rng.uniform(6, 10))
    body = O._rect(img.shape, 2, W - 2, 2, H - 2)
    img[body] = wood.astype(np.uint8)
    O._outline(img, body, wood)
    n = int(rng.integers(2, 5))
    ys = np.linspace(2 + fr, H - 2 - fr, n + 1)
    units = []
    front = _wood(rng) if rng.random() < 0.5 else wood * rng.uniform(0.8, 1.15)
    for k in range(n):
        y0, y1 = int(ys[k]) + 2, int(ys[k + 1]) - 2
        m = O._rect(img.shape, 2 + fr, W - 2 - fr, y0, y1)
        img[m] = np.clip(front, 0, 255).astype(np.uint8)
        img[O._rect(img.shape, 2 + fr, W - 2 - fr, y1 - 1, y1)] = np.clip(front * 0.5, 0, 255).astype(np.uint8)   # shadow under the front
        kn = np.zeros(img.shape[:2], np.uint8)
        cv2.circle(kn, (W // 2, (y0 + y1) // 2), int(rng.uniform(3, 6)), 1, -1)
        img[kn.astype(bool)] = np.clip(front * 0.45, 0, 255).astype(np.uint8)
        units.append(dict(mask=m, parent=None))
    return dict(img=img, units=units)


def fruit_bowl(rng):
    W, H = int(rng.uniform(140, 200)), int(rng.uniform(100, 140))
    img = _canvas(H, W)
    rim = int(H * rng.uniform(0.45, 0.6)); c = O._col(rng)
    bowl = np.zeros((H, W), np.uint8)
    cv2.ellipse(bowl, (W // 2, rim), (W // 2 - 4, H - rim - 4), 0, 0, 180, 1, -1)
    bowl = bowl.astype(bool)
    bowl |= O._rect(img.shape, 4, W - 4, rim - 4, rim + 2)
    O._shade(img, bowl, c, W / 2, W / 2)
    O._outline(img, bowl, c)
    units = []
    n = int(rng.integers(2, 5))
    xs = np.sort(rng.uniform(0.22, 0.78, n)) * W
    for x in xs:
        r = rng.uniform(13, 22)
        f = np.zeros((H, W), np.uint8)
        cv2.circle(f, (int(x), int(rim - r * rng.uniform(0.5, 1.0))), int(r), 1, -1)
        f = f.astype(bool) & ~bowl
        fc = O._col(rng, 40, 230)
        O._shade(img, f, fc, x, r)
        for u in units:                                                   # a later fruit hides part of an earlier one
            u["mask"] &= ~f
        units.append(dict(mask=f, parent=None))
    units = [u for u in units if u["mask"].sum() > 60]
    return dict(img=img, units=units)


def pencil_cup(rng):
    W, H = int(rng.uniform(90, 130)), int(rng.uniform(160, 220))
    img = _canvas(H, W)
    cup_top = int(H * rng.uniform(0.45, 0.6)); cw = rng.uniform(40, 60); c = O._col(rng)
    cup = O._poly(img.shape, [(W / 2 - cw / 2, cup_top), (W / 2 + cw / 2, cup_top), (W / 2 + cw / 2 - 3, H - 3), (W / 2 - cw / 2 + 3, H - 3)])
    units = []
    for k in range(int(rng.integers(2, 5))):
        ang = np.deg2rad(rng.uniform(-22, 22)); L = rng.uniform(0.5, 0.95) * cup_top + 20; pw = rng.uniform(10, 14)   # at least two patches wide
        base = np.array([W / 2 + rng.uniform(-cw / 3, cw / 3), cup_top + 15])
        d = np.array([np.sin(ang), -np.cos(ang)]); nrm = np.array([d[1], -d[0]])
        tip = base + d * L
        body = [base - nrm * pw / 2, base + nrm * pw / 2, tip + nrm * pw / 2, tip - nrm * pw / 2]
        m = O._poly(img.shape, body)
        pt = O._poly(img.shape, [tip + nrm * pw / 2, tip - nrm * pw / 2, tip + d * 8])
        pc = O._col(rng)
        img[m] = pc.astype(np.uint8); img[pt & ~m] = (150, 190, 220)
        mm = (m | pt) & ~cup
        for u in units:
            u["mask"] &= ~mm
        units.append(dict(mask=mm, parent=None))
    O._shade(img, cup, c, W / 2, cw / 2)
    O._outline(img, cup, c)
    for u in units:
        u["mask"] &= ~cup
    units = [u for u in units if u["mask"].sum() > 40]
    return dict(img=img, units=units)


def book_stack(rng):
    W, H = int(rng.uniform(110, 170)), int(rng.uniform(70, 120))
    img = _canvas(H, W)
    n = int(rng.integers(3, 6))
    hs = rng.uniform(10, 20, n); hs = hs / hs.sum() * (H - 6)
    y = H - 3
    units = []
    for k in range(n):
        bw = rng.uniform(0.6, 0.95) * (W - 8); x0 = rng.uniform(4, W - 4 - bw)
        m = O._rect(img.shape, x0, x0 + bw, y - hs[k], y)
        c = O._col(rng)
        img[m] = c.astype(np.uint8)
        img[O._rect(img.shape, x0 + bw - 4, x0 + bw, y - hs[k] + 2, y - 2) & m] = (235, 235, 228)     # page edges
        O._outline(img, m, c)
        units.append(dict(mask=m, parent=None))
        y -= hs[k]
    # the bottom book rests on the paper: it is the 'rest' of this scene, not a unit
    return dict(img=img, units=units[1:])


def chair(rng):
    W, H = int(rng.uniform(90, 130)), int(rng.uniform(150, 200))
    img = _canvas(H, W)
    wood = _wood(rng); seat_y = int(H * rng.uniform(0.5, 0.6)); lw = int(rng.uniform(6, 10))
    back = O._rect(img.shape, 6, 6 + lw, 4, seat_y) | O._rect(img.shape, 6, W - 30, 4, 4 + lw * 2)
    seat = O._rect(img.shape, 4, W - 4, seat_y, seat_y + lw)
    legs = O._rect(img.shape, 6, 6 + lw, seat_y + lw, H - 3) | O._rect(img.shape, W - 6 - lw, W - 6, seat_y + lw, H - 3)
    img[back | seat | legs] = wood.astype(np.uint8)
    if rng.random() < 0.5:                                                    # an upholstered seat in another colour
        img[seat] = O._col(rng).astype(np.uint8)
    O._outline(img, back | seat | legs, wood)
    return dict(img=img, units=[])


def mug(rng):
    W, H = 110, int(rng.uniform(80, 120))
    img = _canvas(H, W)
    bw = rng.uniform(45, 65); c = O._col(rng)
    body = O._rect(img.shape, 10, 10 + bw, 4, H - 4)
    O._shade(img, body, c, 10 + bw / 2, bw / 2)
    hm = np.zeros((H, W), np.uint8)
    cv2.ellipse(hm, (int(10 + bw), H // 2), (int(rng.uniform(14, 22)), int(H * 0.25)), 0, -90, 90, 1, int(rng.uniform(5, 9)))
    hm = hm.astype(bool) & ~body
    img[hm] = np.clip(c * rng.uniform(0.7, 1.0), 0, 255).astype(np.uint8)
    if rng.random() < 0.6:                                                    # a printed logo
        lg = np.zeros((H, W), np.uint8)
        cv2.circle(lg, (int(10 + bw / 2), H // 2), int(bw * 0.2), 1, -1)
        img[lg.astype(bool) & body] = O._col(rng).astype(np.uint8)
    O._outline(img, body | hm, c)
    return dict(img=img, units=[])


def desk_lamp(rng):
    W, H = int(rng.uniform(100, 140)), int(rng.uniform(150, 200))
    img = _canvas(H, W)
    metal, shade = O._col(rng), O._col(rng)
    base = O._rect(img.shape, 10, 60, H - 10, H - 3)
    a = np.array([35, H - 10]); b = a + np.array([rng.uniform(10, 40), -rng.uniform(60, 90)]); cpt = b + np.array([rng.uniform(25, 50), rng.uniform(5, 30)])
    arm = np.zeros((H, W), np.uint8)
    cv2.line(arm, tuple(int(v) for v in a), tuple(int(v) for v in b), 1, 12); cv2.line(arm, tuple(int(v) for v in b), tuple(int(v) for v in cpt), 1, 12)   # two patches wide
    head = O._poly(img.shape, [cpt + (-12, -6), cpt + (12, -6), cpt + (22, 20), cpt + (-22, 20)])
    m = base | arm.astype(bool)
    img[m] = metal.astype(np.uint8)
    img[head] = shade.astype(np.uint8)
    O._outline(img, m | head, metal)
    return dict(img=img, units=[])


def _whole(fn):
    def gen(rng):
        return dict(img=fn(rng)["img"], units=[])
    return gen


SINGLE = {k: _single(v) for k, v in O.SEPARABLE.items()}
ARRANGED = dict(table=table, bookshelf=bookshelf, cabinet=cabinet, fruit_bowl=fruit_bowl, pencil_cup=pencil_cup, book_stack=book_stack)
WHOLE = dict(hammer=_whole(O.hammer), screwdriver=_whole(O.screwdriver), vase=_whole(O.vase), goblet=_whole(O.goblet),
             chair=chair, mug=mug, desk_lamp=desk_lamp)
FAMILIES = dict(**SINGLE, **ARRANGED, **WHOLE)
GROUP = {**{k: "single" for k in SINGLE}, **{k: "arranged" for k in ARRANGED}, **{k: "whole" for k in WHOLE}}


def make(family, seed):
    sc = FAMILIES[family](np.random.default_rng(seed))
    sc["family"] = family
    obj = _objmask(sc["img"])
    for u in sc["units"]:
        u["mask"] = u["mask"] & obj
    sc["units"] = [u for u in sc["units"] if u["mask"].sum() >= 30]
    return sc


def partition(sc):
    """Finest ground-truth partition of the object pixels: label image (0 = paper), segments 1..k."""
    obj = _objmask(sc["img"])
    code = np.zeros(obj.shape, np.int64)
    for i, u in enumerate(sc["units"]):
        code |= (u["mask"].astype(np.int64) << i)
    lab = np.zeros(obj.shape, int)
    _, inv = np.unique(code[obj], return_inverse=True)
    lab[obj] = inv + 1
    return lab


def transform(sc, kind):
    """Changes at test time only: rotation, scale, colour, light, noise.  Applied to the picture and every unit mask alike."""
    img = sc["img"].copy()
    masks = [u["mask"].copy() for u in sc["units"]]
    h, w = img.shape[:2]
    if kind == "none":
        pass
    elif kind == "rot90":
        img = np.ascontiguousarray(np.rot90(img, 1)); masks = [np.ascontiguousarray(np.rot90(m, 1)) for m in masks]
    elif kind == "rot180":
        img = np.ascontiguousarray(img[::-1, ::-1]); masks = [np.ascontiguousarray(m[::-1, ::-1]) for m in masks]
    elif kind.startswith("scale"):
        k = float(kind[5:])
        size = (int(round(w * k)), int(round(h * k)))
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
        masks = [cv2.resize(m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool) for m in masks]
    elif kind == "recolour":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); obj = _objmask(img)
        hsv[..., 0][obj] = ((hsv[..., 0][obj].astype(int) + 60) % 180).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    elif kind == "dark":
        obj = _objmask(img); img[obj] = (img[obj] * 0.6).astype(np.uint8)
    elif kind == "noise":
        obj = _objmask(img)
        n = np.random.default_rng(0).normal(0, 12, img.shape)
        img = np.where(obj[..., None], np.clip(img + n, 0, 234), img).astype(np.uint8)
    else:
        raise ValueError(kind)
    out = dict(img=img, units=[dict(mask=m, parent=u["parent"]) for m, u in zip(masks, sc["units"])], family=sc["family"])
    return out
