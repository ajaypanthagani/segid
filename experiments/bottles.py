"""A procedural bottle dataset: random sizes, shapes, colours, cap styles, ribs and labels, on a white background.
Returns the picture and where the cap ends (so results can be scored); the model is never given either."""
import cv2
import numpy as np

TRAIN_BODY = [(200, 140, 70), (110, 170, 90), (205, 200, 195), (60, 130, 190), (170, 120, 100)]
TRAIN_CAP = [(180, 60, 40), (60, 60, 200), (40, 40, 40), (70, 150, 70), (200, 200, 200)]
HARD_BODY = [(190, 110, 200), (160, 170, 60), (80, 90, 220), (120, 120, 60)]
HARD_CAP = [(40, 200, 230), (190, 70, 150), (50, 130, 240)]


LAST = {}


def make_bottle(rng, hard=False):
    H = int(rng.uniform(150, 190)); W = 130
    bw = rng.uniform(28, 95) if hard else rng.uniform(40, 78)
    neck_w = bw * rng.uniform(0.3, 0.6)
    cap_w = neck_w * (rng.uniform(1.0, 1.4) if hard else rng.uniform(1.05, 1.3))
    cap_h = rng.uniform(8, 16)
    neck_h = rng.uniform(5, 14)
    sh = rng.uniform(0.08, 0.25) * H
    cx = W / 2
    top, ycap = 4.0, 4.0 + cap_h
    yneck, ybody = ycap + neck_h, ycap + neck_h + sh
    ybot = H - 4.0
    body = np.array((HARD_BODY if hard else TRAIN_BODY)[rng.integers(len(HARD_BODY if hard else TRAIN_BODY))], float)
    capc = np.array((HARD_CAP if hard else TRAIN_CAP)[rng.integers(len(HARD_CAP if hard else TRAIN_CAP))], float)
    img = np.full((H, W, 3), 255, np.uint8)

    # silhouette of the bottle below the cap: neck, shoulder, body, rounded bottom
    t = np.linspace(0, 1, 14)
    sh_x = neck_w / 2 + (bw / 2 - neck_w / 2) * (1 - np.cos(t * np.pi / 2))
    sh_y = yneck + (ybody - yneck) * t
    left = [(cx - neck_w / 2, ycap), (cx - neck_w / 2, yneck)] + [(cx - x, y) for x, y in zip(sh_x, sh_y)] + [(cx - bw / 2, ybot - 8), (cx - bw / 2 + 6, ybot)]
    pts = np.array(left + [(2 * cx - x, y) for x, y in left[::-1]], np.float32)
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    m = m.astype(bool)
    xs = np.arange(W)[None, :].repeat(H, 0)
    tt = np.clip((xs - cx) / (bw / 2 + 1e-6), -1, 1)
    shade = 0.68 + 0.32 * np.sqrt(np.clip(1 - tt ** 2, 0, 1))
    img[m] = np.clip(body[None, :] * shade[m][:, None], 0, 255).astype(np.uint8)
    if rng.random() < 0.6:                                     # label band
        y0, y1 = int(ybody + 0.18 * (ybot - ybody)), int(ybody + 0.5 * (ybot - ybody))
        lab = m.copy(); lab[:y0] = False; lab[y1:] = False
        lc = rng.integers(60, 255, 3)
        img[lab] = (0.6 * img[lab] + 0.4 * lc).astype(np.uint8)
    for k in range(int(rng.integers(0, 9))):                   # ribs
        y = int(rng.uniform(ybody + 6, ybot - 10))
        r = m.copy(); r[:y] = False; r[y + 1:] = False
        img[r] = np.clip(img[r] * 0.8, 0, 255).astype(np.uint8)
        r2 = m.copy(); r2[:y + 1] = False; r2[y + 3:] = False
        img[r2] = np.clip(img[r2] * 1.15 + 10, 0, 255).astype(np.uint8)
    cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnt, -1, tuple(float(v) for v in body * 0.55), 1)

    # cap (drawn on top), with a ring under it in most bottles
    x0, x1 = int(round(cx - cap_w / 2)), int(round(cx + cap_w / 2))
    y0, y1 = int(top), int(round(ycap))
    cap = np.zeros((H, W), bool); cap[y0:y1, x0:x1] = True
    img[cap] = capc.astype(np.uint8)
    for x in range(x0 + 2, x1 - 1, 3):                         # ridges
        img[y0 + 1:y1 - 1, x] = np.clip(capc * 0.8, 0, 255).astype(np.uint8)
    cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), tuple(float(v) for v in capc * 0.5), 1)
    ring = bool(rng.random() < (0.3 if hard else 0.85))
    if ring:
        img[y1:y1 + 2, int(cx - neck_w / 2):int(cx + neck_w / 2)] = np.clip(capc * 0.6, 0, 255).astype(np.uint8)
    LAST.update(ring=ring, flush=float(cap_w / neck_w), cap_h=float(cap_h), neck_h=float(neck_h), bw=float(bw), contrast=float(np.linalg.norm(capc - body)), sh=float(sh))
    return img, float(ycap)
