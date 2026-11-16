"""Capped objects that are not bottles: jar (wide flat lid), pen (same-colour cap), tube (flat screw cap, wider body), spray can (nozzle)."""
import cv2, numpy as np

def _shade(img, mask, col, cx, half):
    xs = np.arange(img.shape[1])[None, :].repeat(img.shape[0], 0)
    t = np.clip((xs - cx) / (half + 1e-6), -1, 1)
    sh = 0.7 + 0.3 * np.sqrt(np.clip(1 - t ** 2, 0, 1))
    img[mask] = np.clip(np.array(col, float)[None] * sh[mask][:, None], 0, 255).astype(np.uint8)

def rect(shape, x0, x1, y0, y1):
    m = np.zeros(shape, bool); m[int(y0):int(y1), int(x0):int(x1)] = True; return m

def jar(rng):
    H, W = 150, 130; img = np.full((H, W, 3), 255, np.uint8); cx = W / 2
    bw = rng.uniform(60, 85); lw = bw * rng.uniform(0.95, 1.05); lh = rng.uniform(14, 22)
    body = rng.integers(60, 230, 3); lid = rng.integers(30, 200, 3)
    _shade(img, rect((H, W), cx - bw / 2, cx + bw / 2, 6 + lh, H - 6), body, cx, bw / 2)
    img[rect((H, W), cx - lw / 2, cx + lw / 2, 6, 6 + lh)] = lid
    return img, 6 + lh

def pen(rng):
    H, W = 190, 130; img = np.full((H, W, 3), 255, np.uint8); cx = W / 2
    bw = rng.uniform(14, 20); c = rng.integers(40, 230, 3); ch = rng.uniform(30, 48)
    img[rect((H, W), cx - bw / 2, cx + bw / 2, 6, H - 20)] = c               # same colour all along
    img[rect((H, W), cx - bw / 2 - 2, cx + bw / 2 + 2, 6, 6 + ch)] = np.clip(c * 0.9, 0, 255).astype(np.uint8)
    img[rect((H, W), cx - 1, cx + 1, 10, 6 + ch - 4)] = np.clip(c * 0.6, 0, 255).astype(np.uint8)   # clip
    tip = np.array([[cx - bw / 2, H - 20], [cx + bw / 2, H - 20], [cx, H - 6]], np.int32)
    cv2.fillPoly(img, [tip], tuple(int(v) for v in c * 0.5))
    return img, 6 + ch

def tube(rng):
    H, W = 170, 130; img = np.full((H, W, 3), 255, np.uint8); cx = W / 2
    bw = rng.uniform(40, 58); cw = bw * rng.uniform(0.35, 0.5); ch = rng.uniform(10, 16)
    body = rng.integers(60, 240, 3); cap = rng.integers(30, 220, 3)
    pts = np.array([[cx - cw / 2, 6 + ch], [cx + cw / 2, 6 + ch], [cx + bw / 2, 40], [cx + bw / 2, H - 6], [cx - bw / 2, H - 6], [cx - bw / 2, 40]], np.int32)
    m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [pts], 1); _shade(img, m.astype(bool), body, cx, bw / 2)
    img[rect((H, W), cx - cw / 2 * 1.15, cx + cw / 2 * 1.15, 6, 6 + ch)] = cap
    return img, 6 + ch

def spray(rng):
    H, W = 190, 130; img = np.full((H, W, 3), 255, np.uint8); cx = W / 2
    bw = rng.uniform(36, 50); body = rng.integers(60, 230, 3); cap = rng.integers(30, 220, 3)
    nz = bw * 0.28; nh = rng.uniform(8, 12); capw = bw * 0.7; caph = rng.uniform(26, 36)
    _shade(img, rect((H, W), cx - bw / 2, cx + bw / 2, 6 + nh + caph, H - 6), body, cx, bw / 2)
    img[rect((H, W), cx - capw / 2, cx + capw / 2, 6 + nh, 6 + nh + caph)] = cap
    img[rect((H, W), cx - nz / 2, cx + nz / 2 + 6, 6, 6 + nh)] = np.clip(np.array(cap) * 0.7, 0, 255).astype(np.uint8)
    return img, 6 + nh + caph

if __name__ == "__main__":
    for name, f, seeds in [("jar", jar, (1, 2)), ("pen", pen, (1, 2)), ("tube", tube, (1, 2)), ("spray", spray, (1, 2))]:
        for s in seeds:
            img, y = f(np.random.default_rng(s))
            cv2.imwrite(f"test_bottles/other_caps/{name}_{s}.png", cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
