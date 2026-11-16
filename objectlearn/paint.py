"""A paint screen the model can watch.

You paint objects, cut them into parts, and drag parts around.  The model is given what a perfect tracker would give
it for the things that are drawn, and nothing else: a set of small sensing patches that keep their identity as the
drawn things move, each patch's position and whether it is currently visible, and the rendered image.  The blank
canvas has no patches, so it is never an object.  The model is not told what exists, how you cut, or what is an object.
Objects are what touches, and a part of an object becomes its own object only when it is seen to move apart.

Each drawn thing, and each part you cut from it, is a rigid piece: a pixel mask and colours in its own frame plus an
integer offset.  Patches live on a grid in a piece's frame, so a piece carries its patches with it.  A patch is hidden
while another piece covers it.
"""
import base64

import cv2
import numpy as np
from scipy import ndimage

from .sim import Frame

PAPER = (236, 242, 244)          # BGR
MIN_PIECE_PIXELS = 30


def shift(a, dx, dy):
    """out[y + dy, x + dx] = a[y, x]; zero fill."""
    H, W = a.shape[:2]
    out = np.zeros_like(a)
    if abs(dx) >= W or abs(dy) >= H:
        return out
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[yd, xd] = a[ys, xs]
    return out


class Piece:
    def __init__(self, shape):
        H, W = shape
        self.mask = np.zeros((H, W), bool)
        self.rgb = np.zeros((H, W, 3), np.uint8)
        self.off = np.zeros(2, int)


class PaintWorld:
    def __init__(self, W=320, H=240, s=10, cap=2000):
        self.W, self.H, self.s, self.cap = W, H, s, cap
        self.gw, self.gh = W // s, H // s
        self.pieces, self.order = {}, []          # z order, bottom to top (the paper is not a piece)
        self.next_id, self.active, self.selected = 1, None, None
        self.n = 0                                 # patches exist only where something has been drawn
        self.p_piece = np.zeros(cap, int)
        self.p_local = np.zeros((cap, 2), float)

    # ----------------------------------------------------------------- helpers
    def _off(self, pid):
        return self.pieces[pid].off if pid in self.pieces else np.zeros(2, int)

    def _to_local(self, m, off):
        return shift(m, -int(off[0]), -int(off[1]))

    def _to_canvas(self, m, off):
        return shift(m, int(off[0]), int(off[1]))

    def _brush(self, pts, r):
        m = np.zeros((self.H, self.W), np.uint8)
        p = np.round(np.asarray(pts, float)).astype(np.int32).reshape(-1, 1, 2)
        if len(p) > 1:
            cv2.polylines(m, [p], False, 1, thickness=2 * r + 1)
        for q in p:
            cv2.circle(m, (int(q[0, 0]), int(q[0, 1])), r, 1, -1)
        return m.astype(bool)

    def render(self):
        canvas = np.empty((self.H, self.W, 3), np.uint8)
        canvas[:] = PAPER
        owner = np.zeros((self.H, self.W), np.int32)
        for pid in self.order:
            pc = self.pieces[pid]
            m = self._to_canvas(pc.mask, pc.off)
            canvas[m] = self._to_canvas(pc.rgb, pc.off)[m]
            owner[m] = pid
        return canvas, owner

    # ----------------------------------------------------------------- actions
    def new_object(self):
        pid = self.next_id
        self.next_id += 1
        self.pieces[pid] = Piece((self.H, self.W))
        self.order.append(pid)
        self.active = pid
        return pid

    def _front(self, pid):
        if pid in self.order:
            self.order.remove(pid)
            self.order.append(pid)

    def _drop_empty(self):
        for pid in [p for p in self.order if not self.pieces[p].mask.any()]:
            self.order.remove(pid)
            if self.active == pid:
                self.active = None
            if self.selected == pid:
                self.selected = None

    def stroke(self, pts, color_bgr, r, erase=False):
        mc = self._brush(pts, int(np.clip(r, 1, 14)))
        if erase:
            for pc in self.pieces.values():
                pc.mask &= ~self._to_local(mc, pc.off)
            self._drop_empty()
            return
        if self.active not in self.pieces or self.active not in self.order:
            self.new_object()
        pc = self.pieces[self.active]
        for pid, q in self.pieces.items():
            if pid != self.active:
                q.mask &= ~self._to_local(mc, q.off)
        ml = self._to_local(mc, pc.off)
        pc.mask |= ml
        pc.rgb[ml] = color_bgr
        self._front(self.active)
        self._drop_empty()
        self._seed(self.active)

    def add_image(self, bgr, alpha, x=None, y=None):
        """Place an image as a new object.  Pixels with alpha < 128 are left as paper.  It is scaled to fit and centred
        unless a position is given.  Cut and move work on it like on anything painted."""
        if (alpha >= 128).all():            # an opaque picture: paper-coloured pixels connected to its border are paper
            near = (bgr.min(axis=2) >= 235).astype(np.uint8)
            n, lab = cv2.connectedComponents(near, connectivity=4)
            edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
            fg = (~np.isin(lab, edge[edge > 0])).astype(np.uint8)
            k = max(3, (min(fg.shape) // 40) | 1)               # bridge pale stripes that reach the outline, fill holes
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            fg[:] = 0
            if cnts:
                cv2.drawContours(fg, [max(cnts, key=cv2.contourArea)], -1, 1, -1)
            alpha = (fg * 255).astype(np.uint8)
        h, w = bgr.shape[:2]
        k = min(1.0, 0.8 * self.W / w, 0.8 * self.H / h)
        if k < 1.0:
            bgr = cv2.resize(bgr, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA)
            alpha = cv2.resize(alpha, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_AREA)
            h, w = bgr.shape[:2]
        x0 = int(self.W / 2 - w / 2) if x is None else int(x - w / 2)
        y0 = int(self.H / 2 - h / 2) if y is None else int(y - h / 2)
        x0, y0 = int(np.clip(x0, 0, self.W - w)), int(np.clip(y0, 0, self.H - h))
        m = np.zeros((self.H, self.W), bool)
        m[y0:y0 + h, x0:x0 + w] = alpha >= 128
        if not m.any():
            return None
        pid = self.new_object()
        pc = self.pieces[pid]
        for q in self.pieces.values():
            if q is not pc:
                q.mask &= ~m
        pc.mask = m
        pc.rgb[y0:y0 + h, x0:x0 + w] = bgr
        self._drop_empty()
        self._seed(pid)
        return pid

    def _seed(self, pid):
        """New patches for painted cells of this piece that do not have one yet."""
        pc = self.pieces[pid]
        s = self.s
        cx, cy = np.meshgrid((np.arange(self.gw) + .5) * s, (np.arange(self.gh) + .5) * s)
        inmask = pc.mask[cy.astype(int), cx.astype(int)]
        have = np.zeros((self.gh, self.gw), bool)
        sel = self.p_piece[:self.n] == pid
        if sel.any():
            li = (self.p_local[:self.n][sel] // s).astype(int)
            have[np.clip(li[:, 1], 0, self.gh - 1), np.clip(li[:, 0], 0, self.gw - 1)] = True
        ys, xs = np.nonzero(inmask & ~have)
        k = min(len(ys), self.cap - self.n)
        if k <= 0:
            return
        a = self.n
        self.p_piece[a:a + k] = pid
        self.p_local[a:a + k, 0] = (xs[:k] + .5) * s
        self.p_local[a:a + k, 1] = (ys[:k] + .5) * s
        self.n += k

    def cut(self, pts, mark=True):
        """Draw a line across a piece.  If it separates the piece, the pieces become independent; the seam is drawn as a
        darker line.  Nothing moves, so the model sees no change until you drag a part away."""
        pts = np.asarray(pts, float)
        if len(pts) < 2:
            return False
        # the piece the line crosses most (ties go to the one on top)
        canvas_line = self._brush(pts, 1)
        best, pid = 0, 0
        for cand in self.order:
            q = self.pieces[cand]
            hit = int((canvas_line & self._to_canvas(q.mask, q.off)).sum())
            if hit >= best and hit > 0:
                best, pid = hit, cand
        if pid == 0:
            return False
        pc = self.pieces[pid]
        lp = np.round(pts - pc.off).astype(np.int32).reshape(-1, 1, 2)
        line = np.zeros((self.H, self.W), np.uint8)
        cv2.polylines(line, [lp], False, 1, thickness=3)
        line = line.astype(bool)
        n, lab = cv2.connectedComponents((pc.mask & ~line).astype(np.uint8), connectivity=4)
        areas = np.bincount(lab.ravel(), minlength=n)
        keep = [i for i in range(1, n) if areas[i] >= MIN_PIECE_PIXELS]
        seam = line & pc.mask
        if mark:
            pc.rgb[seam] = (pc.rgb[seam] * 0.5).astype(np.uint8)
        if len(keep) < 2:
            return False
        labk = np.zeros_like(lab)
        for j, i in enumerate(keep, 1):
            labk[lab == i] = j
        _, (iy, ix) = ndimage.distance_transform_edt(labk == 0, return_indices=True)
        full = labk[iy, ix]
        full[~pc.mask] = 0
        sizes = [(full == j).sum() for j in range(1, len(keep) + 1)]
        ranked = [int(j0) + 1 for j0 in np.argsort(sizes)[::-1]]
        ids = {ranked[0]: pid}
        at = self.order.index(pid)
        for j in ranked[1:]:
            q = Piece((self.H, self.W))
            nid = self.next_id
            self.next_id += 1
            q.off = pc.off.copy()
            m = full == j
            q.mask, q.rgb[m] = m.copy(), pc.rgb[m]
            self.pieces[nid] = q
            at += 1
            self.order.insert(at, nid)
            ids[j] = nid
        pc.mask = full == ranked[0]
        sel = np.flatnonzero(self.p_piece[:self.n] == pid)
        ly = np.clip(np.rint(self.p_local[sel, 1]).astype(int), 0, self.H - 1)
        lx = np.clip(np.rint(self.p_local[sel, 0]).astype(int), 0, self.W - 1)
        labs = full[ly, lx]
        for j, nid in ids.items():
            if nid != pid:
                self.p_piece[sel[labs == j]] = nid
        return True

    def grab(self, x, y):
        _, owner = self.render()
        x, y = int(round(x)), int(round(y))
        pid = int(owner[y, x]) if (0 <= x < self.W and 0 <= y < self.H) else 0
        self.selected = pid if pid else None
        if pid:
            self._front(pid)
        return self.selected

    def drag(self, dx, dy):
        if self.selected in self.pieces:
            self.pieces[self.selected].off += np.array([int(round(dx)), int(round(dy))])

    def release(self):
        self.selected = None

    # ------------------------------------------------------------------- frame
    def detach(self, canvas_mask):
        """Free the given pixels (on the canvas) from the piece they belong to, as a cut along the part's whole outline would.
        Used by scripted experiments to lift a book out of a shelf or a cup off a table.  Returns the new piece's id."""
        best, pid = 0, 0
        for cand in self.order:
            q = self.pieces[cand]
            hit = int((canvas_mask & self._to_canvas(q.mask, q.off)).sum())
            if hit > best:
                best, pid = hit, cand
        if pid == 0:
            return None
        pc = self.pieces[pid]
        m = self._to_local(canvas_mask, pc.off) & pc.mask
        if m.sum() < MIN_PIECE_PIXELS or (pc.mask & ~m).sum() < MIN_PIECE_PIXELS:
            return None
        q = Piece((self.H, self.W))
        nid = self.next_id
        self.next_id += 1
        q.off = pc.off.copy()
        q.mask, q.rgb[m] = m.copy(), pc.rgb[m]
        pc.mask = pc.mask & ~m
        self.pieces[nid] = q
        self.order.insert(self.order.index(pid) + 1, nid)
        sel = np.flatnonzero(self.p_piece[:self.n] == pid)
        ly = np.clip(np.rint(self.p_local[sel, 1]).astype(int), 0, self.H - 1)
        lx = np.clip(np.rint(self.p_local[sel, 0]).astype(int), 0, self.W - 1)
        self.p_piece[sel[m[ly, lx]]] = nid
        return nid

    def frame(self):
        canvas, owner = self.render()
        n = self.n
        offs = np.zeros((self.next_id + 1, 2))
        for pid, pc in self.pieces.items():
            offs[pid] = pc.off
        q = self.p_local[:n] + offs[self.p_piece[:n]]
        ix, iy = np.rint(q[:, 0]).astype(int), np.rint(q[:, 1]).astype(int)
        inside = (ix >= 0) & (ix < self.W) & (iy >= 0) & (iy < self.H)
        top = owner[np.clip(iy, 0, self.H - 1), np.clip(ix, 0, self.W - 1)]
        vis = np.zeros(self.cap, bool)
        vis[:n] = inside & (top == self.p_piece[:n])
        pos = np.zeros((self.cap, 3), np.float32)
        pos[:n, :2] = q
        rgb = np.zeros((self.cap, 3), np.uint8)                # the colour each patch sampled from its own piece
        for pid, pc in self.pieces.items():
            sel = np.flatnonzero(self.p_piece[:n] == pid)
            if sel.size:
                lx = np.clip(np.rint(self.p_local[sel, 0]).astype(int), 0, self.W - 1)
                ly = np.clip(np.rint(self.p_local[sel, 1]).astype(int), 0, self.H - 1)
                rgb[sel] = pc.rgb[ly, lx]
        return Frame(pos=pos, nrm=np.zeros((self.cap, 3), np.float32), col=np.zeros(self.cap, np.float32),
                     vis=vis, img=canvas, patch_rgb=rgb), canvas


class PaintApp:
    """The world plus the model.  One call = one frame of the video the model sees."""

    def __init__(self, memory=None, W=320, H=240, s=10, cap=2000, mode="image", refresh_every=3, latent=False):
        from .session import Session
        self.mode, self.refresh_every = mode, refresh_every
        self.W, self.H, self.s, self.cap = W, H, s, cap
        self.world = PaintWorld(W, H, s, cap)
        self.session = Session(patches=cap, s=float(s), eps=1.3 * s, delta=4.0 * s, memory=None if latent else memory, mode=mode, refresh_every=refresh_every,
                               latent=bool(latent), latent_memory=memory if latent else None)

    def apply(self, a):
        w, t = self.world, a.get("t")
        if t == "stroke":
            c = a.get("color", [30, 30, 30])
            w.stroke(a["pts"], (int(c[2]), int(c[1]), int(c[0])), int(a.get("r", 5)), bool(a.get("erase")))
        elif t == "image":
            raw = base64.b64decode(a["data"].split(",")[-1])
            im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
            if im is None:
                return
            if im.ndim == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
            elif im.shape[2] == 3:
                im = np.dstack([im, np.full(im.shape[:2], 255, np.uint8)])
            w.add_image(np.ascontiguousarray(im[:, :, :3]), np.ascontiguousarray(im[:, :, 3]))
        elif t == "newobject":
            w.new_object()
        elif t == "cut":
            w.cut(a["pts"])
        elif t == "grab":
            w.grab(a["x"], a["y"])
        elif t == "drag":
            w.drag(a["dx"], a["dy"])
        elif t == "release":
            w.release()

    def explain(self, x, y):
        return self.session.explain(float(x), float(y))

    def clear(self):
        """Blank paper.  What the model learned is kept."""
        self.world = PaintWorld(self.W, self.H, self.s, self.cap)
        self.session.reset_scene()

    def forget(self):
        self.world = PaintWorld(self.W, self.H, self.s, self.cap)
        self.session.forget()

    def tick(self, actions, t, render=True):
        for a in actions:
            try:
                if a.get("t") == "clear":
                    self.clear()
                elif a.get("t") == "forget":
                    self.forget()
                else:
                    self.apply(a)
            except (KeyError, ValueError, TypeError, IndexError):
                continue
        frame, canvas = self.world.frame()
        res = self.session.ingest(frame, t)
        if not render:
            return dict(formal=res.formal_objects, predicted=res.predicted_objects, stats=res.stats, events=res.events, seams=res.seams,
                        pts=[[int(x), int(y), int(f), int(p)] for (x, y), f, p in zip(res.xy, res.formal, res.predicted)])
        ok, png = cv2.imencode(".png", canvas)
        pts = [[int(x), int(y), int(f), int(p)] for (x, y), f, p in zip(res.xy, res.formal, res.predicted)]
        return dict(w=self.W, h=self.H, image="data:image/png;base64," + base64.b64encode(png.tobytes()).decode(),
                    pts=pts, formal=res.formal_objects, predicted=res.predicted_objects, stats=res.stats,
                    events=res.events, seams=res.seams, active=self.world.active, selected=self.world.selected,
                    pieces=len(self.world.order))
