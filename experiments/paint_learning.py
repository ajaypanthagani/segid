"""How many painted examples does the model need before it separates a fresh painted-and-cut object before any move?

Each training example: paint a block (random size, colour, place), cut it, drag one half clear, hold still.
After each example the memory is probed: a new block is painted and cut but never moved.  Reported: of the seams the
model calls distinct, how many are real (different pieces), and what fraction of real seams it finds.
"""
import sys
import warnings
import numpy as np

from objectlearn.paint import PaintApp

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
app = PaintApp()
BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 16.0
KNN = int(sys.argv[2]) if len(sys.argv) > 2 else 24
app.session.cfg.event_budget = BUDGET
app.session.cfg.k_nn = KNN
N = int(sys.argv[3]) if len(sys.argv) > 3 else 12
T = [0.0]


def tick(acts=()):
    o = app.tick(list(acts), T[0])
    T[0] += 0.12
    return o


def block():
    w, h = int(rng.integers(50, 90)), int(rng.integers(36, 60))
    x0, y0 = int(rng.integers(20, 320 - w - 20)), int(rng.integers(30, 240 - h - 30))
    col = [int(c) for c in rng.integers(40, 230, 3)]
    pts = []
    y = y0
    while y <= y0 + h:
        pts += [[x0, y], [x0 + w, y]]
        y += 6
    cx = x0 + int(w * rng.uniform(0.35, 0.65))
    return dict(t="stroke", pts=pts, color=col, r=6), dict(t="cut", pts=[[cx, y0 - 12], [cx, y0 + h + 12]]), (cx, y0, w, h, x0)


def paint_and_cut():
    s, c, g = block()
    tick([s]); tick(); tick([c]); tick(); tick()
    return g


def train_example():
    app.clear()
    tick(); tick()
    cx, y0, w, h, x0 = paint_and_cut()
    gx = cx + 6
    dx = (w + 30) if gx + w + 30 < 310 else -(w + 30)
    tick([dict(t="grab", x=gx, y=y0 + h // 2)])
    for _ in range(4):
        tick([dict(t="drag", dx=dx / 4, dy=0)])
    tick([dict(t="release")])
    for _ in range(14):
        tick()


def probe(n=6):
    tp = fp = real = 0
    shown = 0
    saved = app.session.agent.learn
    app.session.agent.learn = False
    for _ in range(n):
        app.clear()
        tick(); tick()
        paint_and_cut()
        for _ in range(4):
            o = tick()
        ag, w = app.session.agent, app.world
        ag.refresh(w.frame()[0])
        ii, jj, rows = ag.interfaces()
        n_w = w.n
        pp = w.p_piece[:n_w]
        m = (ii < n_w) & (jj < n_w)
        ii, jj, rows = ii[m], jj[m], rows[m]
        truth = pp[ii] != pp[jj]
        st = ag.stat_now[rows]
        tp += int(((st == 1) & truth).sum()); fp += int(((st == 1) & ~truth).sum()); real += int(truth.sum())
        shown += sum(not x["background"] for x in o["predicted"])
    app.session.agent.learn = saved
    return tp, fp, real, shown / n


print("examples | predicted-distinct seams: true / false | real seams found | predicted objects shown per probe")
print(f"{0:8d} | {'-':>10}", flush=True)
print(f'event_budget={BUDGET} k_nn={KNN}')
for k in range(1, N + 1):
    train_example()
    # restate current memory on the probe scenes: the probe scenes are created after training, so statements use it
    tp, fp, real, shown = probe()
    print(f"{k:8d} | {tp:5d} / {fp:3d}           | {tp / max(real, 1):.2f}             | {shown:.1f}", flush=True)
