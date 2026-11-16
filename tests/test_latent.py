import numpy as np

from objectlearn.latent import LatentMemory
from objectlearn.traits import T, TRAIT_NAMES, candidates


def bottle_patches(s=5.0, neck=5, cap=7, cap_rows=3, neck_rows=2, body=14, body_rows=16, cx=40.0, step=0.0):
    pts = []
    for r in range(cap_rows):
        for q in range(cap):
            pts.append((cx + (q - (cap - 1) / 2) * s, r * s))
    for r in range(neck_rows):
        for q in range(neck):
            pts.append((cx + (q - (neck - 1) / 2) * s, (cap_rows + r) * s))
    for r in range(body_rows):
        for q in range(body):
            pts.append((cx + (q - (body - 1) / 2) * s, (cap_rows + neck_rows + r) * s))
    return np.array(pts, float) + step, cap_rows * s


def sig_for(pos, ycap, cap_val=80.0, body_val=160.0):
    L = np.where(pos[:, 1] < ycap - 1e-6, cap_val, body_val)
    return np.stack([L, 128 + 0 * L, 128 + 0 * L], 1)


def true_cut_candidate(pos, X, masks, meta, ycap):
    for i, m in enumerate(masks):
        if m.sum() == (pos[:, 1] < ycap).sum() and np.all(m == (pos[:, 1] < ycap)):
            return i
    return None


def test_traits_shape_and_finite():
    pos, yc = bottle_patches()
    X, masks, meta = candidates(pos, sig_for(pos, yc), 5.0)
    assert X.shape[1] == T == len(TRAIT_NAMES) and len(X) == len(masks) == len(meta) > 50
    assert np.isfinite(X).all()


def test_cap_cut_is_a_candidate_and_stands_out():
    pos, yc = bottle_patches()
    X, masks, meta = candidates(pos, sig_for(pos, yc), 5.0)
    i = true_cut_candidate(pos, X, masks, meta, yc)
    assert i is not None
    name = {n: k for k, n in enumerate(TRAIT_NAMES)}
    assert X[i, name["step_cut"]] > 1.0                        # a clear signal change at the cut
    assert X[i, name["cut_vs_widest"]] < -0.5                  # narrow compared with the widest place


def test_traits_ignore_position_signal_values_and_scale():
    pos, yc = bottle_patches()
    X0, _, _ = candidates(pos, sig_for(pos, yc), 5.0)
    X1, _, _ = candidates(pos + np.array([37.3, -12.1]), sig_for(pos, yc, 20.0, 90.0) * 1.0, 5.0)     # moved, different signal values, same change size
    X1b, _, _ = candidates(pos + np.array([37.3, -12.1]), sig_for(pos, yc) + 17.0, 5.0)                # signal offset
    X2, _, _ = candidates(pos * 2.0, sig_for(pos, yc), 10.0)                                           # everything twice as large
    assert np.allclose(X0, X1b, atol=1e-4)
    assert np.allclose(X0, X2, atol=1e-4)
    assert X1.shape == X0.shape


def test_part_is_always_the_smaller_side():
    pos, yc = bottle_patches()
    _, masks, _ = candidates(pos, sig_for(pos, yc), 5.0)
    assert (masks.sum(1) <= len(pos) / 2).all()


def test_memory_reads_behaviour_from_neighbours_and_abstains_far_away():
    rng = np.random.default_rng(0)
    A = rng.normal(0, 0.3, (60, 5)); B = rng.normal(0, 0.3, (60, 5)); B[:, 0] += 3.0
    m = LatentMemory(k_nn=10)
    m.add(np.vstack([A, B]), np.r_[np.zeros(60), np.ones(60)])
    q = m.query(np.vstack([A[:3] + 0.1, B[:3] + 0.1, np.full((1, 5), 30.0)]))
    assert (q["p"][:3] < 0.3).all() and (q["p"][3:6] > 0.7).all()
    assert q["statement"][3] == 1 and q["statement"][0] == 2
    assert q["statement"][6] == 0                               # far from everything: unknown
    rel = m.relevance()
    assert rel.argmax() == 0                                    # the trait that co-varies with the behaviour counts most


def test_memory_ignores_unobserved_behaviour_and_roundtrips(tmp_path):
    m = LatentMemory(k_nn=5)
    X = np.random.default_rng(1).normal(size=(20, 4))
    y = np.r_[np.ones(10), np.full(10, np.nan)]
    m.add(X, y)
    assert m.query(X[:2])["support"].max() > 0
    m.save(tmp_path / "m.npz")
    m2 = LatentMemory(k_nn=5); m2.load(tmp_path / "m.npz")
    assert len(m2) == 20 and np.allclose(m2.query(X[:3])["p"], m.query(X[:3])["p"])


def test_app_learns_from_events_and_reads_a_new_picture(tmp_path):
    from experiments.bottles import make_bottle
    from objectlearn.paint import PaintApp
    path = str(tmp_path / "latent.npz")
    app = PaintApp(s=5, cap=1100, memory=path, mode="patch", refresh_every=999999, latent=True)
    t = [0.0]

    def tick(*a):
        t[0] += 0.1
        return app.tick(list(a), t[0], render=False)

    def put(i):
        img, yc = make_bottle(np.random.default_rng(i))
        app.clear()
        pid = app.world.add_image(np.ascontiguousarray(img), np.full(img.shape[:2], 255, np.uint8), x=130)
        ys, xs = np.nonzero(app.world.pieces[pid].mask)
        return int(app.world.H / 2 - img.shape[0] / 2 + yc), xs.min(), xs.max(), ys.min(), ys.max()

    for i in range(1, 5):
        ycut, xa, xb, ytop, ybot = put(i)
        for _ in range(3):
            tick()
        app.world.cut([[xa - 4, ycut], [xb + 4, ycut]], mark=False)
        for _ in range(4):
            tick()
        tick({"t": "grab", "x": (xa + xb) / 2, "y": (ytop + ycut) / 2})
        for _ in range(16):
            tick({"t": "drag", "dx": 5, "dy": -1})
        tick({"t": "release"})
        for _ in range(12):
            r = tick()
        assert len(r["formal"]) == 2                             # the definition: the cap came away
    lat = app.session.latent
    assert len(lat) > 50 and (lat.Y[:, 0] == 1).sum() >= 4
    assert (lat.Y[:, 0] == 0).sum() > 0
    # the memory was saved and the maths panel can explain a cut
    ycut, xa, xb, ytop, ybot = put(50)
    for _ in range(5):
        tick()
    app.session._lat_due = True
    r = tick()
    assert len(r["formal"]) == 1                                  # nothing has moved: one object by the definition
    e = app.explain((xa + xb) / 2, ycut)
    from objectlearn.regions import T as RT
    assert e["found"] and e["latent"] and len(e["traits"]) == RT and e["memory"] == len(lat)
    app.session.save()
    mem2 = LatentMemory(); mem2.load(path)
    assert len(mem2) == len(lat)
