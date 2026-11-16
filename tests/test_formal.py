"""Unit tests for the formal layer: the section 7 cases and the section 6 invariants."""
import numpy as np
import pytest

from objectlearn.formal import FormalLayer

EPS, K = 1.0, 2


def run(n, frames, pi_fn=None, eps=EPS, k=K):
    """frames: list of (positions (n,2), visible (n,)) ; returns layer."""
    L = FormalLayer(n, eps, k)
    for pos, vis in frames:
        p3 = np.zeros((n, 3))
        p3[:, :2] = pos
        L.step(p3, vis, pi_fn)
    return L


def track(n, spec):
    """spec: list of dicts index -> (x, y) | None (hidden) ; unspecified patches stay unseen."""
    frames = []
    for fr in spec:
        pos = np.zeros((n, 2))
        vis = np.zeros(n, bool)
        for i, v in fr.items():
            if v is not None:
                pos[i] = v
                vis[i] = True
        frames.append((pos, vis))
    return frames


def lerp(a, b, m):
    return [tuple(np.array(a) + (np.array(b) - np.array(a)) * s) for s in np.linspace(0, 1, m)]


def test_static_table_and_floor_one_object():
    floor = {i: (float(i), 0.0) for i in range(6)}
    table = {6: (2.0, 0.8), 7: (3.0, 0.8)}
    fr = [{**floor, **table}] * 6
    L = run(8, track(8, fr))
    assert L.n_objects() == 1


def test_table_lifted_becomes_separate():
    floor = {i: (float(i), 0.0) for i in range(6)}
    frames = []
    for y in list(np.linspace(0.8, 3.0, 6)) + [3.0] * 3:
        frames.append({**floor, 6: (2.0, y), 7: (3.0, y)})
    L = run(8, track(8, frames))
    assert L.n_objects() == 2
    assert L.labels[6] == L.labels[7] != L.labels[0]


def test_mug_first_seen_on_table_then_lifted_then_put_back():
    table = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0)}
    ys = [0.8] * 3 + list(np.linspace(0.8, 4.0, 5)) + [4.0] * 3 + list(np.linspace(4.0, 0.8, 5)) + [0.8] * 4
    frames = [{**table, 3: (1.0, y)} for y in ys]
    L = run(4, track(4, frames))
    assert L.labels[3] != L.labels[0]              # stays separate, no interface recreated
    assert L.labels[0] == L.labels[1] == L.labels[2]
    # and the mug is part of the table object before it is lifted
    L2 = run(4, track(4, frames[:3]))
    assert L2.n_objects() == 1


def test_hand_first_seen_apart_never_joins():
    base = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 0.8)}   # table (0,1) with mug (2)
    far = [(5.0, 5.0)] * 3
    near = lerp((5.0, 5.0), (1.0, 1.6), 6) + [(1.0, 1.6)] * 4
    frames = [{**base, 3: h} for h in far + near]
    L = run(4, track(4, frames))
    assert L.labels[3] != L.labels[2]
    assert L.labels[3] != L.labels[0]


def test_pen_stays_one_object_while_moving():
    frames = []
    for dx in np.linspace(0, 8, 12):
        frames.append({0: (dx, 3.0), 1: (dx + 0.9, 3.0), 2: (dx + 1.8, 3.0)})
    L = run(3, track(3, frames))
    assert L.n_objects() == 1


def test_articulation_one_object():
    frames = []
    for ang in np.linspace(0, np.pi, 12):
        c = (0.9 + 0.9 * np.cos(np.pi - ang) * -1, 0.9 * np.sin(ang))
        frames.append({0: (0.0, 0.0), 1: (0.9, 0.0), 2: (0.9 + 0.9 * np.cos(ang), 0.9 * np.sin(ang))})
    L = run(3, track(3, frames))
    d02 = None
    assert L.n_objects() == 1


def test_occlusion_opens_nothing_then_opens_when_seen_apart():
    a, b = (0.0, 0.0), (0.8, 0.0)
    frames = [{0: a, 1: b}] * 2
    frames += [{0: a, 1: None}] * 5                 # b hidden while it moves away
    L = run(2, track(2, frames))
    assert L.n_objects() == 1
    frames += [{0: a, 1: (4.0, 0.0)}] * 3           # now seen clearly apart
    L = run(2, track(2, frames))
    assert L.n_objects() == 2


def test_apart_must_persist_k_observations():
    frames = [{0: (0.0, 0.0), 1: (0.8, 0.0)}] * 2
    frames += [{0: (0.0, 0.0), 1: (4.0, 0.0)}] * (K - 1)
    frames += [{0: (0.0, 0.0), 1: (0.8, 0.0)}] * 3
    L = run(2, track(2, frames))
    assert L.n_objects() == 1


@pytest.mark.parametrize("floor_first", [True, False])
def test_pi_orders_admission_of_new_part(floor_first):
    # f and a start adjacent, then a leaves for k frames -> two objects
    f, a0, a_far, a_back = (0.0, 0.0), (0.8, 0.0), (6.0, 0.0), (1.4, 0.0)
    frames = [{0: f, 1: a0}] * 2 + [{0: f, 1: a_far}] * (K + 1)
    frames += [{0: f, 1: a_back, 2: (0.5, 0.0)}]    # new patch c adjacent to both
    n = 3

    def pi_fn(ci, cj):
        out = []
        for i, j in zip(ci, cj):
            pair = {int(i), int(j)}
            want = {0, 2} if floor_first else {1, 2}
            out.append(1.0 if pair == want else 0.0)
        return np.array(out)

    L = run(n, track(n, frames), pi_fn)
    assert L.n_objects() == 2
    if floor_first:
        assert L.labels[2] == L.labels[0] != L.labels[1]
    else:
        assert L.labels[2] == L.labels[1] != L.labels[0]


def test_invariants_on_random_scenes():
    rng = np.random.default_rng(0)
    for trial in range(15):
        n = 14
        pos = rng.uniform(0, 3, (n, 2))
        vel = rng.normal(0, 0.12, (n, 2))
        appear = rng.integers(0, 12, n)
        L = FormalLayer(n, EPS, K)
        hist = []
        for t in range(40):
            if t % 8 == 0:                       # occasional kicks create divisions
                vel = rng.normal(0, 0.2, (n, 2)) * (rng.random((n, 1)) < 0.4)
            pos = pos + vel
            vis = (t >= appear) & (rng.random(n) > 0.1)
            p3 = np.zeros((n, 3))
            p3[:, :2] = pos
            L.step(p3, vis)
            hist.append((L.labels.copy(), L.known.copy()))
        for t in range(1, len(hist)):
            lt, kt = hist[t]
            for s in range(t):
                ls, ks = hist[s]
                both = np.flatnonzero(ks)
                for x in both:
                    for y in both:
                        if ls[x] != ls[y]:       # permanent separation
                            assert lt[x] != lt[y]
                        else:                    # monotonicity (reverse direction, same-time pairs)
                            pass
            # partition: every known patch has exactly one object
            assert np.all(lt[kt] >= 0) and np.all(lt[~kt] == -1)
        # monotonicity: same object at the end => same object whenever both were known
        lt, kt = hist[-1]
        for s in range(len(hist) - 1):
            ls, ks = hist[s]
            for x in np.flatnonzero(ks):
                for y in np.flatnonzero(ks):
                    if lt[x] == lt[y]:
                        assert ls[x] == ls[y]


def test_a_divided_object_is_replaced_by_new_objects_that_point_to_it():
    floor = {i: (float(i), 0.0) for i in range(6)}
    still = [{**floor, 6: (2.0, 0.8), 7: (3.0, 0.8)}] * 4
    lifted = [{**floor, 6: (2.0, y), 7: (3.0, y)} for y in list(np.linspace(0.8, 3.0, 6)) + [3.0] * 3]
    n = 8
    L = FormalLayer(n, EPS, K)
    for pos, vis in track(n, still):
        p3 = np.zeros((n, 3)); p3[:, :2] = pos
        L.step(p3, vis)
    old = int(L.labels[0])
    assert L.n_objects() == 1 and set(L.labels) == {old}
    for pos, vis in track(n, lifted):
        p3 = np.zeros((n, 3)); p3[:, :2] = pos
        L.step(p3, vis)
    new = {int(L.labels[0]), int(L.labels[6])}
    assert len(new) == 2 and old not in new                    # the old object does not sustain
    assert all(L.parent[i] == old for i in new)                # each part points back to it


def test_an_undivided_object_keeps_its_id_when_it_gains_patches():
    n = 4
    L = FormalLayer(n, EPS, K)
    frames = track(n, [{0: (0.0, 0.0), 1: (0.9, 0.0)}] * 3 + [{0: (0.0, 0.0), 1: (0.9, 0.0), 2: (1.8, 0.0)}] * 3)
    ids = []
    for pos, vis in frames:
        p3 = np.zeros((n, 3)); p3[:, :2] = pos
        L.step(p3, vis)
        ids.append(int(L.labels[0]))
    assert len(set(ids)) == 1 and L.labels[2] == ids[0]
