import numpy as np

from objectlearn.agent import Agent
from objectlearn.descriptor import D, pair_descriptors
from objectlearn.episode import start_scene
from objectlearn.learned import Config, Estimator, SeamMemory, make_experts
from objectlearn.sim import Frame, make_scene


def test_empty_memory_is_unknown():
    cfg = Config(); ex = make_experts(cfg)
    mem, est = SeamMemory(cfg), Estimator(cfg, ex)
    cnt = mem.query(np.zeros((3, D), np.float32), np.zeros((3, D), np.float32), ex)
    e = est.mix(cnt)
    assert np.all(e["s"] == 0) and np.all(est.statement(e) == 0)


def test_memory_gives_distinct_near_and_unknown_far():
    cfg = Config(); ex = make_experts(cfg)
    rng = np.random.default_rng(0)
    mem, est = SeamMemory(cfg), Estimator(cfg, ex)
    centre = rng.normal(0, 1, D).astype(np.float32)
    other = (centre + 5).astype(np.float32)
    mem.add(centre + rng.normal(0, .05, (300, D)).astype(np.float32), np.ones(300), np.full(300, 0.05))
    mem.add(other + rng.normal(0, .05, (300, D)).astype(np.float32), np.zeros(300), np.full(300, 0.05))
    near_open = (centre + rng.normal(0, .05, (5, D))).astype(np.float32)
    near_held = (other + rng.normal(0, .05, (5, D))).astype(np.float32)
    far = (centre - 20).astype(np.float32)[None]
    s1 = est.statement(est.mix(mem.query(near_open, near_open, ex)))
    s0 = est.statement(est.mix(mem.query(near_held, near_held, ex)))
    sf = est.statement(est.mix(mem.query(far, far, ex)))
    assert np.all(s1 == 1) and np.all(s0 == 2) and np.all(sf == 0)


def test_descriptor_is_rotation_and_translation_invariant():
    rng = np.random.default_rng(3)
    w = make_scene(rng, "train")
    f = w.frame()
    vi = np.flatnonzero(f.vis)
    # pick some adjacent visible pairs
    from scipy.spatial import cKDTree
    pairs = np.array(sorted(cKDTree(f.pos[vi]).query_pairs(0.06)))[::40][:12]
    pairs = vi[pairs]
    th = 0.9
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    g = Frame(pos=f.pos @ R.T + np.array([3., -2., 0.5]), nrm=f.nrm @ R.T, col=f.col, vis=f.vis)
    a1, b1 = pair_descriptors(f, pairs, 0.05)
    a2, b2 = pair_descriptors(g, pairs, 0.05)
    assert np.allclose(a1, a2, atol=1e-4) and np.allclose(b1, b2, atol=1e-4)


def test_pick_and_place_writes_labelled_samples_and_refines_formal():
    rng = np.random.default_rng(7)
    ag = Agent(Config(), seed=0)
    w = start_scene(ag, rng, "train")
    bid = [b.id for b in w.bodies if not b.floor][0]
    for f in w.pick_and_place(bid, 0.2, (0.1, 0.0)):
        ag.observe(f)
    entry = ag.end_event()
    assert entry is not None and entry["n_open"] > 0
    y, wt = ag.mem.y, ag.mem.w
    assert (y == 1).any() and np.all(wt[y == 1] <= ag.cfg.event_budget + 1e-6)
    assert np.all(wt[y == 0] <= ag.cfg.w_max + 1e-6)
    # every opened candidate pair really joined two different bodies in the world (box vs support)
    ii, jj = np.nonzero(np.triu(ag.formal.opened, 1))
    assert np.all(w.patch_body[ii] != w.patch_body[jj])
    # predicted partition refines the formal partition
    pl, fl = ag.predict_partition(), ag.formal.labels
    for lab in np.unique(pl[pl >= 0]):
        assert len(np.unique(fl[pl == lab])) == 1


def test_learned_layer_never_writes_to_formal_state():
    rng = np.random.default_rng(11)
    ag = Agent(Config(), seed=1)
    w = start_scene(ag, rng, "train")
    before = (ag.formal.I.copy(), ag.formal.opened.copy(), ag.formal.labels.copy())
    ag.predict_partition(); ag.choose_grasp(ag.frame, "uncertainty")
    after = (ag.formal.I, ag.formal.opened, ag.formal.labels)
    assert all(np.array_equal(a, b) for a, b in zip(before, after))


def test_refresh_restates_interfaces_from_current_memory():
    rng = np.random.default_rng(5)
    ag = Agent(Config(), seed=0)
    w = start_scene(ag, rng, "train")                       # scene formed with an empty memory
    assert (ag.stat == 0).all()
    for _ in range(4):                                       # learn from a few interventions elsewhere
        w2 = start_scene(ag, rng, "train")
        for b in [b.id for b in w2.bodies if not b.floor][:2]:
            for f in w2.pick_and_place(b, 0.2, (0.1, 0.0)):
                ag.observe(f)
            ag.end_event()
    w = start_scene(ag, rng, "train")
    before = ag.stat.copy()
    ag.refresh()
    assert np.array_equal(ag.stat, before)                   # creation-time statements are untouched
    pl, fl = ag.predict_partition(current=True), ag.formal.labels
    for lab in np.unique(pl[pl >= 0]):
        assert len(np.unique(fl[pl == lab])) == 1            # still a refinement of the formal partition
