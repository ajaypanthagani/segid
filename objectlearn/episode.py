import numpy as np

from .metrics import breakdown, interface_report
from .sim import make_scene


def start_scene(agent, rng, kind, static=4):
    world = make_scene(rng, kind, s=agent.cfg.s)
    agent.new_scene(world.n)
    for f in world.static(static):
        agent.observe(f)
    return world


def probe(agent, rng, kind, n):
    """Frozen evaluation on fresh scenes: no learning, no interventions."""
    agent.learn = False
    rows, br = [], {}
    for _ in range(n):
        w = start_scene(agent, rng, kind)
        rows.append(interface_report(agent, w))
        for c, v in breakdown(agent, w).items():
            br[c] = np.add(br.get(c, [0, 0, 0]), v)
    agent.learn = True
    keys = ["precision", "recall", "unknown_frac", "ari_predicted", "ari_formal", "coherent_wrong", "fp", "tp", "pred_distinct"]
    out = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
    out["breakdown"] = {c: [int(x) for x in v] for c, v in br.items()}     # [unknown, distinct, coherent]
    return out


def intervene(agent, world, rng, policy="random"):
    frame = agent.frame
    patch = agent.choose_grasp(frame, policy)
    if patch is None:
        return None
    bid = int(world.patch_body[patch])
    if world.bodies[bid].floor:
        return None
    ang = rng.uniform(0, 2 * np.pi)
    mag = rng.uniform(0.08, 0.16)
    dxy = mag * np.array([np.cos(ang), np.sin(ang)])
    lo, sz = world._lo(bid), world._size(bid)
    W = world.s * 16
    dxy = np.clip(lo[:2] + dxy, 0.02, W - sz[:2] - 0.02) - lo[:2]
    for f in world.pick_and_place(bid, rng.uniform(0.15, 0.25), dxy):
        agent.observe(f)
    return agent.end_event()


def run_episode(agent, rng, kind="train", n_interventions=3, policy="random"):
    world = start_scene(agent, rng, kind)
    start = interface_report(agent, world)           # statements made before any intervention
    entries = []
    for _ in range(n_interventions):
        e = intervene(agent, world, rng, policy)
        if e:
            entries.append(e)
    known = agent.formal.known
    from .metrics import ari
    end_ari = ari(agent.formal.labels[known], world.patch_body[known])
    return start, entries, end_ari
