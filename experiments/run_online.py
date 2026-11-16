"""Online learning curve.  The model starts empty and learns only from its own interventions.

At checkpoints it is probed (frozen: no learning, no interventions) on fresh scenes from
  - the training distribution (low wide boxes with stacked children), and
  - an unseen distribution (tall thin bodies with a cap, plus a low box).
"""
import argparse, json, time
import warnings

import numpy as np

from objectlearn.agent import Agent
from objectlearn.episode import probe, run_episode
from objectlearn.learned import Config

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=60)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--policy", default="random")
ap.add_argument("--probe_n", type=int, default=6)
ap.add_argument("--checkpoints", default="0,5,10,20,30,45,60")
ap.add_argument("--out", default="experiments/out/run.json")
a = ap.parse_args()

agent = Agent(Config(), seed=a.seed)
rng = np.random.default_rng(a.seed)
cps = {int(x) for x in a.checkpoints.split(",")}
res = dict(args=vars(a), probes=[], episodes=[])
t0 = time.time()


def do_probe(ep):
    row = dict(episode=ep, memory=len(agent.mem), group_weights={k: float(v) for k, v in agent.est.group_weights().items()},
               train=probe(agent, np.random.default_rng(1000), "train", a.probe_n),
               bottle=probe(agent, np.random.default_rng(2000), "bottle", a.probe_n))
    res["probes"].append(row)
    f = lambda d: " ".join(f"{k}={d[k]:.2f}" for k in ("precision", "recall", "unknown_frac", "ari_predicted", "coherent_wrong"))
    print(f"[{time.time()-t0:5.0f}s] ep {ep:3d} mem {row['memory']:6d} gw {row['group_weights']}\n   train : {f(row['train'])}\n   bottle: {f(row['bottle'])}", flush=True)


if 0 in cps:
    do_probe(0)
for ep in range(1, a.episodes + 1):
    start, entries, end_ari = run_episode(agent, rng, "train", 3, a.policy)
    res["episodes"].append(dict(episode=ep, start=start, events=[{k: v for k, v in e.items() if k != "group_w"} for e in entries], end_formal_ari=end_ari))
    if ep in cps:
        do_probe(ep)
        json.dump(res, open(a.out, "w"), indent=1, default=float)
json.dump(res, open(a.out, "w"), indent=1, default=float)
print("done", flush=True)
