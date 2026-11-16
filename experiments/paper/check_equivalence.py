"""Check that scoring from the cache (evaluate.predict) gives exactly the partition the live session predicts.

For a sample of test scenes: load the full training memory into a fresh session, place the scene, let the session make its
prediction, and compare its predicted object labels on every visible patch with evaluate.predict on the cached candidates.

    PYTHONPATH=. .venv/bin/python experiments/paper/check_equivalence.py [per_family]
"""
import json
import sys

import numpy as np

from experiments.paper import evaluate as E
from experiments.paper.collect import Rig
from experiments.paper.scenes import make
from objectlearn.latent import LatentMemory


def same_partition(a, b):
    """Two labelings induce the same partition."""
    pairs = np.unique(np.stack([a, b], 1), axis=0)
    return len(pairs) == len(np.unique(a)) == len(np.unique(b))


def main():
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    X, y, w = E.entries({f: np.arange(E.n_train(f)) for f in E.FAMILIES})
    method = E.Latent(X, y, w)
    rig = Rig()
    agree, total, rows = 0, 0, []
    for f in E.FAMILIES:
        for i, r in enumerate(E.probes(f)[:per]):
            sc = make(f, 50000 + i)
            rig.fresh(sc["img"])
            app = rig.app
            mem = LatentMemory(k_nn=E.SETTINGS["k_nn"], theta_hi=E.SETTINGS["theta"], theta_lo=1 - E.SETTINGS["theta"], relevance=E.SETTINGS.get("relevance", True)); mem.add(X, y, w)
            app.session.latent = mem
            rig.place(sc["img"])
            for _ in range(5):
                rig.tick()
            app.session._lat_due = True
            res = rig.tick()
            fr, _ = app.world.frame()
            lf = app.session._raw_f
            vis = np.flatnonzero(fr.vis & (lf >= 0))
            # the session's predicted label for each visible patch, in the same order as the cache
            lab = {(int(p[0]), int(p[1])): p[3] for p in res["pts"]}
            online = np.array([lab[(int(fr.pos[v, 0]), int(fr.pos[v, 1]))] for v in vis])
            offline = E.predict(method, r)
            ok = len(online) == len(offline) and same_partition(online, offline)
            agree += ok; total += 1
            rows.append(dict(family=f, scene=i, agree=bool(ok), online=int(len(np.unique(online))), offline=int(len(np.unique(offline)))))
            print(f, i, "agree" if ok else "DIFFER", len(np.unique(online)), len(np.unique(offline)), flush=True)
    json.dump(dict(agree=agree, total=total, rows=rows), open(f"{E.OUT}/res_equivalence.json", "w"), indent=1)
    print(f"{agree}/{total} scenes: the cached scoring reproduces the live session's prediction exactly")


if __name__ == "__main__":
    main()
