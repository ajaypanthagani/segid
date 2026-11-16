import numpy as np


def ari(a, b):
    """Adjusted Rand index between two labelings (equal-length int arrays)."""
    a = np.unique(a, return_inverse=True)[1]; b = np.unique(b, return_inverse=True)[1]
    n = len(a)
    if n < 2:
        return 1.0
    cont = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(cont, (a, b), 1)
    c2 = lambda x: x * (x - 1) / 2
    sa, sb, sc = c2(cont.sum(1)).sum(), c2(cont.sum(0)).sum(), c2(cont).sum()
    exp = sa * sb / c2(n)
    mx = 0.5 * (sa + sb)
    return 1.0 if mx == exp else (sc - exp) / (mx - exp)


def interface_report(agent, world):
    """Score the statements made about every admitted interface against ground truth."""
    ii, jj, rows = agent.interfaces()
    gt = world.patch_body[ii] != world.patch_body[jj]
    st = agent.stat[rows]
    dist, coh, unk = st == 1, st == 2, st == 0
    tp = int((dist & gt).sum()); fp = int((dist & ~gt).sum())
    out = dict(
        n=int(len(ii)), n_true_distinct=int(gt.sum()),
        pred_distinct=int(dist.sum()), tp=tp, fp=fp,
        recall=tp / max(int(gt.sum()), 1), precision=tp / max(tp + fp, 1) if dist.sum() else float("nan"),
        coherent_wrong=int((coh & gt).sum()), coherent=int(coh.sum()), unknown_frac=float(unk.mean()) if len(st) else 1.0)
    known = agent.formal.known
    out["ari_predicted"] = ari(agent.predict_partition()[known], world.patch_body[known])
    out["ari_formal"] = ari(agent.formal.labels[known], world.patch_body[known])
    return out


CATS = ("floor-floor", "body-internal", "body-floor", "body-body")


def categories(world, ii, jj):
    pb = world.patch_body
    same = pb[ii] == pb[jj]
    onfloor = (pb[ii] == 0) | (pb[jj] == 0)
    return np.where(same, np.where(pb[ii] == 0, 0, 1), np.where(onfloor, 2, 3))


def breakdown(agent, world):
    """Counts of statements (unknown / distinct / coherent) for each kind of admitted interface."""
    ii, jj, rows = agent.interfaces()
    cat = categories(world, ii, jj)
    st = agent.stat[rows]
    return {CATS[c]: [int(((cat == c) & (st == s)).sum()) for s in (0, 1, 2)] for c in range(4)}
