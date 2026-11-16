"""All experiments in the paper, from the cached runs of the real system (collect.py).  Each experiment writes its own json.

    PYTHONPATH=. .venv/bin/python experiments/paper/run_experiments.py <name> [<name> ...]
"""
import json
import sys
import time

import numpy as np

from experiments.paper import evaluate as E

OUT = E.OUT
T0 = time.time()


def log(*a):
    print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)


def save(name, obj):
    json.dump(obj, open(f"{OUT}/res_{name}.json", "w"), indent=1, default=lambda v: None if v is None else float(v))


def spec_for(families=E.FAMILIES, n=None, rng=None):
    spec = {}
    for f in families:
        N = E.n_train(f)
        spec[f] = np.arange(N) if n is None else rng.choice(N, min(n, N), replace=False)
    return spec


def build(method, spec, **kw):
    X, y, w = E.entries(spec)
    return dict(latent=E.Latent, logistic=E.Logistic, edge=E.EdgeRule)[method](X, y, w, **kw)


def record(res):
    s = E.summary(res)
    s["groups"] = E.by_group(res)
    s["ci_accuracy"] = E.bootstrap(res, "accuracy")
    s["ci_pq"] = E.bootstrap(res, "pq")
    s["per_family"] = E.per_family(res, "correct")
    s["per_family_pq"] = E.per_family(res, "pq")
    s["per_family_fp"] = E.per_family(res, "fp")
    s["per_family_units"] = {f: float(sum(x["units_found"] for x in res[f]) / max(sum(x["n_units"] for x in res[f]), 1)) for f in res}
    s["per_family_auc"] = E.per_family(res, "auc")
    return s


def short(s):
    return {k: round(v, 3) for k, v in s.items() if isinstance(v, float)}


# --------------------------------------------------------------------------------------------------- experiments
def teaching():
    out = {}
    for f in E.FAMILIES:
        t = E.train(f)
        out[f] = dict(scenes=int(len(t["formal"])), intended=float(np.mean(t["formal_big"] == t["want"])),
                      intended_strict=float(np.mean(t["formal"] == t["want"])), taken=float(t["taken"].mean()),
                      entries=int(len(t["y"])), positives=int((t["y"] > 0.5).sum()), negatives=int((t["y"] <= 0.5).sum()),
                      events=float(t["events"].mean()))
    save("teaching", out)
    log("teaching", {f: round(v["intended"], 2) for f, v in out.items()})


def main_table():
    spec = spec_for()
    out = {}
    methods = {"latent": ("latent", {}), "latent_relevance_flipped": ("latent", dict(relevance=not E.SETTINGS.get("relevance", True))),
               "latent_no_abstention": ("latent", dict(abstain=False)),
               "latent_cuts_only": ("latent", dict(kinds=["cut"])), "latent_regions_only": ("latent", dict(kinds=["region"])),
               "logistic": ("logistic", {}), "edge_rule": ("edge", {})}
    X, y, w = E.entries(spec, carried=None)                     # the app's original settings, for comparison
    out["latent_default"] = record(E.score(E.Latent(X, y, w, k_nn=24, theta_hi=0.65, theta_lo=0.35, relevance=True)))
    log("main latent with the app's original settings", short(out["latent_default"]))
    for name, (m, kw) in methods.items():
        out[name] = record(E.score(build(m, spec, **kw)))
        log("main", name, short(out[name]), {g: round(v.get("accuracy", np.nan), 3) for g, v in out[name]["groups"].items()})
        save("main", out)
    out["never_split"] = record(E.score(E.Never()))
    log("main never", short(out["never_split"]))
    save("main", out)


def ablations():
    spec = spec_for()
    X, y, w = E.entries(spec)
    out = {}
    for g, cols in E.GROUPS.items():
        keep = [c for c in range(X.shape[1]) if c not in cols]
        out[f"without_{g}"] = record(E.score(E.Latent(X, y, w, cols=keep)))
        log("without", g, short(out[f"without_{g}"]))
        out[f"only_{g}"] = record(E.score(E.Latent(X, y, w, cols=cols)))
        log("only", g, short(out[f"only_{g}"]))
        save("ablation", out)
    for k in (6, 12, 48, 96):
        out[f"k_nn_{k}"] = record(E.score(E.Latent(X, y, w, k_nn=k)))
        log("k_nn", k, short(out[f"k_nn_{k}"]))
    for th in (0.55, 0.75):
        out[f"theta_{th}"] = record(E.score(E.Latent(X, y, w, theta_hi=th, theta_lo=1 - th)))
        log("theta", th, short(out[f"theta_{th}"]))
    for wc in (0.05, 0.2, 0.7, 1.0):
        Xc, yc, wcw = E.entries(spec, carried=wc)
        out[f"carried_{wc}"] = record(E.score(E.Latent(Xc, yc, wcw)))
        log("carried weight", wc, short(out[f"carried_{wc}"]))
    save("ablation", out)


def learning_curve(seeds=5):
    out = {}
    for n in (1, 2, 5, 10, 20, 40):
        rows = []
        for sd in range(seeds if n < 40 else 1):
            res = E.score(build("latent", spec_for(n=n, rng=np.random.default_rng(100 + sd))))
            s = E.summary(res); s["groups"] = E.by_group(res)
            rows.append(s)
        out[str(n)] = rows
        log("curve", n, round(np.mean([r["accuracy"] for r in rows]), 3), "+-", round(np.std([r["accuracy"] for r in rows]), 3))
        save("curve", out)


def leave_one_family_out():
    out = {}
    for f in E.FAMILIES:
        others = [g for g in E.FAMILIES if g != f]
        out[f] = {}
        for name in ("latent", "logistic"):
            res = E.score(build(name, spec_for(others)), families=[f])
            out[f][name] = dict(accuracy=E.summary(res)["accuracy"], pq=E.summary(res)["pq"], unit_recall=E.summary(res)["unit_recall"],
                                false_per_scene=E.summary(res)["false_per_scene"], auc=E.summary(res)["auc"], unknown_true=E.summary(res)["unknown_true"])
        log("lofo", f, {k: round(v["accuracy"], 2) for k, v in out[f].items()}, "unknown on true units", out[f]["latent"]["unknown_true"])
        save("lofo", out)


def group_transfer():
    """Train on one kind of thing, test on another: single objects -> arrangements, arrangements -> single objects."""
    out = {}
    for name, train_f, test_f in (("single_to_arranged", E.SINGLE + E.WHOLE, E.ARRANGED), ("arranged_to_single", E.ARRANGED + E.WHOLE, E.SINGLE),
                                  ("single_only_to_whole", E.SINGLE, E.WHOLE), ("arranged_only_to_whole", E.ARRANGED, E.WHOLE)):
        res = E.score(build("latent", spec_for(train_f)), families=test_f)
        out[name] = record(res)
        log("transfer", name, short(out[name]))
    save("transfer", out)


def few_shot(ks=(0, 1, 2, 5, 10), seeds=3):
    out = {}
    for f in E.FAMILIES:
        others = [g for g in E.FAMILIES if g != f]
        rows = {}
        for k in ks:
            accs = []
            for sd in range(seeds if k else 1):
                spec = spec_for(others)
                if k:
                    spec[f] = np.random.default_rng(300 + sd).choice(E.n_train(f), k, replace=False)
                accs.append(E.summary(E.score(build("latent", spec), families=[f])))
            rows[str(k)] = accs
        out[f] = rows
        log("few-shot", f, {k: round(np.mean([a["accuracy"] for a in v]), 2) for k, v in rows.items()})
        save("few_shot", out)


def robustness():
    spec = spec_for()
    lat, lr = build("latent", spec), build("logistic", spec)
    out = {}
    for kind in ("none", "rot90", "rot180", "scale0.7", "recolour", "dark", "noise"):
        out[kind] = dict(latent=record(E.score(lat, kind=kind)), logistic=record(E.score(lr, kind=kind)))
        log("robust", kind, round(out[kind]["latent"]["accuracy"], 3), "logistic", round(out[kind]["logistic"]["accuracy"], 3))
        save("robust", out)


def calibration(bins=10):
    """Reliability of p over every candidate of every test scene; label: the candidate is a true unit (IoU >= 0.75)."""
    res = E.score(build("latent", spec_for()))
    p = np.concatenate([x["p"] for f in res for x in res[f]])
    y = np.concatenate([x["y"] for f in res for x in res[f]])
    edges = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & ((p < b) if b < 1 else (p <= b))
        if m.any():
            rows.append(dict(lo=float(a), hi=float(b), n=int(m.sum()), mean_p=float(p[m].mean()), freq=float(y[m].mean())))
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    out = dict(bins=rows, ece=float(ece), n=int(len(p)), base_rate=float(y.mean()))
    log("calibration ECE", round(ece, 4), "n", len(p), "base rate", round(float(y.mean()), 4))
    save("calibration", out)


def ood():
    """Distance to memory and abstention on the true units: family in memory vs family held out."""
    full = build("latent", spec_for())
    out = {}
    for f in E.FAMILIES:
        held = build("latent", spec_for([g for g in E.FAMILIES if g != f]))
        row = {}
        for tag, m in (("seen", full), ("held_out", held)):
            near, unk, sup = [], [], []
            for r in E.probes(f):
                for o in r["objects"]:
                    if len(o["X"]) == 0:
                        continue
                    q = m(o["X"])
                    near.append(np.median(q["nearest"])); sup.append(np.median(q["support"])); unk.append(np.mean(q["statement"] == 0))
            row[tag] = dict(nearest=float(np.median(near)), support=float(np.median(sup)), unknown=float(np.mean(unk)))
        out[f] = row
        log("ood", f, row)
    save("ood", out)


def failures():
    """Why units are missed: no candidate overlaps the unit well enough (generator), a good candidate exists but was not called
    (memory), or a called candidate overlaps it badly.  Also which traits a relevance-weighted memory would weight most."""
    from objectlearn.latent import LatentMemory
    from objectlearn.regions import TRAIT_NAMES
    meth = build("latent", spec_for())
    out = {}
    for f in E.FAMILIES:
        if E.GROUP[f] == "whole":
            continue
        c = dict(units=0, found=0, no_candidate=0, not_called=0, called_badly=0)
        for r in E.probes(f):
            pred = E.predict(meth, r)
            _, _, _, _, mg = E.match(pred, r["gt_v"])
            for g in r["units"]:
                c["units"] += 1
                if g in mg:
                    c["found"] += 1
                    continue
                best, called = 0.0, False
                for o in r["objects"]:
                    if len(o["masks"]) == 0:
                        continue
                    lab = r["gt_v"][o["loc"]]
                    gm = (r["gt_v"] == g).sum()
                    inter = (o["masks"] & (lab == g)[None, :]).sum(1)
                    iou = inter / np.maximum(o["masks"].sum(1) + gm - inter, 1)
                    best = max(best, float(iou.max()))
                    q = meth(o["X"])
                    called |= bool(((q["statement"] == 1) & (iou >= 0.5)).any())
                if best < 0.5:
                    c["no_candidate"] += 1
                elif not called:
                    c["not_called"] += 1
                else:
                    c["called_badly"] += 1
        out[f] = c
        log("failures", f, c)
    X, y, w = E.entries(spec_for())
    m = LatentMemory(relevance=True); m.add(X, y, w)
    rel = m.relevance()
    out["relevance_top"] = [(TRAIT_NAMES[i], float(rel[i])) for i in np.argsort(-rel)[:8]]
    log("relevance top", out["relevance_top"])
    save("failures", out)


ALL = dict(failures=failures, teaching=teaching, main=main_table, ablations=ablations, curve=learning_curve, lofo=leave_one_family_out, transfer=group_transfer,
           few_shot=few_shot, robust=robustness, calibration=calibration, ood=ood)

if __name__ == "__main__":
    for name in sys.argv[1:]:
        ALL[name]()
    log("done", sys.argv[1:])
