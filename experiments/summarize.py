import glob, json, sys
import numpy as np

runs = {f: json.load(open(f)) for f in sorted(glob.glob("experiments/out/final_random_s*.json"))}
eps = [p["episode"] for p in next(iter(runs.values()))["probes"]]
print("Mean over", len(runs), "seeds (random interventions). Frozen probes on 6 fresh scenes each.\n")
print(f"{'episodes':>8} | {'train prec':>10} {'recall':>7} {'ARI':>6} {'unk':>5} | {'bottle prec':>11} {'recall':>7} {'ARI':>6} {'unk':>5}")
for i, e in enumerate(eps):
    def m(kind, key):
        return np.nanmean([r["probes"][i][kind][key] for r in runs.values()])
    print(f"{e:8d} | {m('train','precision'):10.2f} {m('train','recall'):7.2f} {m('train','ari_predicted'):6.2f} {m('train','unknown_frac'):5.2f} | "
          f"{m('bottle','precision'):11.2f} {m('bottle','recall'):7.2f} {m('bottle','ari_predicted'):6.2f} {m('bottle','unknown_frac'):5.2f}")
print("\nFormal layer alone after 3 interventions (ARI of O_t vs bodies), first 10 vs last 10 episodes:")
for f, r in runs.items():
    a = [e["end_formal_ari"] for e in r["episodes"]]
    print(f"  {f.split('/')[-1]}: {np.mean(a[:10]):.2f} -> {np.mean(a[-10:]):.2f}")
print("\nStatement breakdown on unseen 'bottle' scenes at the end [unknown, distinct, coherent], summed over seeds:")
tot = {}
for r in runs.values():
    for c, v in r["probes"][-1]["bottle"]["breakdown"].items():
        tot[c] = np.add(tot.get(c, [0, 0, 0]), v)
for c, v in tot.items():
    print(f"  {c:14s} {list(map(int, v))}   (true answer: {'distinct' if c in ('body-floor','body-body') else 'coherent'})")
u = glob.glob("experiments/out/final_uncertainty_s0.json")
if u:
    r = json.load(open(u[0])); b = json.load(open("experiments/out/final_random_s0.json"))
    print("\nIntervention policy, seed 0, final probe (random vs uncertainty):")
    for k in ("train", "bottle"):
        for name, rr in (("random", b), ("uncertainty", r)):
            p = rr["probes"][-1][k]
            print(f"  {k:6s} {name:11s} precision {p['precision']:.2f} recall {p['recall']:.2f} ARI {p['ari_predicted']:.2f} unknown {p['unknown_frac']:.2f}")
