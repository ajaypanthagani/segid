import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

r = json.load(open(sys.argv[1]))
P = r["probes"]
x = [p["episode"] for p in P]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for kind, ls in (("train", "-"), ("bottle", "--")):
    ax[0].plot(x, [p[kind]["precision"] for p in P], ls, marker="o", label=f"precision ({kind})")
    ax[0].plot(x, [p[kind]["recall"] for p in P], ls, marker="s", label=f"recall ({kind})")
    ax[1].plot(x, [p[kind]["ari_predicted"] for p in P], ls, marker="o", label=f"predicted objects ({kind})")
    ax[2].plot(x, [p[kind]["unknown_frac"] for p in P], ls, marker="o", label=kind)
ax[0].set_title("Interfaces predicted distinct"); ax[1].set_title("Predicted objects vs truth (ARI)"); ax[2].set_title("Fraction left unknown")
for a in ax:
    a.set_xlabel("episodes of experience (3 interventions each)"); a.set_ylim(0, 1.02); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=130)
