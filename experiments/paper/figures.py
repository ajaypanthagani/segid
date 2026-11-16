"""Figures and LaTeX tables for the paper, from experiments/paper/out/res_*.json and the cached scenes.

    PYTHONPATH=. .venv/bin/python experiments/paper/figures.py
"""
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.paper import evaluate as E
from experiments.paper.scenes import make, partition

OUT = E.OUT
FIG = "paper/figures"
TAB = "paper/tables"
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
                     "font.family": "serif"})
C = {"single": "#2a6fbb", "arranged": "#d1712c", "whole": "#4b8b3b", "all": "#333333", "latent": "#2a6fbb", "logistic": "#999999"}


def load(name):
    return json.load(open(f"{OUT}/res_{name}.json"))


def pct(v):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{100 * v:.1f}"


def num(v, d=2):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


# ------------------------------------------------------------------------------------------------------- figures
def fig_scenes():
    """One example per family with its ground-truth partition."""
    fams = E.FAMILIES
    fig, axs = plt.subplots(3, 7, figsize=(7.0, 3.6))
    for ax, f in zip(axs.ravel(), fams):
        sc = make(f, 50003)
        img = sc["img"][:, :, ::-1].copy()
        lab = partition(sc)
        e = np.zeros(lab.shape, bool)
        e[:-1] |= lab[:-1] != lab[1:]; e[:, :-1] |= lab[:, :-1] != lab[:, 1:]
        e &= lab > 0
        img[cv2.dilate(e.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)] = (230, 20, 20)
        ax.imshow(img); ax.set_title(f.replace("_", " "), fontsize=7, color=C[E.GROUP[f]]); ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.savefig(f"{FIG}/scenes.pdf"); fig.savefig(f"{FIG}/scenes.png")
    plt.close(fig)


def fig_qualitative():
    """Test scenes: ground truth vs. the latent memory's prediction (nothing has moved)."""
    X, y, w = E.entries({f: np.arange(E.n_train(f)) for f in E.FAMILIES})
    meth = E.Latent(X, y, w)
    pick = ["bookshelf", "table", "cabinet", "fruit_bowl", "pencil_cup", "bottle", "pen", "hammer", "mug"]
    fig, axs = plt.subplots(2, len(pick), figsize=(7.0, 2.4))
    for j, f in enumerate(pick):
        r = E.probes(f)[1]
        pred = E.predict(meth, r)
        pal = plt.get_cmap("tab10").colors
        for i, lab in enumerate((r["gt_v"], pred)):
            ax = axs[i, j]
            ids, cnt = np.unique(lab[lab > 0] if i == 0 else lab, return_counts=True)
            order = ids[np.argsort(-cnt)]
            col = {order[0]: (0.62, 0.66, 0.72)}                    # the largest segment (what is left) in neutral grey-blue
            for q, g in enumerate(order[1:]):
                col[g] = pal[q % 10]
            c = np.array([col.get(v, (0.9, 0.9, 0.9)) if not (i == 0 and v == 0) else (0.9, 0.9, 0.9) for v in lab])
            ax.scatter(r["pos"][:, 0], -r["pos"][:, 1], s=1.2, c=c, marker="s", linewidths=0)
            ax.set_aspect("equal"); ax.axis("off")
            if i == 0:
                ax.set_title(f.replace("_", " "), fontsize=7)
        axs[0, 0].text(-0.15, 0.5, "truth", transform=axs[0, 0].transAxes, rotation=90, va="center", ha="right", fontsize=7)
        axs[1, 0].text(-0.15, 0.5, "predicted", transform=axs[1, 0].transAxes, rotation=90, va="center", ha="right", fontsize=7)
    fig.tight_layout(pad=0.2)
    fig.savefig(f"{FIG}/qualitative.pdf"); fig.savefig(f"{FIG}/qualitative.png")
    plt.close(fig)


def fig_curve():
    cur = load("curve")
    ns = sorted(int(k) for k in cur)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.2))
    for key, ax, ylab in (("accuracy", axs[0], "exact partition (%)"), ("pq", axs[1], "panoptic quality (%)")):
        for g in ("single", "arranged", "whole"):
            m = [100 * np.mean([r["groups"][g][key] for r in cur[str(n)]]) for n in ns]
            s = [100 * np.std([r["groups"][g][key] for r in cur[str(n)]]) for n in ns]
            ax.errorbar(ns, m, yerr=s, label=g, color=C[g], marker="o", ms=3, capsize=2, lw=1.2)
        m = [100 * np.mean([r[key] for r in cur[str(n)]]) for n in ns]
        ax.plot(ns, m, color=C["all"], ls="--", lw=1, label="all")
        ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels(ns); ax.set_xlabel("training scenes per family"); ax.set_ylabel(ylab)
        ax.set_ylim(0, 100)
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{FIG}/curve.pdf"); fig.savefig(f"{FIG}/curve.png")
    plt.close(fig)


def fig_lofo():
    lo = load("lofo")
    fams = E.FAMILIES
    x = np.arange(len(fams))
    fig, ax = plt.subplots(figsize=(7.0, 2.1))
    a = [100 * lo[f]["latent"]["accuracy"] for f in fams]
    b = [100 * lo[f]["logistic"]["accuracy"] for f in fams]
    ax.bar(x - 0.2, a, 0.4, color=[C[E.GROUP[f]] for f in fams], label="latent memory")
    ax.bar(x + 0.2, b, 0.4, color=C["logistic"], label="logistic regression")
    ax.set_xticks(x); ax.set_xticklabels([f.replace("_", " ") for f in fams], rotation=60, ha="right", fontsize=6.5)
    ax.set_ylabel("exact partition (%)"); ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{FIG}/lofo.pdf"); fig.savefig(f"{FIG}/lofo.png")
    plt.close(fig)


def fig_few_shot():
    fs = load("few_shot")
    ks = sorted(int(k) for k in next(iter(fs.values())))
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.1), sharey=True)
    for ax, g in zip(axs, ("single", "arranged", "whole")):
        for f in [f for f in E.FAMILIES if E.GROUP[f] == g]:
            m = [100 * np.mean([r["accuracy"] for r in fs[f][str(k)]]) for k in ks]
            ax.plot(ks, m, marker="o", ms=2.5, lw=1, label=f.replace("_", " "))
        ax.set_title(g, fontsize=8, color=C[g]); ax.set_xlabel("scenes of the new family in memory")
        ax.set_xticks(ks); ax.set_ylim(0, 100)
        ax.legend(frameon=False, fontsize=5.5, loc="lower right")
    axs[0].set_ylabel("exact partition (%)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/few_shot.pdf"); fig.savefig(f"{FIG}/few_shot.png")
    plt.close(fig)


def fig_calibration():
    fig, ax = plt.subplots(figsize=(2.6, 2.4))
    ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=0.8)
    for name, lab, col in (("calibration", "validated", C["latent"]), ("calibration_original", "original", C["arranged"])):
        cal = load(name)
        b = [r for r in cal["bins"] if r["n"] >= 20]
        ax.plot([r["mean_p"] for r in b], [r["freq"] for r in b], marker="o", ms=3, color=col, label=f"{lab} (ECE {cal['ece']:.3f})")
    ax.set_xlabel(r"predicted $\hat p$"); ax.set_ylabel("share that are true units")
    ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(f"{FIG}/calibration.pdf"); fig.savefig(f"{FIG}/calibration.png")
    plt.close(fig)


def fig_latent():
    """The stored entries in the three directions of greatest weighted spread, coloured by behaviour and by family group."""
    from objectlearn.latent import LatentMemory
    Xs, ys, gs = [], [], []
    for f in E.FAMILIES:
        t = E.train(f)
        Xs.append(t["X"]); ys.append(t["y"]); gs += [E.GROUP[f]] * len(t["y"])
    X, y, g = np.concatenate(Xs), np.concatenate(ys), np.array(gs)
    mem = LatentMemory(); mem.add(X, y, np.ones(len(y)))
    P = mem.project(X)
    rng = np.random.default_rng(0)
    neg = rng.choice(np.flatnonzero(y < 0.5), 6000, replace=False)
    pos = np.flatnonzero(y > 0.5)
    fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.6))
    for ax, (i, j) in zip(axs, ((0, 1), (0, 2))):
        ax.scatter(P[neg, i], P[neg, j], s=1, c="#bbbbbb", linewidths=0, label="carried whole")
        for gr in ("single", "arranged"):
            sel = pos[g[pos] == gr]
            ax.scatter(P[sel, i], P[sel, j], s=3, c=C[gr], linewidths=0, label=f"came away ({gr})")
        ax.set_xlabel(f"axis {i + 1}"); ax.set_ylabel(f"axis {j + 1}")
    axs[0].legend(frameon=False, fontsize=6, markerscale=3)
    fig.tight_layout()
    fig.savefig(f"{FIG}/latent.pdf"); fig.savefig(f"{FIG}/latent.png")
    plt.close(fig)


# -------------------------------------------------------------------------------------------------------- tables
def tab_main():
    m = load("main")
    names = [("never_split", "Never split (definition only)"), ("edge_rule", "Boundary-contrast rule"), ("logistic", "Logistic regression"),
             ("latent_default", "Latent, original app settings"), ("latent_relevance_flipped", "Latent, relevance-weighted distance"), ("latent_no_abstention", "Latent, no abstention"),
             ("latent_cuts_only", "Latent, straight cuts only"), ("latent_regions_only", "Latent, regions only"), ("latent", r"\textbf{Latent memory (ours)}")]
    rows = []
    for k, label in names:
        s = m[k]
        g = s["groups"]
        rows.append(f"{label} & {pct(s['accuracy'])} {{\\scriptsize [{pct(s['ci_accuracy'][0])}, {pct(s['ci_accuracy'][1])}]}} & {pct(s['pq'])} & "
                    f"{pct(g['single']['accuracy'])} & {pct(g['arranged']['accuracy'])} & {pct(g['whole']['accuracy'])} & "
                    f"{pct(s['unit_recall'])} & {num(s['false_per_scene'])} & {num(s['auc'], 3)} \\\\")
    open(f"{TAB}/main.tex", "w").write("\n".join(rows) + "\n")


def tab_ablation():
    a = load("ablation")
    m = load("main")["latent"]
    rows = [f"Full model & {pct(m['accuracy'])} & {pct(m['pq'])} & {pct(m['groups']['single']['accuracy'])} & {pct(m['groups']['arranged']['accuracy'])} & {pct(m['groups']['whole']['accuracy'])} \\\\ \\midrule"]
    order = [("without_shape", "without shape traits"), ("without_contact", "without contact traits"), ("without_signal", "without signal traits"),
             ("without_where", "without position traits"), ("only_shape", "shape traits only"), ("only_contact", "contact traits only"),
             ("only_signal", "signal traits only"), ("only_where", "position traits only"),
             ("k_nn_6", r"$k=6$"), ("k_nn_12", r"$k=12$"), ("k_nn_48", r"$k=48$"), ("k_nn_96", r"$k=96$"),
             ("theta_0.55", r"$\theta_{+}=0.55,\ \theta_{-}=0.45$"), ("theta_0.75", r"$\theta_{+}=0.75,\ \theta_{-}=0.25$"),
             ("carried_0.05", r"carried-whole weight $w_0=0.05$"), ("carried_0.2", r"carried-whole weight $w_0=0.2$"),
             ("carried_0.7", r"carried-whole weight $w_0=0.7$"), ("carried_1.0", r"carried-whole weight $w_0=1$")]
    for k, label in order:
        if k not in a:
            continue
        s = a[k]
        rows.append(f"{label} & {pct(s['accuracy'])} & {pct(s['pq'])} & {pct(s['groups']['single']['accuracy'])} & {pct(s['groups']['arranged']['accuracy'])} & {pct(s['groups']['whole']['accuracy'])} \\\\"
                    + (" \\midrule" if k in ("without_where", "only_where", "k_nn_96", "theta_0.75") else ""))
    open(f"{TAB}/ablation.tex", "w").write("\n".join(rows) + "\n")


def tab_robust():
    r = load("robust")
    label = dict(none="none", rot90=r"rotated $90^\circ$", rot180=r"rotated $180^\circ$", **{"scale0.7": r"scaled $\times0.7$"},
                 recolour="hue shifted", dark=r"dimmed $\times0.6$", noise=r"noise $\sigma=12$")
    rows = []
    for k in ("none", "rot90", "rot180", "scale0.7", "recolour", "dark", "noise"):
        if k not in r:
            continue
        a, b = r[k]["latent"], r[k]["logistic"]
        rows.append(f"{label[k]} & {pct(a['accuracy'])} & {pct(a['pq'])} & {pct(a['groups']['arranged']['accuracy'])} & {pct(a['groups']['whole']['accuracy'])} & {pct(b['accuracy'])} & {pct(b['pq'])} \\\\")
    open(f"{TAB}/robust.tex", "w").write("\n".join(rows) + "\n")


def tab_teaching():
    t = load("teaching")
    rows = []
    for f in E.FAMILIES:
        s = t[f]
        rows.append(f"{f.replace('_', ' ')} & {E.GROUP[f]} & {s['taken']:.1f} & {pct(s['intended'])} & {pct(s['intended_strict'])} & {s['positives']} & {s['negatives']} \\\\")
    open(f"{TAB}/teaching.tex", "w").write("\n".join(rows) + "\n")


def tab_families():
    m = load("main")
    lo = load("lofo")
    fs = load("few_shot")
    rows = []
    for f in E.FAMILIES:
        a = m["latent"]
        k5 = np.mean([r["accuracy"] for r in fs[f]["5"]]) if "5" in fs[f] else np.nan
        rows.append(f"{f.replace('_', ' ')} & {pct(a['per_family'][f])} & {pct(a['per_family_pq'][f])} & {pct(m['logistic']['per_family'][f])} & "
                    f"{pct(lo[f]['latent']['accuracy'])} & {pct(lo[f]['latent']['unknown_true'])} & {pct(k5)} \\\\")
    open(f"{TAB}/families.tex", "w").write("\n".join(rows) + "\n")


def validation_numbers():
    import glob
    allv = {}
    for fp in glob.glob(f"{OUT}/res_validation*.json"):
        for v in json.load(open(fp)).values():
            v.setdefault("relevance", True)
            allv[(v["relevance"], v["k_nn"], v["w_carried"], v["theta"])] = v
    ch = json.load(open(f"{OUT}/chosen.json"))
    best = allv[(ch["relevance"], ch["k_nn"], ch["w_carried"], ch["theta"])]
    on = max((v for v in allv.values() if v["relevance"]), key=lambda v: v["accuracy"])
    return dict(valAcc=pct(best["accuracy"]), valAccRel=pct(on["accuracy"]), nValConfigs=len(allv))


def numbers():
    """Every number quoted in the text, as LaTeX macros, so the prose cannot drift from the results."""
    m = load("main"); eq = load("equivalence"); ch = json.load(open(f"{OUT}/chosen.json")); t = load("teaching")
    ab = load("ablation"); cur = load("curve"); lo = load("lofo"); tr = load("transfer"); fs = load("few_shot")
    rb = load("robust"); cal = load("calibration"); calo = load("calibration_original"); ood = load("ood"); fa = load("failures")
    lat = m["latent"]
    g = lambda s, grp, k="accuracy": s["groups"][grp][k]
    sep = [f for f in E.FAMILIES if E.GROUP[f] != "whole"]
    fsmean = lambda k, fams: np.mean([np.mean([r["accuracy"] for r in fs[f][str(k)]]) for f in fams])
    units = sum(fa[f]["units"] for f in sep); found = sum(fa[f]["found"] for f in sep)
    arr_units = sum(fa[f]["units"] for f in E.ARRANGED); arr_missed = sum(fa[f]["units"] - fa[f]["found"] for f in E.ARRANGED)
    mac = dict(nFamilies=len(E.FAMILIES), nWholeFam=len(E.WHOLE), nTraits=26, nTrainPerFamily=40, nTrainScenes=40 * len(E.FAMILIES),
               nTestPerFamily=25, nTestScenes=25 * len(E.FAMILIES), nValPerFamily=12, nRobustPerFamily=12,
               accAll=pct(lat["accuracy"]), pqAll=pct(lat["pq"]), accLo=pct(lat["ci_accuracy"][0]), accHi=pct(lat["ci_accuracy"][1]),
               accSingle=pct(g(lat, "single")), accArranged=pct(g(lat, "arranged")), accWhole=pct(g(lat, "whole")),
               pqArranged=pct(g(lat, "arranged", "pq")), unitRecall=pct(lat["unit_recall"]), aucAll=num(lat["auc"], 3),
               fpLatent=num(lat["false_per_scene"]), fpLogistic=num(m["logistic"]["false_per_scene"]),
               splitWholeLatent=pct(lat["split_whole"]), splitWholeLogistic=pct(m["logistic"]["split_whole"]),
               accLogistic=pct(m["logistic"]["accuracy"]), pqLogistic=pct(m["logistic"]["pq"]), accEdge=pct(m["edge_rule"]["accuracy"]),
               accNever=pct(m["never_split"]["accuracy"]), pqNever=pct(m["never_split"]["pq"]),
               accDefault=pct(m["latent_default"]["accuracy"]), accRelOn=pct(m["latent_relevance_flipped"]["accuracy"]),
               accNoAbst=pct(m["latent_no_abstention"]["accuracy"]),
               accCutsOnlySingle=pct(g(m["latent_cuts_only"], "single")), arrCutsOnly=pct(g(m["latent_cuts_only"], "arranged")),
               accRegionsOnlySingle=pct(g(m["latent_regions_only"], "single")), arrRegionsOnly=pct(g(m["latent_regions_only"], "arranged")),
               arrCarriedSeven=pct(g(ab["carried_0.7"], "arranged")),
               curveOne=pct(np.mean([r["accuracy"] for r in cur["1"]])), curveForty=pct(np.mean([r["accuracy"] for r in cur["40"]])),
               lofoMean=pct(np.mean([lo[f]["latent"]["accuracy"] for f in E.FAMILIES])),
               lofoMeanLogistic=pct(np.mean([lo[f]["logistic"]["accuracy"] for f in E.FAMILIES])),
               lofoSingle=pct(np.mean([lo[f]["latent"]["accuracy"] for f in E.SINGLE])),
               lofoArranged=pct(np.mean([lo[f]["latent"]["accuracy"] for f in E.ARRANGED])),
               lofoWhole=pct(np.mean([lo[f]["latent"]["accuracy"] for f in E.WHOLE])),
               trSA=pct(tr["single_to_arranged"]["accuracy"]), trAS=pct(tr["arranged_to_single"]["accuracy"]),
               trSW=pct(tr["single_only_to_whole"]["accuracy"]), trAW=pct(tr["arranged_only_to_whole"]["accuracy"]),
               fsZero=pct(fsmean(0, sep)), fsTen=pct(fsmean(10, sep)), fsZeroSingle=pct(fsmean(0, E.SINGLE)), fsTenSingle=pct(fsmean(10, E.SINGLE)),
               fsZeroArranged=pct(fsmean(0, E.ARRANGED)), fsTenArranged=pct(fsmean(10, E.ARRANGED)),
               robRotNinety=pct(rb["rot90"]["latent"]["accuracy"]), robRotOneEighty=pct(rb["rot180"]["latent"]["accuracy"]),
               robScale=pct(rb["scale0.7"]["latent"]["accuracy"]), robRecolour=pct(rb["recolour"]["latent"]["accuracy"]),
               robDark=pct(rb["dark"]["latent"]["accuracy"]), robNoise=pct(rb["noise"]["latent"]["accuracy"]), robNoiseLogistic=pct(rb["noise"]["logistic"]["accuracy"]),
               eceChosen=num(cal["ece"], 3), eceOriginal=num(calo["ece"], 3), baseRate=f"{100 * cal['base_rate']:.1f}",
               oodRatio=num(np.median([ood[f]["held_out"]["nearest"] / max(ood[f]["seen"]["nearest"], 1e-9) for f in E.FAMILIES]), 1),
               unitsFound=pct(found / units), arrMissed=arr_missed, arrUnits=arr_units,
               missNoCand=pct(sum(fa[f]["no_candidate"] for f in E.ARRANGED) / max(arr_missed, 1)),
               missNotCalled=pct(sum(fa[f]["not_called"] for f in E.ARRANGED) / max(arr_missed, 1)),
               missBadly=pct(sum(fa[f]["called_badly"] for f in E.ARRANGED) / max(arr_missed, 1)),
               relTop=", ".join(n.replace("_", r"\_") for n, _ in fa["relevance_top"][:4]),
               curveOneSingle=pct(np.mean([r["groups"]["single"]["accuracy"] for r in cur["1"]])), curveFortySingle=pct(cur["40"][0]["groups"]["single"]["accuracy"]),
               curveOneArranged=pct(np.mean([r["groups"]["arranged"]["accuracy"] for r in cur["1"]])), curveFortyArranged=pct(cur["40"][0]["groups"]["arranged"]["accuracy"]),
               curveOneWhole=pct(np.mean([r["groups"]["whole"]["accuracy"] for r in cur["1"]])), curveFortyWhole=pct(cur["40"][0]["groups"]["whole"]["accuracy"]),
               fsOne=pct(fsmean(1, sep)), fsTwo=pct(fsmean(2, sep)), fsFive=pct(fsmean(5, sep)),
               secPerScene="0.3", candPerScene="170",
               missNoCandShelf=fa["bookshelf"]["no_candidate"], missNoCandCabinet=fa["cabinet"]["no_candidate"],
               maxBooks=max(len(r["units"]) for r in E.probes("bookshelf")),
               nEquivAgree=eq["agree"], nEquivTotal=eq["total"],
               chosenW=ch["w_carried"], chosenK=ch["k_nn"], chosenTheta=ch["theta"],
               nEntries=f"{sum(v['entries'] for v in t.values()):,}".replace(",", "{,}"), nPositives=sum(v["positives"] for v in t.values()),
               **validation_numbers())
    groups = [("shape", "shape"), ("contact", "contact"), ("signal", "signal"), ("where", "position")]
    worst = min(groups, key=lambda kv: ab[f"without_{kv[0]}"]["accuracy"])
    best_only = max(groups, key=lambda kv: ab[f"only_{kv[0]}"]["accuracy"])
    k96 = ab["k_nn_96"]
    mac["ablationText"] = (f"Every trait group contributes: removing the {worst[1]} traits costs most (exact {pct(ab['without_' + worst[0]]['accuracy'])}\\% against {pct(lat['accuracy'])}\\%), "
                           f"and no group alone comes close (the best, {best_only[1]}, reaches {pct(ab['only_' + best_only[0]]['accuracy'])}\\%). "
                           f"The neighbourhood size matters most among the settings: at $k=96$ accuracy falls to {pct(k96['accuracy'])}\\% with {num(k96['false_per_scene'])} false segments per scene, "
                           f"because with carried-whole entries counting almost nothing a large neighbourhood nearly always contains some part that came away. "
                           f"Raising the carried-whole weight trades recall for caution: at $w_0=1$ false segments fall to {num(ab['carried_1.0']['false_per_scene'])} per scene "
                           f"but unit recall falls to {pct(ab['carried_1.0']['unit_recall'])}\\% and accuracy to {pct(ab['carried_1.0']['accuracy'])}\\%.")
    with open(f"{TAB}/numbers.tex", "w") as fh:
        for k, v in mac.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


if __name__ == "__main__":
    for fn in (numbers, fig_scenes, fig_curve, fig_lofo, fig_few_shot, fig_calibration, fig_latent, fig_qualitative,
               tab_main, tab_ablation, tab_robust, tab_teaching, tab_families):
        try:
            fn()
            print("ok", fn.__name__)
        except (OSError, KeyError) as e:
            print("skip", fn.__name__, type(e).__name__, e)
