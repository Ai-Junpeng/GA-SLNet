"""Paper figures + markdown tables from results.json."""
import json
import os

import numpy as np

import config as C
import evaluate as E

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 140, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})

ORDER = ["gaslnet", "cfc", "ltc", "ode", "lstm", "gru", "seq2seq", "transformer"]
PRETTY = {"gaslnet": "GA-SLNet", "cfc": "CfC (fixed-$\\Delta t$)", "ltc": "LTC",
          "ode": "Neural ODE", "lstm": "LSTM", "gru": "GRU", "seq2seq": "Seq2Seq",
          "transformer": "Transformer",
          "gaslnet_nodt": "w/o $\\Delta t$ conditioner", "gaslnet_noattr": "w/o attribute conditioning",
          "gaslnet_k1": "single expert (K=1)", "gaslnet_k2": "K=2", "gaslnet_k4": "K=4",
          "gaslnet_wta": "WTA loss instead of NLL"}
COL = {"gaslnet": "#c0392b"}


def load():
    with open(os.path.join(C.OUT_DIR, "results.json")) as f:
        return json.load(f)


def fig_main(res):
    mm = res.get("main_metrics", {})
    tags = [t for t in ORDER if t in mm]
    ade = [mm[t]["ADE"] for t in tags]
    fde = [mm[t]["FDE"] for t in tags]
    x = np.arange(len(tags))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.4, 3.1))
    b1 = ax.bar(x - w / 2, ade, w, label="ADE", color="#8ea9c1")
    b2 = ax.bar(x + w / 2, fde, w, label="FDE@30min", color="#4a6d8c")
    for i, t in enumerate(tags):
        if t == "gaslnet":
            b1[i].set_color("#c0392b"); b2[i].set_color("#7b241c")
    cv = res.get("cv_baseline")
    if cv:
        ax.axhline(cv["ADE"], ls=":", lw=1.2, color="k")
        ax.text(-0.45, cv["ADE"] * 1.02, "constant velocity", va="bottom", ha="left",
                fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([PRETTY.get(t, t) for t in tags], rotation=18,
                                         ha="right", fontsize=8)
    ax.set_ylabel("displacement error (m)")
    ax.set_title("Main comparison on the held-out test set (no injected gap)")
    ax.set_ylim(0, max(fde) * 1.08)
    for i, v in enumerate(ade):
        ax.text(i - w / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "fig_main.png"))
    plt.close(fig)


def fig_gap(res):
    gp = res.get("gap", {})
    tags = [t for t in ORDER if t in gp]
    gaps = [int(g) for g in C.GAP_SWEEP]
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.3))
    for t in tags:
        y = [gp[t][str(g)]["ADE"] for g in gaps]
        lw = 2.4 if t == "gaslnet" else 1.2
        c = COL.get(t)
        ax[0].plot(np.array(gaps) + 1, y, "o-", ms=4, lw=lw, label=PRETTY.get(t, t),
                   color=c, zorder=5 if t == "gaslnet" else 2)
        y2 = [gp[t][str(g)]["FDE"] for g in gaps]
        ax[1].plot(np.array(gaps) + 1, y2, "o-", ms=4, lw=lw, label=PRETTY.get(t, t),
                   color=c, zorder=5 if t == "gaslnet" else 2)
    for a, lab in zip(ax, ["ADE (m)", "FDE@30min (m)"]):
        a.set_xscale("log"); a.set_xlabel("injected blackout length (min)")
        a.set_ylabel(lab); a.set_xticks(np.array(gaps) + 1)
        a.set_xticklabels([str(g) for g in gaps])
        a.set_title(lab.split(" ")[0] + " degradation under blackout")
    ax[0].legend(frameon=False, fontsize=7.2, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "fig_gap.png"))
    plt.close(fig)


def fig_ablation(res):
    am = res.get("abl_metrics", {})
    base = res.get("main_metrics", {}).get("gaslnet")
    if not am or not base:
        return
    keys = ["gaslnet_nodt", "gaslnet_noattr", "gaslnet_k1", "gaslnet_wta", "gaslnet_k2",
            "gaslnet_k4"]
    keys = [k for k in keys if k in am]
    labels = [PRETTY.get(k, k) for k in keys]
    d_ade = [am[k]["ADE"] - base["ADE"] for k in keys]
    d_fde = [am[k]["FDE"] - base["FDE"] for k in keys]

    fig, ax = plt.subplots(1, 2, figsize=(10.6, 3.4),
                           gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(keys))[::-1]
    ax[0].barh(y + 0.19, d_ade, 0.36, label="$\\Delta$ADE", color="#c0392b")
    ax[0].barh(y - 0.19, d_fde, 0.36, label="$\\Delta$FDE@30", color="#e59866")
    ax[0].axvline(0, color="k", lw=1)
    ax[0].set_yticks(y); ax[0].set_yticklabels(labels, fontsize=8.5)
    ax[0].set_xlabel("error increase vs. full GA-SLNet (m)")
    ax[0].set_title(f"Component ablation (full model ADE={base['ADE']:.0f} m)")
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")
    for yy, v in zip(y, d_ade):
        ax[0].text(v, yy + 0.19, f" {v:+.0f}", va="center",
                   ha="left" if v >= 0 else "right", fontsize=7)
    for yy, v in zip(y, d_fde):
        ax[0].text(v, yy - 0.19, f" {v:+.0f}", va="center",
                   ha="left" if v >= 0 else "right", fontsize=7)

    ag = res.get("abl_gap", {})
    gaps = [int(g) for g in C.GAP_SWEEP]
    series = [("gaslnet", "full GA-SLNet", "#c0392b", 2.4)] + \
             [(k, PRETTY.get(k, k), None, 1.3) for k in keys]
    cmap = plt.get_cmap("tab10")
    for i, (k, lab, c, lw) in enumerate(series):
        if k not in ag:
            continue
        yv = [ag[k][str(g)]["ADE"] for g in gaps]
        ax[1].plot(np.array(gaps) + 1, yv, "o-", ms=3.5, lw=lw,
                   color=c or cmap(i % 10), label=lab)
    ax[1].set_xscale("log"); ax[1].set_xticks(np.array(gaps) + 1)
    ax[1].set_xticklabels([str(g) for g in gaps])
    ax[1].set_xlabel("injected blackout length (min)")
    ax[1].set_ylabel("ADE (m)")
    ax[1].set_title("Ablations under blackout")
    ax[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "fig_ablation.png"))
    plt.close(fig)

    # K sweep curve
    ks, ades = [], []
    for k, tag in ((1, "gaslnet_k1"), (2, "gaslnet_k2"), (3, "gaslnet"), (4, "gaslnet_k4")):
        src = am.get(tag) or res.get("main_metrics", {}).get(tag)
        if src:
            ks.append(k); ades.append(src["ADE"])
    if len(ks) > 1:
        fig, ax = plt.subplots(figsize=(4.0, 2.9))
        ax.plot(ks, ades, "o-", color="#c0392b", ms=6)
        for k, v in zip(ks, ades):
            ax.annotate(f"{v:.0f}", (k, v), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7.5)
        ax.set_xlabel("number of liquid experts $K$")
        ax.set_ylabel("ADE (m)")
        ax.set_xticks(ks)
        ax.set_title("Expert-count sweep")
        fig.tight_layout()
        fig.savefig(os.path.join(C.FIG_DIR, "fig_k_sweep.png"))
        plt.close(fig)


def fig_horizon(res):
    mm = res.get("main_metrics", {})
    tags = [t for t in ORDER if t in mm]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for t in tags:
        ax.plot(C.HORIZONS_MIN, mm[t]["mae_h"], "o-", ms=4, lw=2.2 if t == "gaslnet" else 1.1,
                label=PRETTY.get(t, t), color=COL.get(t))
    ax.set_xlabel("prediction horizon (min)")
    ax.set_ylabel("MAE (m)")
    ax.set_title("Error growth with horizon")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "fig_horizon.png"))
    plt.close(fig)


def table_main(res):
    mm = res.get("main_metrics", {})
    tags = [t for t in ORDER if t in mm]
    lines = ["| Model | ADE (m) | FDE@30 (m) | minADE (m) | NLL | CRPS | ECE | Params | Latency (ms) |",
             "|---|---|---|---|---|---|---|---|---|"]
    cv = res.get("cv_baseline")
    if cv:
        lines.append(f"| Constant velocity | {cv['ADE']:.1f} | {cv['FDE']:.1f} | – | – | – | – | – | – |")
    for t in tags:
        m = mm[t]
        lines.append(f"| {PRETTY.get(t,t)} | {m['ADE']:.1f} | {m['FDE']:.1f} | {m['minADE']:.1f} | "
                     f"{m['NLL']:.3f} | {m['CRPS']:.0f} | {m['ECE']:.3f} | {m['params']/1000:.1f}k | "
                     f"{m['latency_ms']:.1f} |")
    return "\n".join(lines)


def table_abl(res):
    am = res.get("abl_metrics", {})
    base = res.get("main_metrics", {}).get("gaslnet")
    if not base:
        return ""
    keys = ["gaslnet_nodt", "gaslnet_noattr", "gaslnet_k1", "gaslnet_wta", "gaslnet_k2", "gaslnet_k4"]
    lines = ["| Variant | ADE (m) | ΔADE | FDE@30 (m) | ΔFDE | ADE@30min-gap | Δ@gap |",
             "|---|---|---|---|---|---|---|"]
    ag = res.get("abl_gap", {})
    g0 = (res.get("gap", {}).get("gaslnet", {}) or {}).get("30", {}).get("ADE")
    lines.append(f"| **Full GA-SLNet** | {base['ADE']:.1f} | – | {base['FDE']:.1f} | – | "
                 f"{g0:.1f} | – |" if g0 else
                 f"| **Full GA-SLNet** | {base['ADE']:.1f} | – | {base['FDE']:.1f} | – | – | – |")
    for k in keys:
        if k not in am:
            continue
        m = am[k]
        gk = ag.get(k, {}).get("30", {}).get("ADE")
        da = m["ADE"] - base["ADE"]
        df = m["FDE"] - base["FDE"]
        dg = (gk - g0) if (gk is not None and g0 is not None) else None
        lines.append(f"| {PRETTY.get(k,k)} | {m['ADE']:.1f} | {da:+.1f} | {m['FDE']:.1f} | "
                     f"{df:+.1f} | {gk:.1f} | {dg:+.1f} |" if gk else
                     f"| {PRETTY.get(k,k)} | {m['ADE']:.1f} | {da:+.1f} | {m['FDE']:.1f} | {df:+.1f} | – | – |")
    return "\n".join(lines)


def table_gap(res):
    gp = res.get("gap", {})
    tags = [t for t in ORDER if t in gp]
    head = "| Model | " + " | ".join(f"gap={g}" for g in C.GAP_SWEEP) + " | Δ(60−0) |"
    lines = [head, "|---" * (len(C.GAP_SWEEP) + 2) + "|"]
    for t in tags:
        vals = [gp[t][str(g)]["ADE"] for g in C.GAP_SWEEP]
        lines.append(f"| {PRETTY.get(t,t)} | " + " | ".join(f"{v:.0f}" for v in vals) +
                     f" | {vals[-1]-vals[0]:+.0f} |")
    return "\n".join(lines)


def all_figs():
    res = load()
    fig_main(res); fig_gap(res); fig_ablation(res); fig_horizon(res)
    out = dict(main=table_main(res), abl=table_abl(res), gap=table_gap(res))
    with open(os.path.join(C.OUT_DIR, "tables.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(table_main(res)); print(); print(table_abl(res)); print(); print(table_gap(res))
    return out


if __name__ == "__main__":
    all_figs()
