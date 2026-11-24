"""Figures + markdown report for the v2 blackout experiments."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as C
import train2 as T

GAP_SWEEP = [0, 5, 10, 15, 20, 30]
COLORS = dict(gaslnet2="#d62728", transformer2="#1f77b4", lstm2="#2ca02c",
              gru2="#8c564b", seq2seq2="#9467bd", cfc2="#ff7f0e",
              ltc2="#17becf", ode2="#7f7f7f")
NICE = dict(gaslnet2="GA-SLNet (ours)", transformer2="Transformer", lstm2="LSTM",
            gru2="GRU", seq2seq2="Seq2Seq", cfc2="CfC (fixed dt)",
            ltc2="LTC", ode2="Neural ODE")


def _load():
    return T.load_results2()


def fig_gap(res):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, met, ttl in zip(axes, ("ADE", "MedADE", "NLL"),
                            ("ADE vs blackout", "Median ADE vs blackout", "NLL vs blackout")):
        for tag, row in res["main"].items():
                if tag not in COLORS:
                    continue
                gs = [g for g in GAP_SWEEP if str(g) in row]
                v = [row[str(g)][met] for g in gs]
                lw = 2.8 if tag == "gaslnet2" else 1.3
                ax.plot(gs, v, marker="o", ms=4, lw=lw, color=COLORS[tag],
                        label=NICE.get(tag, tag),
                        zorder=5 if tag == "gaslnet2" else 3)
        ax.set_xlabel("injected blackout [min]")
        ax.set_ylabel(met if met != "NLL" else "NLL (nats)")
        ax.set_title(ttl)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "fig2_gap.png"), dpi=160)
    plt.close(fig)


def fig_main(res):
    tags = [t for t in res["main"] if "0" in res["main"][t]]
    tags.sort(key=lambda t: res["main"][t]["0"]["ADE"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, gap in zip(axes, ("0", "30")):
        names = [NICE.get(t, t) for t in tags]
        ade = [res["main"][t][gap]["ADE"] for t in tags]
        cols = [COLORS.get(t, "#333") for t in tags]
        ax.barh(names, ade, color=cols)
        ax.invert_yaxis()
        ax.set_xlabel("ADE [m]")
        ax.set_title(f"blackout = {gap} min")
        for i, v in enumerate(ade):
            ax.text(v, i, f" {v:.0f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "fig2_main.png"), dpi=160)
    plt.close(fig)


def md_table(res):
    L = []
    L.append("| model | ADE@0 | FDE@0 | MedADE@0 | NLL@0 | ADE@30 | MedADE@30 | NLL@30 | params | lat ms |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for tag, row in res["main"].items():
        if "0" not in row:
            continue
        r0, r30 = row["0"], row["30"]
        L.append(f"| {NICE.get(tag, tag)} | {r0['ADE']:.0f} | {r0['FDE']:.0f} | "
                 f"{r0['MedADE']:.0f} | {r0['NLL']:.3f} | {r30['ADE']:.0f} | "
                 f"{r30['MedADE']:.0f} | {r30['NLL']:.3f} | "
                 f"{row.get('params', 0)/1e3:.1f}k | {row.get('latency_ms', 0):.1f} |")
    return "\n".join(L)


def main():
    res = _load()
    if "main" in res:
        fig_gap(res)
        fig_main(res)
    rep = ["# GA-SLNet v2 — blackout prediction results\n"]
    if "dataset" in res:
        rep.append(f"train windows: {res['dataset']['train_samples']}, "
                   f"test windows: {res['dataset']['test_samples']}\n")
    if "cv_baseline" in res:
        rep.append("## CV baseline\n")
        for g, r in res["cv_baseline"].items():
            rep.append(f"- gap {g}: ADE {r['ADE']:.0f} m, FDE {r['FDE']:.0f} m, "
                       f"MedADE {r['MedADE']:.0f} m")
    if "main" in res:
        rep.append("\n## main\n")
        rep.append(md_table(res))
    if "ablation" in res:
        rep.append("\n## ablation\n")
        rep.append("| variant | ADE@0 | ADE@15 | ADE@30 | NLL@0 |")
        rep.append("|---|---|---|---|---|")
        for tag, row in res["ablation"].items():
            rep.append(f"| {tag} | {row['0']['ADE']:.0f} | {row['15']['ADE']:.0f} | "
                       f"{row['30']['ADE']:.0f} | {row['0']['NLL']:.3f} |")
    if "seeds" in res:
        rep.append("\n## seeds\n")
        for tag, row in res["seeds"].items():
            rep.append(f"- {tag}: ADE@0 {row['0']['ADE']:.0f}, ADE@30 {row['30']['ADE']:.0f}, "
                       f"MedADE@0 {row['0']['MedADE']:.0f}, NLL@0 {row['0']['NLL']:.3f}")
    if "cross" in res:
        rep.append("\n## cross-region (train Gulf -> test Pacific)\n")
        for tag, row in res["cross"].items():
            rep.append(f"- {tag}: g0 ADE {row['g0']['ADE']:.0f} / g30 ADE {row['g30']['ADE']:.0f}")
    with open(os.path.join(C.OUT_DIR, "report2.md"), "w") as f:
        f.write("\n".join(rep))
    print("report2.md + fig2_*.png written")


if __name__ == "__main__":
    main()
