"""Interpretability + qualitative analyses for GA-SLNet."""
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as C
import dataset as D
import evaluate as E
import models as M

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})

H = C.HORIZONS_MIN


def tau_vs_gap(tag="gaslnet", n=256):
    model, _ = E.load(tag)
    ds = D.AISDataset(os.path.join(C.PROC_DIR, "eval.npz"), split="test",
                      region_filter=C.TRAIN_REGIONS)
    dl = DataLoader(ds, batch_size=256, shuffle=False)
    b = next(iter(dl))
    b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
    L = int(b["irr_mask"].sum(1).max())
    f = b["irr"][:, :L]
    mask = b["irr_mask"][:, :L]
    attr = b["attr"]
    gaps = [1, 2, 5, 10, 15, 30, 60, 120]
    taus, gates = [], []
    with torch.no_grad():
        for g in gaps:
            dt = b["irr_dt"][:, :L].clone()
            dt[:, -1] = float(g)                     # last interval = blackout length
            model(f, dt, mask, attr)
            taus.append(model.last_tau[:, -1, :].mean(0).cpu().numpy())
            gates.append(model.last_gate[:, -1, :].mean(0).cpu().numpy())
    taus = np.stack(taus)          # gaps, K
    gates = np.stack(gates)        # gaps, K
    np.savez(os.path.join(C.OUT_DIR, "tau_gap.npz"), gaps=gaps, taus=taus, gates=gates)

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.2))
    for k in range(taus.shape[1]):
        ax[0].plot(gaps, taus[:, k], "o-", ms=4, label=f"expert {k+1}")
    ax[0].set_xscale("log"); ax[0].set_xlabel("blackout length $\\Delta t$ (min)")
    ax[0].set_ylabel("learned time constant $\\tau$ (min)")
    ax[0].set_title("(a) $\\tau$ adapts to the gap")
    ax[0].legend(frameon=False, fontsize=8)
    bottom = np.zeros(len(gaps))
    for k in range(gates.shape[1]):
        ax[1].bar(range(len(gaps)), gates[:, k], bottom=bottom, label=f"expert {k+1}",
                  width=0.7)
        bottom += gates[:, k]
    ax[1].set_xticks(range(len(gaps))); ax[1].set_xticklabels(gaps)
    ax[1].set_xlabel("blackout length $\\Delta t$ (min)"); ax[1].set_ylabel("gate weight $\\pi$")
    ax[1].set_title("(b) gating shifts with the gap")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "fig_tau_gap.png"))
    plt.close(fig)
    return gaps, taus, gates


def gate_on_maneuver(tag="gaslnet", n=400):
    model, _ = E.load(tag)
    ds = D.AISDataset(os.path.join(C.PROC_DIR, "eval.npz"), split="test",
                      region_filter=C.TRAIN_REGIONS)
    ds.idx = ds.idx[:n]
    dl = DataLoader(ds, batch_size=n, shuffle=False)
    b = next(iter(dl))
    b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
    L = int(b["irr_mask"].sum(1).max())
    with torch.no_grad():
        model(b["irr"][:, :L], b["irr_dt"][:, :L], b["irr_mask"][:, :L], b["attr"])
    gate = model.last_gate.cpu().numpy()            # n,T,K
    turn = b["irr"][:, :L, 7].cpu().numpy()          # normalised turn rate
    msk = b["irr_mask"][:, :L].cpu().numpy()
    # average |turn| per sample in the second half
    score = (np.abs(turn) * msk)[:, L // 2:].sum(1) / np.clip(msk[:, L // 2:].sum(1), 1, None)
    order = np.argsort(-score)
    hi = order[: n // 5]
    lo = order[-n // 5:]
    tt = np.arange(L)
    np.savez(os.path.join(C.OUT_DIR, "gate_maneuver.npz"),
             gate_hi=gate[hi].mean(0), gate_lo=gate[lo].mean(0),
             turn_hi=np.abs(turn[hi]).mean(0) * msk[hi].mean(0),
             turn_lo=np.abs(turn[lo]).mean(0) * msk[lo].mean(0))

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.0), sharey=True)
    for a, g, lab in ((ax[0], gate[hi].mean(0), "turning segments"),
                      (ax[1], gate[lo].mean(0), "straight segments")):
        for k in range(g.shape[1]):
            a.plot(tt, g[:, k], label=f"expert {k+1}", lw=1.4)
        a.set_title(lab); a.set_xlabel("time step in observation window")
        a.legend(frameon=False, fontsize=8)
    ax[0].set_ylabel("gate weight $\\pi$")
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "fig_gate_maneuver.png"))
    plt.close(fig)


def qualitative(tags=("gaslnet", "cfc", "lstm"), k=4, gap=10.0):
    ds = D.AISDataset(os.path.join(C.PROC_DIR, "eval.npz"), split="test",
                      region_filter=C.TRAIN_REGIONS, gap_min=gap)
    picks = ds.idx[:k]
    items = [ds[i] for i in range(k)]
    b = torch.utils.data.default_collate(items)
    bc = {kk: (v.cuda() if torch.is_tensor(v) else v) for kk, v in b.items()}
    preds, names = {}, []
    for tag in tags:
        model, name = E.load(tag)
        v = M.view_of(name)
        with torch.no_grad():
            y, lg, _ = model(bc[v], bc[f"{v}_dt"], bc[f"{v}_mask"], bc["attr"])
        best = lg.argmax(1) if lg is not None else torch.zeros(k, dtype=torch.long)
        mu = y[torch.arange(k), best].cpu().numpy()
        preds[tag] = mu
        names.append(tag)

    fig, axes = plt.subplots(1, k, figsize=(3.0 * k, 3.1))
    truth_km = b["y"].numpy()                        # km
    for j in range(k):
        ax = axes[j]
        raw = b["irr"][j].numpy()
        mask = b["irr_mask"][j].numpy()
        e = raw[mask, 0] - raw[mask, 0][-1]
        n = raw[mask, 1] - raw[mask, 1][-1]
        ax.plot(e / 1000, n / 1000, "o-", ms=2.2, lw=1.0, color="#4c72b0", label="observed")
        ax.plot(np.r_[0, truth_km[j, :, 0]], np.r_[0, truth_km[j, :, 1]], "s-", ms=3.2,
                lw=1.6, color="#2a9d4b", label="ground truth")
        for tag, c in zip(names, ["#d1495b", "#e8a33d", "#7b5ea7"]):
            p = preds[tag][j]
            ax.plot(np.r_[0, p[:, 0]], np.r_[0, p[:, 1]], "--", lw=1.5, color=c, label=tag)
        ax.scatter([0], [0], marker="*", s=110, color="k", zorder=5)
        ax.set_title(f"sample {j+1}  (gap {gap:.0f} min)", fontsize=9)
        ax.set_xlabel("East displacement (km)")
        if j == 0:
            ax.set_ylabel("North displacement (km)")
        ax.set_aspect("equal", adjustable="datalim")
        if j == 0:
            ax.legend(frameon=False, fontsize=7.5, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "fig_qualitative.png"))
    plt.close(fig)
    return preds


if __name__ == "__main__":
    if os.path.exists(os.path.join(C.OUT_DIR, "gaslnet.pt")):
        tau_vs_gap()
        gate_on_maneuver()
        qualitative()
        print("interpret done")
