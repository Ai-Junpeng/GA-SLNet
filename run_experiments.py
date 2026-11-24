"""Run the experiment grid. Idempotent: a config with an existing .pt is skipped.

  python run_experiments.py --stage main
  python run_experiments.py --stage ablation
  python run_experiments.py --stage cross
  python run_experiments.py --stage eval
"""
import argparse
import os
import time

import torch

import config as C
import evaluate as E
import train as T

MAIN = [
    ("gaslnet", "gaslnet", {}, "nll"),
    ("lstm", "lstm", {}, "nll"),
    ("gru", "gru", {}, "nll"),
    ("seq2seq", "seq2seq", {}, "nll"),
    ("transformer", "transformer", {}, "nll"),
    ("cfc", "cfc", {}, "nll"),
    ("ltc", "ltc", {}, "nll"),
    ("ode", "ode", {}, "nll"),
]

ABL = [
    ("gaslnet_nodt", "gaslnet", dict(use_dt=False), "nll"),
    ("gaslnet_noattr", "gaslnet", dict(use_attr=False), "nll"),
    ("gaslnet_k1", "gaslnet", dict(use_switch=False), "nll"),
    ("gaslnet_k2", "gaslnet", dict(K=2), "nll"),
    ("gaslnet_k4", "gaslnet", dict(K=4), "nll"),
    ("gaslnet_wta", "gaslnet", {}, "wta"),
]

CROSS = [("gaslnet_gulf", "gaslnet", {}, "nll"), ("lstm_gulf", "lstm", {}, "nll")]


def do_train(group):
    for item in group:
        tag, name, kw, loss = item
        pt = os.path.join(C.OUT_DIR, f"{tag}.pt")
        if os.path.exists(pt):
            print(f"[skip] {tag} (exists)", flush=True)
            continue
        print(f"\n=== train {tag} ===", flush=True)
        regions = ["Gulf"] if tag.endswith("_gulf") else None
        T.train_one(name, tag, kw=kw, loss=loss, train_regions=regions)


def do_eval():
    res = {}
    res["main_metrics"] = {}
    av = {t: os.path.exists(os.path.join(C.OUT_DIR, f"{t}.pt")) for t, *_ in MAIN}
    res["dataset"] = dict(train_samples=int(len(__import__("dataset").AISDataset(
        os.path.join(C.PROC_DIR, "train.npz"), split="train"))),
        test_samples=int(len(__import__("dataset").AISDataset(
            os.path.join(C.PROC_DIR, "eval.npz"), split="test",
            region_filter=C.TRAIN_REGIONS))))
    res["cv_baseline"] = E.cv_baseline("test", C.TRAIN_REGIONS)

    print("\n--- main ---", flush=True)
    for tag, _, _, _ in MAIN:
        if not av[tag]:
            continue
        m = E.evaluate(tag, split="test", region=C.TRAIN_REGIONS)
        m.update(E.efficiency(tag))
        res["main_metrics"][tag] = m
        print(f"{tag:13s} ADE={m['ADE']:7.1f} FDE={m['FDE']:7.1f} NLL={m['NLL']:6.3f} "
              f"P={m['params']:6d} lat={m['latency_ms']:.1f}ms", flush=True)
        E.save_results(res)

    print("\n--- ablation ---", flush=True)
    res["abl_metrics"] = {}
    for tag, _, _, _ in ABL:
        if not os.path.exists(os.path.join(C.OUT_DIR, f"{tag}.pt")):
            continue
        m = E.evaluate(tag, split="test", region=C.TRAIN_REGIONS)
        m.update(E.efficiency(tag))
        res["abl_metrics"][tag] = m
        print(f"{tag:15s} ADE={m['ADE']:7.1f} FDE={m['FDE']:7.1f} NLL={m['NLL']:6.3f}", flush=True)
        E.save_results(res)

    print("\n--- cross-region ---", flush=True)
    res["cross_metrics"] = {}
    for tag, _, _, _ in CROSS:
        if not os.path.exists(os.path.join(C.OUT_DIR, f"{tag}.pt")):
            continue
        m = E.evaluate(tag, split="test", region=[C.CROSS_REGION])
        res["cross_metrics"][tag] = m
        print(f"{tag:13s} Pacific ADE={m['ADE']:7.1f} FDE={m['FDE']:7.1f}", flush=True)
        E.save_results(res)

    print("\n--- gap sweep ---", flush=True)
    res["gap"] = {}
    for tag, _, _, _ in MAIN:
        if not av[tag]:
            continue
        row = {}
        for g in C.GAP_SWEEP:
            m = E.evaluate(tag, split="test", region=C.TRAIN_REGIONS, gap=float(g))
            row[str(g)] = dict(ADE=m["ADE"], FDE=m["FDE"], NLL=m["NLL"], CRPS=m["CRPS"])
        res["gap"][tag] = row
        print(f"{tag:13s} " + "  ".join(f"g{g}:{row[str(g)]['ADE']:.0f}" for g in C.GAP_SWEEP),
              flush=True)
        E.save_results(res)

    print("\n--- ablation gap sweep ---", flush=True)
    res["abl_gap"] = {}
    for tag, _, _, _ in ABL:
        if not os.path.exists(os.path.join(C.OUT_DIR, f"{tag}.pt")):
            continue
        row = {}
        for g in C.GAP_SWEEP:
            m = E.evaluate(tag, split="test", region=C.TRAIN_REGIONS, gap=float(g))
            row[str(g)] = dict(ADE=m["ADE"], FDE=m["FDE"])
        res["abl_gap"][tag] = row
        print(f"{tag:15s} " + "  ".join(f"g{g}:{row[str(g)]['ADE']:.0f}" for g in C.GAP_SWEEP),
              flush=True)
        E.save_results(res)
    E.save_results(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    a = ap.parse_args()
    torch.manual_seed(C.SEED)
    t0 = time.time()
    if a.stage == "main":
        do_train(MAIN)
    elif a.stage == "ablation":
        do_train(ABL)
    elif a.stage == "cross":
        do_train(CROSS)
    elif a.stage == "eval":
        do_eval()
    else:
        do_train(MAIN); do_train(ABL); do_train(CROSS); do_eval()
    print(f"\nstage {a.stage} done in {time.time()-t0:.0f}s", flush=True)
