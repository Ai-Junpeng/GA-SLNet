"""Experiment orchestration v2. Idempotent: existing .pt is skipped.

  python run2.py --stage quick     # gaslnet2 + 3 baselines, eval gaps {0,15,30}
  python run2.py --stage main      # all 8 models
  python run2.py --stage ablation  # component ablations of gaslnet2
  python run2.py --stage seeds     # 3 seeds for gaslnet2 / transformer2 / lstm2
  python run2.py --stage eval      # full evaluation -> results2.json
"""
import argparse
import os
import time

import torch

import config as C
import train2 as T

QUICK = [
    ("gaslnet2", "gaslnet2", {}, "nll"),
    ("lstm2", "lstm2", {}, "nll"),
    ("transformer2", "transformer2", {}, "nll"),
    ("cfc2", "cfc2", {}, "nll"),
]

MAIN = QUICK + [
    ("gru2", "gru2", {}, "nll"),
    ("seq2seq2", "seq2seq2", {}, "nll"),
    ("ltc2", "ltc2", {}, "nll"),
    ("ode2", "ode2", {}, "nll"),
]

ABL = [
    ("gaslnet2_nodt", "gaslnet2", dict(use_dt=False), "nll"),
    ("gaslnet2_noprop", "gaslnet2", dict(use_prop=False), "nll"),
    ("gaslnet2_noattr", "gaslnet2", dict(use_attr=False), "nll"),
    ("gaslnet2_k1", "gaslnet2", dict(use_switch=False), "nll"),
    ("gaslnet2_k4", "gaslnet2", dict(K=4), "nll"),
    ("gaslnet2_wta", "gaslnet2", {}, "wta"),
]

SEEDS = [2020, 7, 3407]
SEED_MODELS = ["gaslnet2", "transformer2", "lstm2"]

GAP_SWEEP = [0, 5, 10, 15, 20, 30]


def do_train(group, epochs=C.EPOCHS):
    for tag, name, kw, loss in group:
        pt = os.path.join(C.OUT_DIR, f"{tag}.pt")
        if os.path.exists(pt):
            print(f"[skip] {tag}", flush=True)
            continue
        print(f"\n=== train {tag} ===", flush=True)
        regions = ["Gulf"] if tag.endswith("_gulf") else None
        T.train_one(name, tag, kw=kw, loss=loss, epochs=epochs, train_regions=regions)


def do_seeds():
    for name in SEED_MODELS:
        for s in SEEDS[1:]:
            tag = f"{name}_s{s}"
            if os.path.exists(os.path.join(C.OUT_DIR, f"{tag}.pt")):
                print(f"[skip] {tag}", flush=True)
                continue
            print(f"\n=== train {tag} ===", flush=True)
            T.train_one(name, tag, seed=s)


def eval_row(tag, gaps=GAP_SWEEP):
    row = {}
    for g in gaps:
        m = T.evaluate2(tag, split="test", region=C.TRAIN_REGIONS, gap=float(g))
        row[str(g)] = {k: m[k] for k in ("ADE", "FDE", "MedADE", "P90ADE", "NLL", "CRPS", "ECE")}
    return row


def do_eval(quick=False):
    res = T.load_results2()
    group = QUICK if quick else MAIN
    res["dataset"] = dict(
        train_samples=int(len(T.build_ds("train"))),
        test_samples=int(len(T.build_ds("test", region=C.TRAIN_REGIONS))))
    res["cv_baseline"] = {str(g): T.cv_baseline2("test", C.TRAIN_REGIONS, float(g))
                          for g in (0, 15, 30)}

    print("\n--- main gap sweep ---", flush=True)
    res.setdefault("main", {})
    for tag, *_ in group:
        if not os.path.exists(os.path.join(C.OUT_DIR, f"{tag}.pt")):
            continue
        res["main"][tag] = eval_row(tag)
        res["main"][tag].update(T.efficiency2(tag))
        r0 = res["main"][tag]["0"]; r30 = res["main"][tag]["30"]
        print(f"{tag:16s} g0 ADE={r0['ADE']:6.0f} Med={r0['MedADE']:6.0f} NLL={r0['NLL']:6.3f} | "
              f"g30 ADE={r30['ADE']:6.0f} Med={r30['MedADE']:6.0f} NLL={r30['NLL']:6.3f} | "
              f"P={res['main'][tag]['params']/1e3:.1f}k lat={res['main'][tag]['latency_ms']:.1f}ms",
              flush=True)
        T.save_results2(res)

    if quick:
        T.save_results2(res)
        return

    print("\n--- ablation (gaps 0/15/30) ---", flush=True)
    res.setdefault("ablation", {})
    for tag, *_ in ABL:
        if not os.path.exists(os.path.join(C.OUT_DIR, f"{tag}.pt")):
            continue
        res["ablation"][tag] = eval_row(tag, gaps=(0, 15, 30))
        r0 = res["ablation"][tag]["0"]
        print(f"{tag:18s} g0 ADE={r0['ADE']:6.0f} NLL={r0['NLL']:6.3f}", flush=True)
        T.save_results2(res)

    print("\n--- seeds (gap 0/30) ---", flush=True)
    res.setdefault("seeds", {})
    for name in SEED_MODELS:
        for s in SEEDS:
            tag = name if s == SEEDS[0] else f"{name}_s{s}"
            if not os.path.exists(os.path.join(C.OUT_DIR, f"{tag}.pt")):
                continue
            res["seeds"][tag] = eval_row(tag, gaps=(0, 30))
        T.save_results2(res)

    print("\n--- cross region (train Gulf -> test Pacific) ---", flush=True)
    res.setdefault("cross", {})
    for tag, name, kw, loss in [("gaslnet2_gulf", "gaslnet2", {}, "nll"),
                                ("lstm2_gulf", "lstm2", {}, "nll"),
                                ("transformer2_gulf", "transformer2", {}, "nll")]:
        if not os.path.exists(os.path.join(C.OUT_DIR, f"{tag}.pt")):
            continue
        m = T.evaluate2(tag, split="test", region=[C.CROSS_REGION], gap=0.0)
        m30 = T.evaluate2(tag, split="test", region=[C.CROSS_REGION], gap=30.0)
        res["cross"][tag] = dict(g0={k: m[k] for k in ("ADE", "FDE", "MedADE", "NLL")},
                                 g30={k: m30[k] for k in ("ADE", "FDE", "MedADE", "NLL")})
        print(f"{tag:18s} Pacific g0 ADE={m['ADE']:6.0f}  g30 ADE={m30['ADE']:6.0f}", flush=True)
        T.save_results2(res)
    T.save_results2(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="quick")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    a = ap.parse_args()
    torch.manual_seed(C.SEED)
    t0 = time.time()
    if a.stage == "quick":
        do_train(QUICK, epochs=a.epochs)
        do_eval(quick=True)
    elif a.stage == "main":
        do_train(MAIN)
    elif a.stage == "ablation":
        do_train(ABL)
    elif a.stage == "cross":
        do_train([("gaslnet2_gulf", "gaslnet2", {}, "nll"),
                  ("lstm2_gulf", "lstm2", {}, "nll"),
                  ("transformer2_gulf", "transformer2", {}, "nll")])
    elif a.stage == "seeds":
        do_seeds()
    elif a.stage == "eval":
        do_eval()
    elif a.stage == "all":
        do_train(MAIN); do_train(ABL); do_seeds(); do_eval()
    print(f"\nstage {a.stage} done in {time.time()-t0:.0f}s", flush=True)
