"""Load a trained tag and evaluate it under an arbitrary split / gap / region."""
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as C
import dataset as D
import metrics as MET
import models as M

_cache = {}


def load(tag):
    if tag in _cache:
        return _cache[tag]
    ck = torch.load(os.path.join(C.OUT_DIR, f"{tag}.pt"), map_location="cuda",
                    weights_only=False)
    kw = ck.get("kw", {})
    if ck["name"] == "gaslnet":
        kw = {k: v for k, v in kw.items() if k in ("use_dt", "use_attr", "use_switch", "K", "n_modes")}
    model = M.build(ck["name"], **kw).cuda()
    model.load_state_dict(ck["state"])
    model.eval()
    _cache[tag] = (model, ck["name"])
    return _cache[tag]


def evaluate(tag, split="test", region=None, gap=0.0, n_max=None, ret_raw=False):
    model, name = load(tag)
    view = M.view_of(name)
    ds = D.AISDataset(os.path.join(C.PROC_DIR, "eval.npz"), split=split,
                      region_filter=region, gap_min=gap)
    if n_max and len(ds) > n_max:
        ds.idx = ds.idx[:n_max]
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    y, mu, lg, ls = MET.predict(model, dl, view)
    m = MET.metrics_from(y, mu, lg, ls)
    ece, conf, acc = MET.ece_from(y, mu, lg, ls)
    m["ECE"] = ece
    m["n"] = int(len(y))
    if ret_raw:
        return m, (y, mu, lg, ls), (conf, acc)
    return m


def efficiency(tag, reps=30):
    model, _ = load(tag)
    x = torch.zeros(1, 48, C.N_FEAT).cuda()
    dt = torch.ones(1, 48).cuda()
    mask = torch.ones(1, 48, dtype=torch.bool).cuda()
    at = torch.zeros(1, C.N_ATTR).cuda()
    for _ in range(5):
        model(x, dt, mask, at)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        model(x, dt, mask, at)
    torch.cuda.synchronize()
    lat = (time.time() - t0) / reps * 1000
    params = sum(p.numel() for p in model.parameters())
    return dict(params=params, latency_ms=round(lat, 2))


def load_results(path=None):
    path = path or os.path.join(C.OUT_DIR, "results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_results(res, path=None):
    path = path or os.path.join(C.OUT_DIR, "results.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


def cv_baseline(split="test", region=None, gap=0.0):
    """Constant-velocity extrapolation -- the trivial reference every model must beat."""
    ds = D.AISDataset(os.path.join(C.PROC_DIR, "eval.npz"), split=split,
                      region_filter=region, gap_min=gap)
    X, TM, RM, Y = ds.X[ds.idx], ds.TM[ds.idx], ds.RM[ds.idx], ds.Y[ds.idx]
    err = []
    for i in range(len(X)):
        m = RM[i]
        idx = np.where(m)[0]
        if len(idx) < 2:
            continue
        a, b = idx[-2], idx[-1]
        dt = max(TM[i, b] - TM[i, a], 1e-2)
        v = (X[i, b, :2] - X[i, a, :2]) / dt                   # m/min
        pred = v[None, :] * np.array(C.HORIZONS_MIN)[:, None]
        err.append(np.linalg.norm(pred - Y[i], axis=1))
    err = np.stack(err)
    return dict(ADE=float(err.mean()), FDE=float(err[:, -1].mean()),
                mae_h=err.mean(0).tolist(), n=len(err))
