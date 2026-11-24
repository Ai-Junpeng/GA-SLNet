"""Training / evaluation driver v2 (blackout task)."""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as C
import dataset2 as D2
import metrics2 as MET2
import models2 as M2

torch.backends.cudnn.benchmark = True

VAL_GAPS = (0.0, 15.0, 30.0)


def build_ds(split, region=None, gap=0.0, train=False):
    return D2.AISDataset2(
        os.path.join(C.PROC_DIR, "train2.npz" if split == "train" else "eval2.npz"),
        split=split, region_filter=region, gap_min=gap, train=train)


@torch.no_grad()
def val_ade(model, loaders, view):
    """mean ADE over the validation gap set -> selects for robustness."""
    ades = []
    for dl in loaders:
        y, mu, lg, sg = MET2.predict2(model, dl, view)
        ades.append(MET2.metrics_from2(y, mu, lg, sg)["ADE"])
    return float(np.mean(ades))


def train_one(name, tag, loss="nll", epochs=C.EPOCHS, train_regions=None,
              kw=None, seed=C.SEED, verbose=True, patience=8):
    torch.manual_seed(seed)
    np.random.seed(seed)
    kw = kw or {}
    model = M2.build2(name, **kw).cuda()
    view = M2.view_of2(name)

    tr = build_ds("train", region=train_regions or C.TRAIN_REGIONS, train=True)
    tr_loader = DataLoader(tr, batch_size=C.BATCH, shuffle=True, drop_last=True,
                           num_workers=0, pin_memory=True)
    va_loaders = []
    for g in VAL_GAPS:
        va = build_ds("val", region=C.TRAIN_REGIONS, gap=g)
        va_loaders.append(DataLoader(va, batch_size=256, shuffle=False,
                                     num_workers=0, pin_memory=True))

    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs * max(len(tr_loader), 1), eta_min=C.LR * 0.02)

    best, best_state, hist, bad = 1e9, None, [], 0
    t0 = time.time()
    for ep in range(epochs):
        tr.set_epoch(ep)
        model.train()
        tot, nb = 0.0, 0
        for b in tr_loader:
            b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
            L = max(int(b[f"{view}_mask"].sum(1).max().item()), 8)
            mu, lg, sg = model(b[view][:, :L], b[f"{view}_dt"][:, :L],
                               b[f"{view}_mask"][:, :L], b["attr"],
                               b["lead"], b["tgap"])
            l = MET2.nll_loss2(b["y"], mu, lg, sg) if loss == "nll" else MET2.wta_loss2(b["y"], mu)
            opt.zero_grad(set_to_none=True)
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            opt.step()
            sched.step()
            tot += float(l); nb += 1
        model.eval()
        va = val_ade(model, va_loaders, view)
        hist.append(dict(epoch=ep, train_loss=tot / max(nb, 1), val_ADE=va))
        if verbose:
            print(f"  ep{ep:02d} loss={tot/max(nb,1):.4f} valADE={va:.1f}m "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if va < best - 1e-3:
            best, bad = va, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at ep{ep}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(dict(state=model.state_dict(), name=name, kw=kw, loss=loss,
                    train_regions=train_regions or C.TRAIN_REGIONS),
               os.path.join(C.OUT_DIR, f"{tag}.pt"))
    with open(os.path.join(C.OUT_DIR, f"{tag}_history.json"), "w") as f:
        json.dump(hist, f, indent=1)
    return model, best


_cache = {}


def load(tag):
    if tag in _cache:
        return _cache[tag]
    ck = torch.load(os.path.join(C.OUT_DIR, f"{tag}.pt"), map_location="cuda",
                    weights_only=False)
    kw = ck.get("kw", {})
    if ck["name"] == "gaslnet2":
        kw = {k: v for k, v in kw.items()
              if k in ("use_dt", "use_attr", "use_switch", "use_prop", "K",
                       "n_modes", "hid", "tau0", "rich_dec")}
    model = M2.build2(ck["name"], **kw).cuda()
    model.load_state_dict(ck["state"])
    model.eval()
    _cache[tag] = (model, ck["name"])
    return _cache[tag]


def evaluate2(tag, split="test", region=None, gap=0.0, n_max=None, ret_raw=False):
    model, name = load(tag)
    view = M2.view_of2(name)
    ds = build_ds(split if split != "train" else "val", region=region, gap=gap)
    if n_max and len(ds) > n_max:
        ds.idx = ds.idx[:n_max]
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    y, mu, lg, sg = MET2.predict2(model, dl, view)
    m = MET2.metrics_from2(y, mu, lg, sg)
    ece, conf, acc = MET2.ece_from2(y, mu, lg, sg)
    m["ECE"] = ece
    m["n"] = int(len(y))
    if ret_raw:
        return m, (y, mu, lg, sg), (conf, acc)
    return m


def efficiency2(tag, reps=30):
    model, _ = load(tag)
    x = torch.zeros(1, 48, M2.N_FEAT).cuda()
    dt = torch.ones(1, 48).cuda()
    mask = torch.ones(1, 48, dtype=torch.bool).cuda()
    at = torch.zeros(1, M2.N_ATTR).cuda()
    lead = torch.tensor(C.HORIZONS_MIN).float().view(1, -1).cuda()
    tg = torch.zeros(1).cuda()
    for _ in range(5):
        model(x, dt, mask, at, lead, tg)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        model(x, dt, mask, at, lead, tg)
    torch.cuda.synchronize()
    lat = (time.time() - t0) / reps * 1000
    params = sum(p.numel() for p in model.parameters())
    return dict(params=params, latency_ms=round(lat, 2))


def cv_baseline2(split="test", region=None, gap=0.0):
    """Constant-velocity extrapolation on the blackout task."""
    ds = build_ds(split, region=region, gap=gap)
    err = []
    for i in range(len(ds)):
        b = ds[i]
        m = b["uni_mask"].numpy()
        idx = np.where(m)[0]
        if len(idx) < 2:
            continue
        f = b["uni"].numpy()
        a, bb = idx[-2], idx[-1]
        v = f[bb, :2] * 400.0                              # m/min (rate features)
        lead = b["lead"].numpy()                           # H
        pred = v[None, :] * lead[:, None]                  # H,2 m
        err.append(np.linalg.norm(pred - b["y"].numpy() * 1000.0, axis=1))
    err = np.stack(err)
    return dict(ADE=float(err.mean()), FDE=float(err[:, -1].mean()),
                MedADE=float(np.median(err.mean(1))), n=len(err))


def save_results2(res, path=None):
    path = path or os.path.join(C.OUT_DIR, "results2.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


def load_results2(path=None):
    path = path or os.path.join(C.OUT_DIR, "results2.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gaslnet2")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--loss", default="nll")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--no-dt", action="store_true")
    ap.add_argument("--no-attr", action="store_true")
    ap.add_argument("--no-switch", action="store_true")
    ap.add_argument("--no-prop", action="store_true")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--train-region", default=None)
    a = ap.parse_args()
    kw = {}
    if a.model == "gaslnet2":
        kw["use_dt"] = not a.no_dt
        kw["use_attr"] = not a.no_attr
        kw["use_switch"] = not a.no_switch
        kw["use_prop"] = not a.no_prop
        if a.k:
            kw["K"] = a.k
    tag = a.tag or a.model
    regions = [a.train_region] if a.train_region else None
    train_one(a.model, tag, loss=a.loss, epochs=a.epochs, train_regions=regions,
              kw=kw, seed=a.seed)
