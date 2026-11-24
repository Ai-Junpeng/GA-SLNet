"""Training / evaluation driver for one configuration."""
import argparse
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

torch.backends.cudnn.benchmark = True


def build_ds(split, region=None, gap=0.0, train=False):
    return D.AISDataset(os.path.join(C.PROC_DIR, "eval.npz" if split != "train" else "train.npz"),
                        split=split, region_filter=region, gap_min=gap, train=train)


def train_one(name, tag, loss="nll", epochs=C.EPOCHS, train_regions=None,
              kw=None, seed=C.SEED, verbose=True, patience=8):
    torch.manual_seed(seed)
    np.random.seed(seed)
    kw = kw or {}
    model = M.build(name, **kw).cuda()
    view = M.view_of(name)

    tr = build_ds("train", region=train_regions or C.TRAIN_REGIONS, train=True)
    va = build_ds("val", region=C.TRAIN_REGIONS)
    tr_loader = DataLoader(tr, batch_size=C.BATCH, shuffle=True, drop_last=True,
                           num_workers=0, pin_memory=True)
    va_loader = DataLoader(va, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

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
            y, lg, _ = model(b[view][:, :L], b[f"{view}_dt"][:, :L],
                             b[f"{view}_mask"][:, :L], b["attr"])
            if loss == "nll":
                l = MET.nll_loss(b["y"], y, lg, model.log_sigma)
            else:
                l = MET.wta_loss(b["y"], y)
            opt.zero_grad(set_to_none=True)
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            opt.step()
            sched.step()
            tot += float(l); nb += 1
        model.eval()
        yv, muv, lgv, sv = MET.predict(model, va_loader, view)
        mv = MET.metrics_from(yv, muv, lgv, sv)
        hist.append(dict(epoch=ep, train_loss=tot / max(nb, 1), val_ADE=mv["ADE"],
                         val_NLL=mv["NLL"]))
        if verbose:
            print(f"  ep{ep:02d} loss={tot/max(nb,1):.4f} valADE={mv['ADE']:.1f}m "
                  f"valNLL={mv['NLL']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if mv["ADE"] < best - 1e-3:
            best, bad = mv["ADE"], 0
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gaslnet")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--loss", default="nll")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--no-dt", action="store_true")
    ap.add_argument("--no-attr", action="store_true")
    ap.add_argument("--no-switch", action="store_true")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--modes", type=int, default=None)
    ap.add_argument("--train-region", default=None)
    a = ap.parse_args()
    kw = {}
    if a.model == "gaslnet":
        kw["use_dt"] = not a.no_dt
        kw["use_attr"] = not a.no_attr
        kw["use_switch"] = not a.no_switch
        if a.k:
            kw["K"] = a.k
        if a.modes:
            kw["n_modes"] = a.modes
    tag = a.tag or f"{a.model}"
    regions = [a.train_region] if a.train_region else None
    train_one(a.model, tag, loss=a.loss, epochs=a.epochs, train_regions=regions, kw=kw)


if __name__ == "__main__":
    main()
