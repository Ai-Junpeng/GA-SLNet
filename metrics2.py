"""Metrics v2: heteroscedastic mixture NLL + calibration (distances in metres)."""
import numpy as np
import torch

LOG2PI = float(np.log(2 * np.pi))


def nll_loss2(y, mu, logits, sigma):
    """y (B,H,2), mu (B,M,H,2), sigma (B,M,H,2), all in km."""
    if y.dim() == 3:
        y = y.unsqueeze(1)
    r = (y - mu) / sigma
    ll = -0.5 * (r ** 2).sum(-1) - torch.log(sigma).sum(-1) - LOG2PI   # B,M,H
    ll = ll.sum(-1)                                                    # B,M
    if logits is not None:
        lp = torch.log_softmax(logits, -1)
    else:
        lp = torch.zeros(mu.shape[0], 1, device=mu.device)
    return -torch.logsumexp(lp + ll, dim=1).mean()


def wta_loss2(y, mu):
    if y.dim() == 3:
        y = y.unsqueeze(1)
    d = ((y - mu) ** 2).sum(-1).mean(-1)
    return d.min(dim=1).values.mean()


@torch.no_grad()
def predict2(model, loader, view):
    model.eval()
    dev = next(model.parameters()).device
    ys, mus, lgs, sgs = [], [], [], []
    for b in loader:
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        mu, lg, sg = model(b[view], b[f"{view}_dt"], b[f"{view}_mask"],
                           b["attr"], b["lead"], b["tgap"])
        ys.append(b["y"].cpu()); mus.append(mu.cpu()); sgs.append(sg.cpu())
        if lg is not None:
            lgs.append(lg.cpu())
    lg = torch.cat(lgs) if lgs else None
    return torch.cat(ys), torch.cat(mus), lg, torch.cat(sgs)


def _dist(a, b):
    return torch.sqrt(((a - b) ** 2).sum(-1) + 1e-12)


@torch.no_grad()
def metrics_from2(y, mu, lg, sg):
    """y (N,H,2) km, mu (N,M,H,2) km, sg (N,M,H,2) km."""
    N, M, H, _ = mu.shape
    best = lg.argmax(1) if lg is not None else torch.zeros(N, dtype=torch.long)
    rows = torch.arange(N)
    mu_ml = mu[rows, best]
    d_ml = _dist(mu_ml, y) * 1000.0
    d_all = _dist(mu, y.unsqueeze(1)) * 1000.0
    d_min = d_all.min(1).values

    out = dict(
        ADE=float(d_ml.mean()), FDE=float(d_ml[:, -1].mean()),
        minADE=float(d_min.mean()), minFDE=float(d_min[:, -1].mean()),
        mae_h=[float(d_ml[:, i].mean()) for i in range(H)],
        MedADE=float(d_ml.mean(1).median()), MedFDE=float(d_ml[:, -1].median()),
        P90ADE=float(d_ml.mean(1).quantile(0.9)),
    )
    r = (y.unsqueeze(1) - mu) / sg
    ll = (-0.5 * (r ** 2).sum(-1) - torch.log(sg).sum(-1) - LOG2PI).sum(-1)   # N,M
    lp = torch.log_softmax(lg, -1) if lg is not None else torch.zeros(N, 1)
    out["NLL"] = float(-torch.logsumexp(lp + ll, 1).mean())

    g = torch.Generator().manual_seed(0)
    S = 32
    probs = torch.softmax(lg, -1) if lg is not None else torch.ones(N, 1)
    idx = torch.multinomial(probs, S, replacement=True, generator=g)
    samp = mu[rows.unsqueeze(1), idx]
    ssamp = sg[rows.unsqueeze(1), idx]
    samp = samp + torch.randn(samp.shape, generator=g) * ssamp
    term1 = _dist(samp, y.unsqueeze(1)).mean(1)
    term2 = _dist(samp.unsqueeze(2), samp.unsqueeze(1)).mean((1, 2))
    out["CRPS"] = float((term1 - 0.5 * term2).mean())
    out["sigma_km"] = float(sg.mean())
    return out


@torch.no_grad()
def ece_from2(y, mu, lg, sg, bins=10):
    N = mu.shape[0]
    best = lg.argmax(1) if lg is not None else torch.zeros(N, dtype=torch.long)
    rows = torch.arange(N)
    err = _dist(mu[rows, best], y)                       # N,H km
    s = sg[rows, best].mean(-1)                          # N,H km
    conf, acc = [], []
    for p in np.linspace(0.05, 0.95, bins):
        r = s * np.sqrt(-2 * np.log(1 - p))              # Rayleigh quantile, N,H
        conf.append(p)
        acc.append(float((err <= r).float().mean()))
    conf, acc = np.array(conf), np.array(acc)
    return float(np.abs(conf - acc).mean()), conf, acc
