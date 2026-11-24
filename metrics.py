"""Metric utilities (all distances in metres, inputs in km)."""
import numpy as np
import torch

import config as C

H = len(C.HORIZONS_MIN)
LOG2PI = float(np.log(2 * np.pi))


def nll_loss(y, mu, logits, log_sigma):
    sigma = torch.exp(log_sigma).unsqueeze(0).unsqueeze(0)      # 1,1,H,2
    if y.dim() == 3:
        y = y.unsqueeze(1)
    r = (y - mu) / sigma
    ll = -0.5 * (r ** 2).sum(-1) - torch.log(sigma).sum(-1) - H * LOG2PI
    if logits is not None:
        lp = torch.log_softmax(logits, -1).unsqueeze(-1)
    else:
        lp = torch.zeros(mu.shape[0], 1, 1, device=mu.device)
    return -torch.logsumexp(lp + ll, dim=1).mean()


def wta_loss(y, mu):
    if y.dim() == 3:
        y = y.unsqueeze(1)
    d = ((y - mu) ** 2).sum(-1).mean(-1)          # N,M  mean over horizons
    return d.min(dim=1).values.mean()


@torch.no_grad()
def predict(model, loader, view):
    model.eval()
    dev = next(model.parameters()).device
    ys, mus, lgs = [], [], []
    for b in loader:
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        y, logits, _ = model(b[view], b[f"{view}_dt"], b[f"{view}_mask"], b["attr"])
        ys.append(b["y"].cpu())
        mus.append(y.cpu())
        if logits is not None:
            lgs.append(logits.cpu())
    y = torch.cat(ys)
    mu = torch.cat(mus)
    lg = torch.cat(lgs) if lgs else None
    return y, mu, lg, model.log_sigma.detach().cpu()


def _dist(a, b):
    return torch.sqrt(((a - b) ** 2).sum(-1) + 1e-12)


@torch.no_grad()
def metrics_from(y, mu, lg, log_sigma):
    """y (N,H,2) km, mu (N,M,H,2) km."""
    N, M, Hh, _ = mu.shape
    if lg is not None:
        best = lg.argmax(1)
    else:
        best = torch.zeros(N, dtype=torch.long)
    mu_ml = mu[torch.arange(N), best]                            # N,H,2
    d_ml = _dist(mu_ml, y) * 1000.0                              # N,H  metres
    d_all = _dist(mu.unsqueeze(2), y.unsqueeze(1).unsqueeze(2)).squeeze(2) * 1000.0
    d_min = d_all.min(1).values

    out = dict(
        ADE=float(d_ml.mean()),
        FDE=float(d_ml[:, -1].mean()),
        minADE=float(d_min.mean()),
        minFDE=float(d_min[:, -1].mean()),
        rmse_h=[float(torch.sqrt((d_ml[:, i] ** 2).mean())) for i in range(Hh)],
        mae_h=[float(d_ml[:, i].mean()) for i in range(Hh)],
        MedADE=float(d_ml.mean(1).median()),
        MedFDE=float(d_ml[:, -1].median()),
    )
    # mixture NLL
    sigma = torch.exp(log_sigma).unsqueeze(0)
    r = (y.unsqueeze(1) - mu) / sigma
    ll = -0.5 * (r ** 2).sum(-1) - torch.log(sigma).sum(-1) - Hh * LOG2PI
    lp = torch.log_softmax(lg, -1).unsqueeze(-1) if lg is not None else torch.zeros(N, 1, 1)
    out["NLL"] = float(-torch.logsumexp(lp + ll, 1).mean())

    # CRPS via sampling from the predictive mixture
    g = torch.Generator().manual_seed(0)
    S = 32
    idx = torch.multinomial(torch.softmax(lg, -1) if lg is not None else torch.ones(N, 1),
                            S, replacement=True, generator=g).reshape(N, S)
    samp = mu[torch.arange(N).unsqueeze(1), idx]                     # N,S,H,2
    samp = samp + torch.randn(samp.shape, generator=g) * sigma.expand(N, S, Hh, 2)
    term1 = _dist(samp, y.unsqueeze(1)).mean(1)                      # N,H
    term2 = _dist(samp.unsqueeze(2), samp.unsqueeze(1)).mean((1, 2))
    out["CRPS"] = float((term1 - 0.5 * term2).mean())
    out["sigma_km"] = float(torch.exp(log_sigma).mean())
    return out


def ece_from(y, mu, lg, log_sigma, bins=10):
    """Calibration of the predictive 2-D Gaussian: P(err<r) vs empirical."""
    N, M, Hh, _ = mu.shape
    sigma = torch.exp(log_sigma).unsqueeze(0)
    best = lg.argmax(1) if lg is not None else torch.zeros(N, dtype=torch.long)
    mu_ml = mu[torch.arange(N), best]
    err = _dist(mu_ml, y) * 1000.0
    # 2-D isotropic Gaussian -> Rayleigh CDF, sigma averaged over the 2 axes
    s = sigma[0, 0].mean(-1).unsqueeze(0) * 1000.0                    # 1,H
    conf, acc = [], []
    for p in np.linspace(0.05, 0.95, bins):
        r = s * np.sqrt(-2 * np.log(1 - p))
        conf.append(p)
        acc.append(float((err <= r).float().mean()))
    conf, acc = np.array(conf), np.array(acc)
    return float(np.abs(conf - acc).mean()), conf, acc
