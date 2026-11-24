"""Models: GA-SLNet and the baseline family.

GA-SLNet (Gap-Adaptive Switching Liquid Network)
------------------------------------------------
    h_k^{t+1} = h_k^t * exp(-dt_t / tau_k^t) + (1 - exp(-dt_t / tau_k^t)) * f_k(x_t, h_k^t)
    tau_k^t   = softplus( W_k [x_t, h_k^t, phi(dt_t), a] + b_k ) + eps
    pi^t      = softmax( W_g [x_t, mean_k h_k^t, phi(dt_t), a] )
    h^t_mix   = sum_k pi_k^t h_k^t
phi(dt) = [log(1+dt), dt/60, 1{dt > 5 min}]  -- the gap conditioner.

Because exp(-dt/tau) -> 0 as dt -> inf, a long blackout drives every expert
towards its input prior f_k(x, h) instead of replaying stale memory.  Fixed-dt
recurrent baselines have no such guarantee.
"""
import math
import os

import torch
import torch.nn as nn

import config as C


def phi_dt(dt):
    """dt in minutes -> 3-d gap descriptor."""
    dt = dt.clamp(1e-2, 240.0)
    return torch.stack([torch.log1p(dt) / 4.1, dt / 60.0, (dt > 5.0).float()], dim=-1)


class LiquidCell(nn.Module):
    """Closed-form liquid cell (CfC-style, no ODE solver)."""

    def __init__(self, in_dim, hid, cond_dim, use_dt=True):
        super().__init__()
        self.hid, self.use_dt = hid, use_dt
        self.fx = nn.Linear(in_dim, hid)
        self.fh = nn.Linear(hid, hid, bias=False)
        self.tau_net = nn.Sequential(
            nn.Linear(in_dim + hid + cond_dim, hid // 2), nn.Tanh(),
            nn.Linear(hid // 2, hid))
        self.tau0 = nn.Parameter(torch.tensor(1.8))
        nn.init.zeros_(self.tau_net[-1].weight)
        nn.init.zeros_(self.tau_net[-1].bias)

    def forward(self, x, h, dt, cond):
        f = torch.tanh(self.fx(x) + self.fh(h))
        if not self.use_dt:
            dt = torch.ones_like(dt)
        tau = torch.nn.functional.softplus(self.tau_net(torch.cat([x, h, cond], -1)) + self.tau0) + 0.05
        decay = torch.exp(-dt.unsqueeze(-1) / tau)
        return decay * h + (1.0 - decay) * f, tau


class VecLiquid(nn.Module):
    """K liquid experts evaluated as one batched tensor op (no python loop over K)."""

    def __init__(self, in_dim, hid, cond_dim, K):
        super().__init__()
        self.K, self.hid = K, hid
        self.Wx = nn.Parameter(torch.empty(K, hid, in_dim))
        self.bx = nn.Parameter(torch.zeros(K, hid))
        self.Wh = nn.Parameter(torch.empty(K, hid, hid))
        self.tw1 = nn.Parameter(torch.empty(K, hid // 2, in_dim + hid + cond_dim))
        self.tb1 = nn.Parameter(torch.zeros(K, hid // 2))
        self.tw2 = nn.Parameter(torch.zeros(K, hid, hid // 2))
        self.tb2 = nn.Parameter(torch.zeros(K, hid))
        self.tau0 = nn.Parameter(torch.full((K, 1), 1.8))
        for p in (self.Wx, self.Wh, self.tw1):
            nn.init.xavier_uniform_(p)

    def forward(self, x, h, dt, cond, use_dt=True):
        f = torch.tanh(torch.einsum('bf,khf->bkh', x, self.Wx) + self.bx +
                       torch.einsum('bkh,kgh->bkg', h, self.Wh))
        B = x.shape[0]
        z = torch.cat([x.unsqueeze(1).expand(B, self.K, -1), h,
                       cond.unsqueeze(1).expand(B, self.K, -1)], -1)
        z1 = torch.tanh(torch.einsum('bkc,kdc->bkd', z, self.tw1) + self.tb1)
        raw = torch.einsum('bkd,khd->bkh', z1, self.tw2) + self.tb2 + self.tau0
        tau = torch.nn.functional.softplus(raw) + 0.05
        d = dt if use_dt else torch.ones_like(dt)
        decay = torch.exp(-d.view(B, 1, 1) / tau)
        return decay * h + (1.0 - decay) * f, tau


class GASLNet(nn.Module):
    def __init__(self, in_dim=C.N_FEAT, attr_dim=C.N_ATTR, hid=C.HIDDEN,
                 K=C.K_EXPERTS, n_modes=C.N_MODES, use_dt=True, use_attr=True,
                 use_switch=True):
        super().__init__()
        self.K = K if use_switch else 1
        self.use_dt, self.use_attr, self.hid = use_dt, use_attr, hid
        self.use_switch = use_switch
        cond_dim = (3 if use_dt else 0) + (attr_dim if use_attr else 0)
        self.cond_dim, self.attr_dim = max(cond_dim, 1), attr_dim
        self.cell = VecLiquid(in_dim, hid, self.cond_dim, self.K)
        self.gate = nn.Linear(in_dim + hid + self.cond_dim, self.K)
        dec_in = hid + (attr_dim if use_attr else 0)
        self.dec = nn.Sequential(nn.Linear(dec_in, 128), nn.GELU(), nn.Dropout(C.DROPOUT),
                                 nn.Linear(128, 128), nn.GELU())
        self.head = nn.Linear(128, n_modes * len(C.HORIZONS_MIN) * 2)
        self.mode_logit = nn.Linear(128, n_modes)
        self.n_modes, self.nh = n_modes, len(C.HORIZONS_MIN)
        self.log_sigma = nn.Parameter(torch.full((self.nh, 2), math.log(3.0)))

    def _cond(self, dt, attr):
        parts = []
        if self.use_dt:
            parts.append(phi_dt(dt))
        if self.use_attr:
            parts.append(attr)
        if not parts:
            return dt.new_zeros(dt.shape[0], 1)
        return torch.cat(parts, -1)

    def forward(self, f, dt, mask, attr, return_tau=False):
        B, T, _ = f.shape
        h = f.new_zeros(B, self.K, self.hid)
        h_mix = f.new_zeros(B, self.hid)
        taus, pis = [], []
        for t in range(T):
            x = f[:, t]
            c = self._cond(dt[:, t], attr)
            hs, tau = self.cell(x, h, dt[:, t], c, self.use_dt)
            m = mask[:, t].unsqueeze(-1).float()
            h = m.unsqueeze(-1) * hs + (1 - m.unsqueeze(-1)) * h
            pi = torch.softmax(self.gate(torch.cat([x, h.mean(1), c], -1)), -1)
            h_mix = m * (pi.unsqueeze(-1) * h).sum(1) + (1 - m) * h_mix
            taus.append(tau.mean(-1))
            pis.append(pi)
        z = torch.cat([h_mix, attr], -1) if self.use_attr else h_mix
        z = self.dec(z)
        y = self.head(z).view(B, self.n_modes, self.nh, 2) + cv_anchor(f, mask)
        self.last_tau = torch.stack(taus, 1).detach()
        self.last_gate = torch.stack(pis, 1).detach()
        return y, self.mode_logit(z), None




def cv_anchor(f, mask):
    """Constant-velocity anchor: last observed velocity extrapolated to each horizon.

    Shared by EVERY model (ours and baselines) so the comparison stays fair; the
    network only has to learn the residual.
    """
    if not getattr(C, "USE_CV_ANCHOR", False):
        h = torch.tensor(C.HORIZONS_MIN, device=f.device).float()
        return torch.zeros(f.shape[0], 1, len(h), 2, device=f.device)
    n = (mask.sum(1) - 1).clamp(min=0)
    b = torch.arange(f.shape[0], device=f.device)
    v = f[b, n][:, :2] * 400.0                      # m/min
    h = torch.tensor(C.HORIZONS_MIN, device=f.device).float()
    return (v.unsqueeze(1) * h.view(1, -1, 1)).unsqueeze(1) / 1000.0   # B,1,H,2 (km)


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------
class _SeqBase(nn.Module):
    def __init__(self, n_modes=1):
        super().__init__()
        self.nh, self.n_modes = len(C.HORIZONS_MIN), n_modes
        self.log_sigma = nn.Parameter(torch.full((self.nh, 2), math.log(3.0)))


class LSTMPred(_SeqBase):
    def __init__(self, hid=64, attr_dim=C.N_ATTR):
        super().__init__()
        self.rnn = nn.LSTM(C.N_FEAT, hid, 1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid + attr_dim, 128), nn.GELU(),
                                  nn.Linear(128, self.nh * 2))

    def forward(self, f, dt, mask, attr):
        out, _ = self.rnn(f)
        last = (out * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        y = self.head(torch.cat([last, attr], -1)).view(-1, 1, self.nh, 2) + cv_anchor(f, mask)
        return y, None, None


class GRUPred(LSTMPred):
    def __init__(self, hid=64, attr_dim=C.N_ATTR):
        nn.Module.__init__(self)
        _SeqBase.__init__(self)
        self.rnn = nn.GRU(C.N_FEAT, hid, 1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid + attr_dim, 128), nn.GELU(),
                                  nn.Linear(128, self.nh * 2))


class Seq2SeqPred(_SeqBase):
    def __init__(self, hid=64, attr_dim=C.N_ATTR):
        super().__init__()
        self.enc = nn.LSTM(C.N_FEAT, hid, 1, batch_first=True)
        self.dec = nn.GRUCell(2, hid)
        self.out = nn.Linear(hid, 2)
        self.hid, self.attr_dim = hid, attr_dim
        self.h0 = nn.Linear(hid + attr_dim, hid)

    def forward(self, f, dt, mask, attr):
        out, _ = self.enc(f)
        last = (out * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        h = torch.tanh(self.h0(torch.cat([last, attr], -1)))
        inp = f.new_zeros(f.shape[0], 2)
        ys = []
        for _ in range(self.nh):
            h = self.dec(inp, h)
            d = self.out(h)
            ys.append(d)
            inp = d
        y = torch.stack(ys, 1).unsqueeze(1) + cv_anchor(f, mask)
        return y, None, None


class TransformerPred(_SeqBase):
    def __init__(self, d=64, nhead=4, layers=2, attr_dim=C.N_ATTR):
        super().__init__()
        self.proj = nn.Linear(C.N_FEAT, d)
        self.pos = nn.Parameter(torch.zeros(1, C.MAX_OBS_STEPS, d))
        nn.init.normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d, nhead, 128, C.DROPOUT, batch_first=True,
                                         norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.Linear(d + attr_dim, 128), nn.GELU(),
                                  nn.Linear(128, self.nh * 2))
        self.d = d

    def forward(self, f, dt, mask, attr):
        x = self.proj(f) + self.pos[:, :f.shape[1]]
        x = self.enc(x, src_key_padding_mask=~mask)
        last = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        y = self.head(torch.cat([last, attr], -1)).view(-1, 1, self.nh, 2) + cv_anchor(f, mask)
        return y, None, None


class VanillaCfC(_SeqBase):
    """CfC on the uniform grid: dt is forced to a constant step (prior practice)."""

    def __init__(self, hid=64, attr_dim=C.N_ATTR):
        super().__init__()
        self.cell = LiquidCell(C.N_FEAT, hid, attr_dim, use_dt=False)
        self.head = nn.Sequential(nn.Linear(hid + attr_dim, 128), nn.GELU(),
                                  nn.Linear(128, self.nh * 2))

    def forward(self, f, dt, mask, attr):
        h = f.new_zeros(f.shape[0], self.cell.hid)
        for t in range(f.shape[1]):
            hn, _ = self.cell(f[:, t], h, torch.ones_like(dt[:, t]), attr)
            m = mask[:, t].unsqueeze(-1).float()
            h = m * hn + (1 - m) * h
        y = self.head(torch.cat([h, attr], -1)).view(-1, 1, self.nh, 2) + cv_anchor(f, mask)
        return y, None, None


class LTCPred(_SeqBase):
    """Simplified liquid time-constant network (explicit Euler, fixed step)."""

    def __init__(self, hid=64, attr_dim=C.N_ATTR):
        super().__init__()
        self.hid = hid
        self.wx = nn.Linear(C.N_FEAT, hid)
        self.wh = nn.Linear(hid, hid, bias=False)
        self.tau_n = nn.Linear(C.N_FEAT + hid + attr_dim, hid)
        self.a_n = nn.Linear(hid, hid)
        self.head = nn.Sequential(nn.Linear(hid + attr_dim, 128), nn.GELU(),
                                  nn.Linear(128, self.nh * 2))

    def forward(self, f, dt, mask, attr):
        B, T, _ = f.shape
        h = f.new_zeros(B, self.hid)
        for t in range(T):
            x = f[:, t]
            s = torch.sigmoid(self.wx(x) + self.wh(h))
            tau = torch.nn.functional.softplus(self.tau_n(torch.cat([x, h, attr], -1))) + 0.1
            a = torch.tanh(self.a_n(h))
            dt = 1.0
            hn = (h + dt * s * a) / (1.0 + dt * (1.0 / tau + s))   # implicit Euler
            m = mask[:, t].unsqueeze(-1).float()
            h = m * hn + (1 - m) * h
        y = self.head(torch.cat([h, attr], -1)).view(-1, 1, self.nh, 2) + cv_anchor(f, mask)
        return y, None, None


class NeuralODEPred(_SeqBase):
    """Neural ODE with explicit Euler integration (2 sub-steps), fixed step."""

    def __init__(self, hid=64, attr_dim=C.N_ATTR):
        super().__init__()
        self.hid = hid
        self.f = nn.Sequential(nn.Linear(C.N_FEAT + hid, hid), nn.Tanh(),
                               nn.Linear(hid, hid))
        self.h0 = nn.Linear(C.N_FEAT + attr_dim, hid)
        self.head = nn.Sequential(nn.Linear(hid + attr_dim, 128), nn.GELU(),
                                  nn.Linear(128, self.nh * 2))

    def forward(self, f, dt, mask, attr):
        B, T, _ = f.shape
        h = torch.tanh(self.h0(torch.cat([f[:, 0], attr], -1)))
        for t in range(T):
            x = f[:, t]
            for _ in range(2):
                h = h + 0.5 * self.f(torch.cat([x, h], -1))
            m = mask[:, t].unsqueeze(-1).float()
            h = m * h + (1 - m) * h
        y = self.head(torch.cat([h, attr], -1)).view(-1, 1, self.nh, 2) + cv_anchor(f, mask)
        return y, None, None


def build(name, **kw):
    table = dict(gaslnet=GASLNet, lstm=LSTMPred, gru=GRUPred, seq2seq=Seq2SeqPred,
                 transformer=TransformerPred, cfc=VanillaCfC, ltc=LTCPred, ode=NeuralODEPred)
    return table[name](**kw)


# view selector: only GA-SLNet consumes the irregular stream
def view_of(name):
    return "irr" if name in ("gaslnet",) else "uni"
