"""Models v2 for the blackout-prediction task.

GA-SLNet v2 changes vs v1
--------------------------
1. Time-bearing inputs (12-d): displacement dE=v*dt and log dt keep the time
   interval multiplicative instead of dividing it away; a clock channel gives
   the global position inside the window.
2. Blackout propagation (theory-backed): after the anchor, every liquid expert
   is advanced over the blackout g with its last input held constant,
       h <- f + (h - f) * exp(-g / tau(x_last, h, phi(g), a))
   so stale memory decays provably onto the learned prior as g grows.
3. Time-conditioned decoder: each horizon is decoded with phi(lead_i); the same
   head is shared by ALL models (baselines included) so the comparison is fair.
4. Heteroscedastic mixture output: per-mode per-horizon (mu, sigma) replaces the
   single global log_sigma -> sharper NLL and honest calibration.

Output contract: forward(f, dt, mask, attr, lead, tgap)
    -> mu (B,M,H,2) km, logits (B,M) or None, sigma (B,M,H,2) km
"""
import math

import torch
import torch.nn as nn

import config as C

N_FEAT = 12
N_ATTR = C.N_ATTR
NH = len(C.HORIZONS_MIN)


def phi_dt(dt):
    dt = dt.clamp(1e-2, 240.0)
    return torch.stack([torch.log1p(dt) / 4.1, dt / 60.0, (dt > 5.0).float()], dim=-1)


def phi_lead(lead):
    lead = lead.clamp(0.0, 240.0)
    return torch.stack([torch.log1p(lead) / 4.7, lead / 75.0], dim=-1)


def last_step(f, mask):
    n = (mask.sum(1) - 1).clamp(min=0)
    b = torch.arange(f.shape[0], device=f.device)
    return f[b, n]


# --------------------------------------------------------------------------
# shared time-conditioned mixture head
# --------------------------------------------------------------------------
class TimeCondHead(nn.Module):
    def __init__(self, state_dim, attr_dim=N_ATTR, n_modes=C.N_MODES, n_h=NH, hid=128):
        super().__init__()
        self.n_modes, self.n_h = n_modes, n_h
        self.trunk = nn.Sequential(nn.Linear(state_dim + attr_dim, hid), nn.GELU(),
                                   nn.Linear(hid, hid), nn.GELU())
        self.per_h = nn.Sequential(nn.Linear(hid + 2, hid), nn.GELU(),
                                   nn.Linear(hid, n_modes * 4))
        self.logits = nn.Linear(hid, n_modes)

    def forward(self, z, attr, lead):
        B = z.shape[0]
        z = self.trunk(torch.cat([z, attr], -1))                       # B,hid
        ph = phi_lead(lead)                                            # B,H,2
        hin = torch.cat([z.unsqueeze(1).expand(B, self.n_h, z.shape[-1]), ph], -1)
        out = self.per_h(hin).view(B, self.n_h, self.n_modes, 4).permute(0, 2, 1, 3)
        mu = out[..., :2]
        sigma = torch.nn.functional.softplus(out[..., 2:] + 1.0) * 0.5 + 0.02
        return mu, self.logits(z), sigma


# --------------------------------------------------------------------------
# GA-SLNet v2
# --------------------------------------------------------------------------
class VecLiquid(nn.Module):
    def __init__(self, in_dim, hid, cond_dim, K, tau0=1.8):
        super().__init__()
        self.K, self.hid = K, hid
        self.Wx = nn.Parameter(torch.empty(K, hid, in_dim))
        self.bx = nn.Parameter(torch.zeros(K, hid))
        self.Wh = nn.Parameter(torch.empty(K, hid, hid))
        self.tw1 = nn.Parameter(torch.empty(K, hid // 2, in_dim + hid + cond_dim))
        self.tb1 = nn.Parameter(torch.zeros(K, hid // 2))
        self.tw2 = nn.Parameter(torch.zeros(K, hid, hid // 2))
        self.tb2 = nn.Parameter(torch.zeros(K, hid))
        self.tau0 = nn.Parameter(torch.full((K, 1), float(tau0)))
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


class GASLNet2(nn.Module):
    def __init__(self, in_dim=N_FEAT, attr_dim=N_ATTR, hid=C.HIDDEN,
                 K=C.K_EXPERTS, n_modes=C.N_MODES, use_dt=True, use_attr=True,
                 use_switch=True, use_prop=True, tau0=30.0, rich_dec=True):
        super().__init__()
        self.K = K if use_switch else 1
        self.use_dt, self.use_attr, self.hid = use_dt, use_attr, hid
        self.use_switch, self.use_prop, self.rich_dec = use_switch, use_prop, rich_dec
        cond_dim = (3 if use_dt else 0) + (attr_dim if use_attr else 0)
        self.cond_dim, self.attr_dim = max(cond_dim, 1), attr_dim
        self.cell = VecLiquid(in_dim, hid, self.cond_dim, self.K, tau0=tau0)
        self.gate = nn.Linear(in_dim + hid + self.cond_dim, self.K)
        state_dim = (2 * hid if rich_dec else hid) + in_dim
        self.head = TimeCondHead(state_dim, attr_dim, n_modes)

    def _cond(self, dt, attr):
        parts = []
        if self.use_dt:
            parts.append(phi_dt(dt))
        if self.use_attr:
            parts.append(attr)
        if not parts:
            return dt.new_zeros(dt.shape[0], 1)
        return torch.cat(parts, -1)

    def forward(self, f, dt, mask, attr, lead, tgap, return_tau=False):
        B, T, _ = f.shape
        h = f.new_zeros(B, self.K, self.hid)
        taus = []
        for t in range(T):
            x = f[:, t]
            c = self._cond(dt[:, t], attr)
            hs, tau = self.cell(x, h, dt[:, t], c, self.use_dt)
            m = mask[:, t].float().view(B, 1, 1)
            h = m * hs + (1 - m) * h
            if return_tau:
                taus.append(tau.mean(-1))
        x_last = last_step(f, mask)
        c_last = self._cond(tgap.clamp(min=1e-2), attr)
        pi = torch.softmax(self.gate(torch.cat([x_last, h.mean(1), c_last], -1)), -1)
        h_pre = (pi.unsqueeze(-1) * h).sum(1)          # state at the anchor
        if self.use_prop:                              # blackout propagation over tgap
            h, tau_p = self.cell(x_last, h, tgap, c_last, self.use_dt)
        h_mix = (pi.unsqueeze(-1) * h).sum(1)          # state at prediction origin
        z = torch.cat([h_mix, h_pre, x_last], -1) if self.rich_dec else \
            torch.cat([h_mix, x_last], -1)
        mu, lg, sig = self.head(z, attr, lead)
        if return_tau:
            self.last_tau = torch.stack(taus, 1).detach()
            self.last_gate = pi.detach()
        return mu, lg, sig


# --------------------------------------------------------------------------
# baselines v2 (same time-conditioned head, same x_last summary)
# --------------------------------------------------------------------------
class LiquidCell(nn.Module):
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


class LSTMPred2(nn.Module):
    def __init__(self, hid=64, attr_dim=N_ATTR):
        super().__init__()
        self.rnn = nn.LSTM(N_FEAT, hid, 1, batch_first=True)
        self.head = TimeCondHead(hid + N_FEAT, attr_dim)

    def forward(self, f, dt, mask, attr, lead, tgap):
        out, _ = self.rnn(f)
        h_last = last_step(out, mask)
        z = torch.cat([h_last, last_step(f, mask)], -1)
        return self.head(z, attr, lead)


class GRUPred2(LSTMPred2):
    def __init__(self, hid=64, attr_dim=N_ATTR):
        nn.Module.__init__(self)
        self.rnn = nn.GRU(N_FEAT, hid, 1, batch_first=True)
        self.head = TimeCondHead(hid + N_FEAT, attr_dim)


class Seq2SeqPred2(nn.Module):
    def __init__(self, hid=64, attr_dim=N_ATTR):
        super().__init__()
        self.enc = nn.LSTM(N_FEAT, hid, 1, batch_first=True)
        self.head = TimeCondHead(hid + N_FEAT, attr_dim)

    def forward(self, f, dt, mask, attr, lead, tgap):
        out, _ = self.enc(f)
        h_last = last_step(out, mask)
        z = torch.cat([h_last, last_step(f, mask)], -1)
        return self.head(z, attr, lead)


class TransformerPred2(nn.Module):
    def __init__(self, d=64, nhead=4, layers=2, attr_dim=N_ATTR):
        super().__init__()
        self.proj = nn.Linear(N_FEAT, d)
        self.pos = nn.Parameter(torch.zeros(1, C.MAX_OBS_STEPS, d))
        nn.init.normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d, nhead, 128, C.DROPOUT, batch_first=True,
                                         norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = TimeCondHead(d + N_FEAT, attr_dim)

    def forward(self, f, dt, mask, attr, lead, tgap):
        x = self.proj(f) + self.pos[:, :f.shape[1]]
        x = self.enc(x, src_key_padding_mask=~mask)
        h_last = last_step(x, mask)
        z = torch.cat([h_last, last_step(f, mask)], -1)
        return self.head(z, attr, lead)


class VanillaCfC2(nn.Module):
    def __init__(self, hid=64, attr_dim=N_ATTR):
        super().__init__()
        self.cell = LiquidCell(N_FEAT, hid, attr_dim, use_dt=False)
        self.head = TimeCondHead(hid + N_FEAT, attr_dim)

    def forward(self, f, dt, mask, attr, lead, tgap):
        h = f.new_zeros(f.shape[0], self.cell.hid)
        for t in range(f.shape[1]):
            hn, _ = self.cell(f[:, t], h, torch.ones_like(dt[:, t]), attr)
            m = mask[:, t].unsqueeze(-1).float()
            h = m * hn + (1 - m) * h
        z = torch.cat([h, last_step(f, mask)], -1)
        return self.head(z, attr, lead)


class LTCPred2(nn.Module):
    def __init__(self, hid=64, attr_dim=N_ATTR):
        super().__init__()
        self.hid = hid
        self.wx = nn.Linear(N_FEAT, hid)
        self.wh = nn.Linear(hid, hid, bias=False)
        self.tau_n = nn.Linear(N_FEAT + hid + attr_dim, hid)
        self.a_n = nn.Linear(hid, hid)
        self.head = TimeCondHead(hid + N_FEAT, attr_dim)

    def forward(self, f, dt, mask, attr, lead, tgap):
        h = f.new_zeros(f.shape[0], self.hid)
        for t in range(f.shape[1]):
            x = f[:, t]
            s = torch.sigmoid(self.wx(x) + self.wh(h))
            tau = torch.nn.functional.softplus(self.tau_n(torch.cat([x, h, attr], -1))) + 0.1
            a = torch.tanh(self.a_n(h))
            hn = (h + s * a) / (1.0 + (1.0 / tau + s))
            m = mask[:, t].unsqueeze(-1).float()
            h = m * hn + (1 - m) * h
        z = torch.cat([h, last_step(f, mask)], -1)
        return self.head(z, attr, lead)


class NeuralODEPred2(nn.Module):
    def __init__(self, hid=64, attr_dim=N_ATTR):
        super().__init__()
        self.hid = hid
        self.f = nn.Sequential(nn.Linear(N_FEAT + hid, hid), nn.Tanh(),
                               nn.Linear(hid, hid))
        self.h0 = nn.Linear(N_FEAT + attr_dim, hid)
        self.head = TimeCondHead(hid + N_FEAT, attr_dim)

    def forward(self, f, dt, mask, attr, lead, tgap):
        h = torch.tanh(self.h0(torch.cat([f[:, 0], attr], -1)))
        for t in range(f.shape[1]):
            x = f[:, t]
            for _ in range(2):
                h = h + 0.5 * self.f(torch.cat([x, h], -1))
            m = mask[:, t].unsqueeze(-1).float()
            h = m * h + (1 - m) * h
        z = torch.cat([h, last_step(f, mask)], -1)
        return self.head(z, attr, lead)


def build2(name, **kw):
    table = dict(gaslnet2=GASLNet2, lstm2=LSTMPred2, gru2=GRUPred2,
                 seq2seq2=Seq2SeqPred2, transformer2=TransformerPred2,
                 cfc2=VanillaCfC2, ltc2=LTCPred2, ode2=NeuralODEPred2)
    return table[name](**kw)


def view_of2(name):
    return "irr" if name.startswith("gaslnet2") else "uni"
