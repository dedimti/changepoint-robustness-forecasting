#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Controlled experiment: WHEN does the causal local-shift augmentation help?

Hypothesis (from the M5 vs Medicaid contrast in the paper): the augmentation
benefits forecasting at changepoints to the extent that changepoints are
PERSISTENT level shifts rather than TRANSIENT spikes. We vary a persistence
parameter p in [0,1] (p=1: shift fully retained; p=0: shift fully decays back,
i.e. a transient spike) and measure the augmentation's reduction in MAE_cp.

Reuses the repository's causal_local_shift, sliding-window, and GlobalLSTM logic
verbatim so the mechanism matches the manuscript.
"""
import numpy as np
import torch
import torch.nn as nn
import ruptures as rpt

SEED = 7
rng = np.random.default_rng(SEED)

# ---- repo-faithful pieces -------------------------------------------------
def detect_changepoints(x, penalty=3.0, min_size=4):
    z = (x - x.mean()) / (x.std() + 1e-8)
    try:
        bkps = rpt.Pelt(model="l2", min_size=min_size).fit(z).predict(pen=penalty)
        return [b for b in bkps if b < len(x)]
    except Exception:
        return []

def causal_local_shift(xz):
    s = np.zeros_like(xz)
    for t in range(5, len(xz)):
        s[t] = xz[t-2:t+1].mean() - xz[t-5:t-2].mean()
    return s

class GlobalLSTM(nn.Module):
    def __init__(self, n_in, hidden, horizon):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, horizon)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

# ---- synthetic data with controllable persistence ------------------------
def make_series(persistence, T=120, n_breaks=3, noise=0.30):
    """Piecewise series. At each break, level jumps by delta; the jump then
    relaxes toward baseline with retained fraction = persistence over ~6 steps.
    persistence=1 -> permanent shift; persistence=0 -> transient spike."""
    x = np.zeros(T, np.float32)
    base = 0.0
    # choose break points spaced out
    cps = sorted(rng.choice(range(15, T-15), size=n_breaks, replace=False))
    seg_level = base
    cur = base
    true_cps = []
    decay_len = 6
    i = 0
    pending = []  # list of (start, delta) transient components
    levels = np.full(T, base, np.float32)
    lvl = base
    for t in range(T):
        if t in cps:
            delta = rng.uniform(2.5, 4.0) * rng.choice([-1, 1])
            # permanent part retained = persistence*delta; transient part = (1-persistence)*delta decays
            lvl = lvl + persistence * delta
            pending.append([t, (1.0 - persistence) * delta])
            true_cps.append(t)
        trans = 0.0
        for comp in pending:
            st, d = comp
            age = t - st
            if age < decay_len:
                trans += d * (1.0 - age / decay_len)
        levels[t] = lvl + trans
    x = levels + rng.normal(0, noise, T).astype(np.float32)
    return x, true_cps

def build_windows(series, L, H, use_shift, half_width=2, pelt_penalty=3.0):
    X, Y, meta_i, meta_t, is_test, crit_by = [], [], [], [], [], {}
    for i, (x, _) in enumerate(series):
        T = len(x); ntr = int(T*0.7)
        mu, sd = x[:ntr].mean(), x[:ntr].std()+1e-8
        xz = np.clip((x-mu)/sd, -10, 10).astype(np.float32)
        sh = causal_local_shift(xz)
        crit = set()
        for c in detect_changepoints(x, pelt_penalty):
            for d in range(-half_width, half_width+1):
                crit.add(c+d)
        crit_by[i] = crit
        for t in range(L, T-H+1):
            feat = np.stack([xz[t-L:t], sh[t-L:t]], axis=1) if use_shift else xz[t-L:t][:,None]
            X.append(feat); Y.append(xz[t:t+H]); meta_i.append(i); meta_t.append(t)
            is_test.append(t>=ntr)
    return (np.asarray(X,np.float32), np.asarray(Y,np.float32),
            np.asarray(meta_i), np.asarray(meta_t), np.asarray(is_test,bool), crit_by)

def train_eval(series, use_shift, L=12, H=4, hidden=32, epochs=40, lr=1e-2, bs=256):
    X,Y,MI,MT,MTest,crit_by = build_windows(series, L, H, use_shift)
    tr = np.where(~MTest)[0]; te = np.where(MTest)[0]
    torch.manual_seed(SEED)
    net = GlobalLSTM(X.shape[2], hidden, H)
    opt = torch.optim.Adam(net.parameters(), lr=lr); lf = nn.MSELoss()
    net.train()
    for _ in range(epochs):
        np.random.shuffle(tr)
        for b in range(0, len(tr), bs):
            ix = tr[b:b+bs]; opt.zero_grad()
            loss = lf(net(torch.from_numpy(X[ix])), torch.from_numpy(Y[ix]))
            loss.backward(); opt.step()
    net.eval()
    preds = np.empty((len(te), H), np.float32)
    with torch.no_grad():
        for b in range(0, len(te), 4096):
            ix = te[b:b+4096]; preds[b:b+len(ix)] = net(torch.from_numpy(X[ix])).numpy()
    Yte, Ite, Tte = Y[te], MI[te], MT[te]
    e_cp, e_no = [], []
    for k in range(len(preds)):
        i,t = int(Ite[k]), int(Tte[k]); crit = crit_by[i]
        for h in range(H):
            err = abs(preds[k,h]-Yte[k,h])
            (e_cp if (t+h) in crit else e_no).append(err)
    mae_cp = float(np.mean(e_cp)); mae_no = float(np.mean(e_no))
    return mae_cp, mae_no, mae_cp/mae_no

# ---- sweep persistence ----------------------------------------------------
print(f"{'persist':>8} {'base_cp':>8} {'aug_cp':>8} {'Δ%':>7} {'rho_base':>9} {'rho_aug':>8}")
rows = []
N_SERIES = 200
for p in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    series = [make_series(p) for _ in range(N_SERIES)]
    b_cp, b_no, b_rho = train_eval(series, use_shift=False)
    a_cp, a_no, a_rho = train_eval(series, use_shift=True)
    impr = 100.0*(b_cp - a_cp)/b_cp
    rows.append((p, b_cp, a_cp, impr, b_rho, a_rho))
    print(f"{p:>8.1f} {b_cp:>8.3f} {a_cp:>8.3f} {impr:>6.1f}% {b_rho:>9.3f} {a_rho:>8.3f}")

import json
json.dump([{"persistence":r[0],"base_mae_cp":r[1],"aug_mae_cp":r[2],
            "improvement_pct":r[3],"rho_base":r[4],"rho_aug":r[5]} for r in rows],
          open("persistence_sweep.json","w"), indent=1)
print("\nsaved persistence_sweep.json")
