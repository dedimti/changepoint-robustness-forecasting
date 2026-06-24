#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Cross-dataset validation of the changepoint robustness ratio (rho) and the
causal local-shift augmentation on the M5 (Walmart) retail-demand dataset.

Companion experiment for the manuscript:
  "A Changepoint Robustness Metric and Regime-Shift Augmentation for
   Pharmaceutical Demand Forecasting" (submitted to Knowledge-Based Systems).

WHAT THIS SCRIPT DOES
  1. Loads the public M5 daily unit-sales data (sales_train_validation.csv).
  2. Aggregates daily -> weekly (273 weeks) and daily -> monthly (~68 months).
  3. Detects changepoints with PELT (l2 cost) and keeps changepoint-heavy series.
  4. Trains a single GLOBAL LSTM forecaster, comparing:
        (a) value-only input              [baseline]
        (b) value + causal local-shift     [proposed augmentation]
  5. Reports MAE, MAE_cp, MAE_normal, rho, and a Diebold-Mariano test on the
     critical-region squared errors. All metrics are computed on standardized
     series, exactly as in the Medicaid experiments of the paper.

KEY FINDING (reproduced below in the RESULTS block):
  * rho reproduces the changepoint-fragility pattern on M5 in BOTH resolutions
    (weekly rho = 1.504, monthly rho = 1.545), confirming the DIAGNOSTIC METRIC
    generalises across domains.
  * The local-shift AUGMENTATION does NOT transfer to retail demand: it slightly
    increases changepoint error (weekly DM = -3.47; monthly DM = -3.64), because
    M5 changepoints are dominated by transient promotional/holiday spikes rather
    than the persistent level shifts seen in policy-driven pharmaceutical demand.
  This scope condition is reported honestly in Section 5.15 of the manuscript.

DATA
  Download the M5 competition data (Kaggle: m5-forecasting-accuracy) and place
  sales_train_validation.csv in the path given by --data, or pass --data to a
  directory that already contains it.

USAGE
  python m5_changepoint_validation.py --data ./m5-forecasting-accuracy \
                                      --resolution weekly --n_series 1000
  python m5_changepoint_validation.py --data ./m5-forecasting-accuracy \
                                      --resolution monthly --n_series 1000

REQUIREMENTS
  numpy, pandas, torch, ruptures, scipy   (see requirements note at bottom)

REPRODUCIBILITY
  Fixed seed (default 2024). Exact numbers may vary by a few thousandths with
  library/hardware differences; the qualitative findings are stable.

Author: Dedi Irawan, Sudarmaji, Guna Yanti Kemala Sari Siregar
        Fakultas Ilmu Komputer, Universitas Muhammadiyah Metro, Indonesia
License: MIT
================================================================================
"""

import os
import json
import time
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import ruptures as rpt
from scipy import stats


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def get_args():
    p = argparse.ArgumentParser(description="M5 cross-dataset validation of rho "
                                            "and the causal local-shift augmentation.")
    p.add_argument("--data", type=str, default="./m5-forecasting-accuracy",
                   help="Directory containing sales_train_validation.csv")
    p.add_argument("--resolution", choices=["weekly", "monthly"], default="weekly",
                   help="Temporal aggregation level.")
    p.add_argument("--n_series", type=int, default=1000,
                   help="Number of changepoint-heavy series to use.")
    p.add_argument("--lookback", type=int, default=12, help="Input window length L.")
    p.add_argument("--horizon", type=int, default=None,
                   help="Forecast horizon H (default: 4 weekly, 3 monthly).")
    p.add_argument("--half_width", type=int, default=1,
                   help="Critical-neighbourhood half-width w (in periods).")
    p.add_argument("--pelt_penalty", type=float, default=3.0, help="PELT penalty beta.")
    p.add_argument("--min_size", type=int, default=3, help="PELT minimum segment size.")
    p.add_argument("--min_cps", type=int, default=3,
                   help="Keep series with at least this many changepoints.")
    p.add_argument("--min_mean", type=float, default=5.0,
                   help="Minimum mean demand to treat a series as active.")
    p.add_argument("--min_train_sd", type=float, default=1.0,
                   help="Drop series whose train portion is (near-)constant.")
    p.add_argument("--train_frac", type=float, default=0.8, help="Train fraction.")
    p.add_argument("--hidden", type=int, default=32, help="LSTM hidden size.")
    p.add_argument("--epochs", type=int, default=40, help="Training epochs.")
    p.add_argument("--lr", type=float, default=1e-2, help="Adam learning rate.")
    p.add_argument("--batch_size", type=int, default=2048, help="Mini-batch size.")
    p.add_argument("--seed", type=int, default=2024, help="Random seed.")
    p.add_argument("--threads", type=int, default=2, help="Torch CPU threads.")
    p.add_argument("--out", type=str, default="m5_results.json",
                   help="Where to write the JSON results.")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Data loading and aggregation
# --------------------------------------------------------------------------- #
def load_and_aggregate(data_dir, resolution, min_mean):
    """Load M5 daily sales and aggregate to weekly or monthly demand series."""
    path = os.path.join(data_dir, "sales_train_validation.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Could not find {path}. Download the M5 data (Kaggle: "
            f"m5-forecasting-accuracy) and point --data at its folder.")
    df = pd.read_csv(path)
    day_cols = [c for c in df.columns if c.startswith("d_")]
    daily = df[day_cols].values.astype(np.float32)            # (30490, 1913)

    if resolution == "weekly":
        period = 7
    else:                                                     # monthly ~ 4 weeks
        # first go to weeks, then group 4 weeks into a month (matches the paper)
        n_weeks = daily.shape[1] // 7
        weekly = daily[:, :n_weeks * 7].reshape(daily.shape[0], n_weeks, 7).sum(2)
        n_months = weekly.shape[1] // 4
        agg = weekly[:, :n_months * 4].reshape(weekly.shape[0], n_months, 4).sum(2)
        active = (agg.mean(1) >= min_mean) & np.all(np.isfinite(agg), axis=1)
        return agg[active]

    n = daily.shape[1] // period
    agg = daily[:, :n * period].reshape(daily.shape[0], n, period).sum(2)
    active = (agg.mean(1) >= min_mean) & np.all(np.isfinite(agg), axis=1)
    return agg[active]


# --------------------------------------------------------------------------- #
# Changepoint detection (PELT) and causal feature construction
# --------------------------------------------------------------------------- #
def detect_changepoints(x, penalty, min_size):
    """PELT (l2) changepoints on a z-scored series; endpoint dropped."""
    z = (x - x.mean()) / (x.std() + 1e-8)
    try:
        bkps = rpt.Pelt(model="l2", min_size=min_size).fit(z).predict(pen=penalty)
        return [b for b in bkps if b < len(x)]
    except Exception:
        return []


def causal_local_shift(xz):
    """Causal local-shift feature s_t = mean(x[t-2:t]) - mean(x[t-5:t-3])."""
    s = np.zeros_like(xz)
    for t in range(5, len(xz)):
        s[t] = xz[t - 2:t + 1].mean() - xz[t - 5:t - 2].mean()
    return s


def select_series(W, args):
    """Select changepoint-heavy, non-degenerate series; precompute features."""
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(W.shape[0])

    heavy = []
    for i in order:
        x = W[i].astype(np.float32)
        ntr = int(len(x) * args.train_frac)
        if x[:ntr].std() < args.min_train_sd:           # drop near-constant train part
            continue
        if len(detect_changepoints(x, args.pelt_penalty, args.min_size)) >= args.min_cps:
            heavy.append(i)
        if len(heavy) >= args.n_series:
            break

    series = {}
    for i in heavy:
        x = W[i].astype(np.float32)
        T = len(x)
        ntr = int(T * args.train_frac)
        mu, sd = x[:ntr].mean(), x[:ntr].std() + 1e-8
        xz = np.clip((x - mu) / sd, -10, 10)            # standardize (train stats), clip
        sh = causal_local_shift(xz)
        crit = set()
        for c in detect_changepoints(x, args.pelt_penalty, args.min_size):
            for d in range(-args.half_width, args.half_width + 1):
                crit.add(c + d)
        series[i] = dict(xz=xz, sh=sh, ntr=ntr, crit=crit, T=T)
    return heavy, series


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class GlobalLSTM(nn.Module):
    """Single-layer global LSTM forecaster with a linear multi-step head."""
    def __init__(self, n_in, hidden, horizon):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def build_windows(series_ids, series, L, H, use_shift):
    """Build sliding windows; returns arrays and per-window metadata."""
    X, Y, meta_i, meta_t, is_test = [], [], [], [], []
    for i in series_ids:
        s = series[i]
        xz, sh, ntr, T = s["xz"], s["sh"], s["ntr"], s["T"]
        for t in range(L, T - H + 1):
            feat = np.stack([xz[t - L:t], sh[t - L:t]], axis=1) if use_shift \
                else xz[t - L:t][:, None]
            X.append(feat)
            Y.append(xz[t:t + H])
            meta_i.append(i)
            meta_t.append(t)
            is_test.append(t >= ntr)
    return (np.asarray(X, np.float32), np.asarray(Y, np.float32),
            np.asarray(meta_i), np.asarray(meta_t), np.asarray(is_test, bool))


def train_and_evaluate(series_ids, series, args, use_shift, tag):
    """Train the global LSTM and compute MAE, MAE_cp, MAE_normal, rho."""
    X, Y, MI, MT, MTest = build_windows(series_ids, series, args.lookback,
                                        args.horizon, use_shift)
    tr_idx = np.where(~MTest)[0]
    te_idx = np.where(MTest)[0]

    torch.manual_seed(args.seed)
    net = GlobalLSTM(X.shape[2], args.hidden, args.horizon)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    net.train()
    for _ in range(args.epochs):
        np.random.shuffle(tr_idx)
        for b in range(0, len(tr_idx), args.batch_size):
            ix = tr_idx[b:b + args.batch_size]
            opt.zero_grad()
            loss = loss_fn(net(torch.from_numpy(X[ix])), torch.from_numpy(Y[ix]))
            loss.backward()
            opt.step()

    net.eval()
    preds = np.empty((len(te_idx), args.horizon), np.float32)
    with torch.no_grad():
        for b in range(0, len(te_idx), 4096):
            ix = te_idx[b:b + 4096]
            preds[b:b + len(ix)] = net(torch.from_numpy(X[ix])).numpy()

    Yte, Ite, Tte = Y[te_idx], MI[te_idx], MT[te_idx]
    e_cp, e_normal, e_all, crit_sq = [], [], [], {}
    for k in range(len(preds)):
        i, t = int(Ite[k]), int(Tte[k])
        crit = series[i]["crit"]
        for h in range(args.horizon):
            period = t + h
            err = abs(preds[k, h] - Yte[k, h])
            e_all.append(err)
            if period in crit:
                e_cp.append(err)
                crit_sq[(i, period)] = crit_sq.get((i, period), 0.0) \
                    + float((preds[k, h] - Yte[k, h]) ** 2)
            else:
                e_normal.append(err)

    mae = float(np.mean(e_all))
    mae_cp = float(np.mean(e_cp))
    mae_normal = float(np.mean(e_normal))
    res = dict(tag=tag, mae=mae, mae_cp=mae_cp, mae_normal=mae_normal,
               rho=mae_cp / mae_normal, n_cp_points=len(e_cp),
               n_series=len(series_ids))
    return res, crit_sq


def diebold_mariano(sq_base, sq_prop):
    """DM statistic on paired critical-region squared errors (base - proposed)."""
    keys = [k for k in sq_base if k in sq_prop]
    d = np.array([sq_base[k] - sq_prop[k] for k in keys])     # >0 => proposed better
    dm = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
    try:
        _, wilcoxon_p = stats.wilcoxon(d)
        wilcoxon_p = float(wilcoxon_p)
    except Exception:
        wilcoxon_p = float("nan")
    return dm, wilcoxon_p, len(keys)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = get_args()
    if args.horizon is None:
        args.horizon = 4 if args.resolution == "weekly" else 3
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    t0 = time.time()
    print(f"[1/4] Loading M5 and aggregating to {args.resolution} demand ...")
    W = load_and_aggregate(args.data, args.resolution, args.min_mean)
    print(f"      active series: {W.shape[0]} x {W.shape[1]} periods "
          f"({time.time() - t0:.0f}s)")

    print(f"[2/4] Detecting changepoints (PELT) and selecting "
          f">= {args.min_cps}-changepoint series ...")
    series_ids, series = select_series(W, args)
    print(f"      selected series: {len(series_ids)} ({time.time() - t0:.0f}s)")

    print("[3/4] Training baseline (value-only) ...")
    base, base_sq = train_and_evaluate(series_ids, series, args,
                                       use_shift=False, tag="baseline")
    print(f"      [baseline ] MAE={base['mae']:.3f}  MAE_cp={base['mae_cp']:.3f}  "
          f"MAE_normal={base['mae_normal']:.3f}  rho={base['rho']:.3f}  "
          f"({time.time() - t0:.0f}s)")

    print("[4/4] Training proposed (value + causal local-shift) ...")
    prop, prop_sq = train_and_evaluate(series_ids, series, args,
                                       use_shift=True, tag="proposed")
    print(f"      [proposed ] MAE={prop['mae']:.3f}  MAE_cp={prop['mae_cp']:.3f}  "
          f"MAE_normal={prop['mae_normal']:.3f}  rho={prop['rho']:.3f}  "
          f"({time.time() - t0:.0f}s)")

    dm, wilcoxon_p, n_pairs = diebold_mariano(base_sq, prop_sq)
    sign = "proposed better" if dm > 0 else "baseline better"
    print("\n================== SUMMARY ==================")
    print(f"Resolution         : {args.resolution}")
    print(f"Series (>= {args.min_cps} cps)  : {len(series_ids)}")
    print(f"Baseline rho       : {base['rho']:.3f}  (MAE_cp={base['mae_cp']:.3f})")
    print(f"Proposed rho       : {prop['rho']:.3f}  (MAE_cp={prop['mae_cp']:.3f})")
    print(f"DM (base-proposed) : {dm:+.2f}  [{sign}]  "
          f"n={n_pairs}  Wilcoxon p={wilcoxon_p:.2e}")
    print("=============================================")

    results = dict(resolution=args.resolution, config=vars(args),
                   baseline=base, proposed=prop,
                   dm=dm, wilcoxon_p=wilcoxon_p, n_pairs=n_pairs)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.out}  (total {time.time() - t0:.0f}s)")


# --------------------------------------------------------------------------- #
# Reference results obtained by the authors (1,000 changepoint-heavy series,
# seed 2024). Reproduced here for documentation; rerun the script to verify.
#
#   M5 WEEKLY
#     baseline : MAE=0.636  MAE_cp=0.919  MAE_normal=0.611  rho=1.504
#     proposed : MAE=0.644  MAE_cp=0.927  MAE_normal=0.620  rho=1.497
#     DM(base-proposed) = -3.47   (n=4335,  Wilcoxon p = 7.65e-01)
#
#   M5 MONTHLY
#     baseline : MAE=0.746  MAE_cp=1.090  MAE_normal=0.705  rho=1.545
#     proposed : MAE=0.770  MAE_cp=1.111  MAE_normal=0.730  rho=1.522
#     DM(base-proposed) = -3.64   (n=1500,  Wilcoxon p = 7.42e-06)
#
# INTERPRETATION
#   rho reproduces the changepoint-fragility pattern in an independent domain
#   (the DIAGNOSTIC generalises), whereas the local-shift AUGMENTATION does not
#   transfer to retail demand, whose changepoints are dominated by transient
#   spikes rather than the persistent level shifts of pharmaceutical demand.
#   See Section 5.15 of the manuscript.
#
# requirements.txt
#   numpy>=1.24
#   pandas>=2.0
#   torch>=2.0
#   ruptures>=1.1.9
#   scipy>=1.10
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
