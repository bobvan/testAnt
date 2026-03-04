#!/usr/bin/env python3
"""
export_qerr_debug.py — Export qErr debug CSV for gnuplot inspection.

Outputs one row per aligned epoch with:
  epoch               sequential integer
  utc_time            wall-clock UTC (from TOP TIM-TP)
  qerr_top_ps         raw TIM-TP qErr from TOP receiver (ps)
  qerr_bot_ps         raw TIM-TP qErr from BOT receiver (ps)
  qerr_top_smooth_ps  rolling-median of qerr_top (--smooth epochs)
  qerr_bot_smooth_ps  rolling-median of qerr_bot
  qerr_diff_ps        qerr_top - qerr_bot  (what the correction actually subtracts)
  ticc_interval_a_ps  chA PPS interval deviation from 1 s, in ps
  ticc_interval_b_ps  chB PPS interval deviation from 1 s, in ps
  ticc_cumphase_a_ps  cumulative phase walk from chA intervals (ps) — anticipated sawtooth
  ticc_cumphase_b_ps  cumulative phase walk from chB intervals (ps) — anticipated sawtooth
  ticc_cumphase_smooth_a_ps  long-window smooth of cumphase_a (--long-smooth epochs)
  ticc_cumphase_smooth_b_ps  long-window smooth of cumphase_b
  raw_diff_ns         raw chA − chB TICC difference (ns)

The TICC cumulative phase walk is computed from consecutive PPS interval
deviations: (chA[n] - chA[n-1] - 1.0) * 1e12 ps, summed.  This is an
independent (receiver-agnostic) view of oscillator phase walk that should
track each receiver's qErr sawtooth if the correction is working correctly.

Usage:
    python scripts/export_qerr_debug.py \\
        --ticc  data/foo_ticc.csv  \\
        --timtp data/foo_timtp.csv \\
        --out   data/foo_qerr_debug.csv \\
        [--smooth 300] [--long-smooth 1200]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ── loading (mirrors analyze_pps.py) ────────────────────────────────────── #

def load_ticc(path: Path) -> pd.DataFrame:
    """
    Load TICC CSV, pair chA/chB by integer second.
    Returns DataFrame: integer_sec, chA_ts, chB_ts, raw_diff_ns
    """
    _BOUNDARY_GUARD_S = 100e-9

    df = pd.read_csv(path)
    df["integer_sec"] = df["timestamp_s"].astype(int)
    frac = df["timestamp_s"] - df["integer_sec"]
    bad = (frac < _BOUNDARY_GUARD_S) | (frac > 1.0 - _BOUNDARY_GUARD_S)
    if bad.any():
        raise ValueError(f"TICC: {bad.sum()} edge(s) within 100 ns of second boundary")

    piv = (
        df.pivot_table(index="integer_sec", columns="channel",
                       values="timestamp_s", aggfunc="first")
          .rename(columns={"chA": "chA_ts", "chB": "chB_ts"})
          .dropna()
          .reset_index()
          .sort_values("integer_sec")
          .reset_index(drop=True)
    )
    piv["raw_diff_ns"] = (piv["chA_ts"] - piv["chB_ts"]) * 1e9
    return piv


def load_timtp(path: Path) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["tow_s"] = (df["tow_ms"] // 1000).astype(int)
    return {
        rx: grp.sort_values("timestamp").reset_index(drop=True)
        for rx, grp in df.groupby("receiver")
    }


# ── GPS-second join (mirrors analyze_pps.py) ────────────────────────────── #

def gps_join(ticc: pd.DataFrame,
             top_df: pd.DataFrame,
             bot_df: pd.DataFrame,
             gps_offset: int) -> pd.DataFrame:
    df = ticc.copy()
    df["gps_sec"] = df["integer_sec"] + gps_offset

    top_q  = top_df.set_index("tow_s")["qerr_ps"]
    bot_q  = bot_df.set_index("tow_s")["qerr_ps"]
    top_ts = top_df.set_index("tow_s")["timestamp"]

    corr_tow = df["gps_sec"] - 1   # TIM-TP at S-1 predicts PPS edge at S
    df["qerr_top_ps"] = corr_tow.map(top_q)
    df["qerr_bot_ps"] = corr_tow.map(bot_q)
    df["utc_time"]    = corr_tow.map(top_ts)
    return df.dropna(subset=["qerr_top_ps", "qerr_bot_ps"]).reset_index(drop=True)


def best_gps_offset(ticc: pd.DataFrame,
                    timtp: dict[str, pd.DataFrame]) -> tuple[int, int]:
    """Return (gps_offset, sign) that minimises corrected-diff std."""
    top = timtp["TOP"]
    bot = timtp["BOT"]
    naive = int(top["tow_s"].iloc[0]) - int(ticc["integer_sec"].iloc[0])

    best_std = np.inf
    best_offset, best_sign = naive, +1

    for delta in (-1, 0, +1, +2):
        joined = gps_join(ticc, top, bot, naive + delta)
        if joined.empty:
            continue
        for sign in (+1, -1):
            corr = ((joined["chA_ts"] + sign * joined["qerr_top_ps"] * 1e-12) -
                    (joined["chB_ts"] + sign * joined["qerr_bot_ps"] * 1e-12))
            std = float(corr.std() * 1e9)
            if std < best_std:
                best_std = std
                best_offset, best_sign = naive + delta, sign

    return best_offset, best_sign


# ── TICC cumulative phase walk ───────────────────────────────────────────── #

def cumulative_phase(ts_sorted: np.ndarray,
                     long_window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    From sorted TICC timestamps, compute:
      interval_ps   — per-epoch PPS interval deviation from 1 s (ps)
                      (NaN for the first epoch — no prior interval)
      cumphase_ps   — cumulative sum of interval_ps (phase walk, ps)
      smooth_ps     — rolling-median of cumphase_ps (anticipated sawtooth trend)

    All arrays have the same length as ts_sorted.
    """
    intervals = np.diff(ts_sorted)                 # N-1 values
    deviations_ps = (intervals - 1.0) * 1e12       # deviation from perfect 1 s

    # Prepend NaN so length matches ts_sorted
    interval_ps = np.concatenate([[np.nan], deviations_ps])
    cumphase_ps = np.concatenate([[np.nan], np.cumsum(deviations_ps)])

    smooth_ps = (pd.Series(cumphase_ps)
                   .rolling(long_window, center=True, min_periods=long_window // 4)
                   .median()
                   .values)

    return interval_ps, cumphase_ps, smooth_ps


# ── main ─────────────────────────────────────────────────────────────────── #

def main():
    ap = argparse.ArgumentParser(
        description="Export qErr debug CSV for gnuplot inspection"
    )
    ap.add_argument("--ticc",        required=True, help="_ticc.csv input")
    ap.add_argument("--timtp",       required=True, help="_timtp.csv input")
    ap.add_argument("--out",         required=True, help="Output .csv path")
    ap.add_argument("--smooth",      type=int, default=300,
                    help="Rolling-median window for qErr smoothing (epochs, default 300)")
    ap.add_argument("--long-smooth", type=int, default=1200,
                    help="Rolling-median window for TICC phase smoothing (epochs, default 1200)")
    args = ap.parse_args()

    print(f"Loading TICC  : {args.ticc}")
    ticc = load_ticc(Path(args.ticc))
    print(f"  {len(ticc)} paired epochs")

    print(f"Loading TIM-TP: {args.timtp}")
    timtp = load_timtp(Path(args.timtp))
    for rx, grp in timtp.items():
        print(f"  {rx}: {len(grp)} rows  "
              f"qerr range [{grp['qerr_ps'].min():+d}, {grp['qerr_ps'].max():+d}] ps")

    gps_off, sign = best_gps_offset(ticc, timtp)
    print(f"  Best alignment: gps_offset={gps_off}, sign={sign:+d}")

    df = gps_join(ticc, timtp["TOP"], timtp["BOT"], gps_off)
    print(f"  {len(df)} epochs after join")

    # qErr smoothing (short window — shows receiver sawtooth)
    df["qerr_top_smooth_ps"] = (pd.Series(df["qerr_top_ps"].values)
                                  .rolling(args.smooth, center=True,
                                           min_periods=args.smooth // 4)
                                  .median()
                                  .values)
    df["qerr_bot_smooth_ps"] = (pd.Series(df["qerr_bot_ps"].values)
                                  .rolling(args.smooth, center=True,
                                           min_periods=args.smooth // 4)
                                  .median()
                                  .values)
    df["qerr_diff_ps"] = df["qerr_top_ps"] - df["qerr_bot_ps"]

    # TICC cumulative phase walk (independent oscillator estimate)
    chA_sorted = np.sort(ticc["chA_ts"].values)
    chB_sorted = np.sort(ticc["chB_ts"].values)

    n = len(df)
    intv_a, cum_a, sm_a = cumulative_phase(chA_sorted, args.long_smooth)
    intv_b, cum_b, sm_b = cumulative_phase(chB_sorted, args.long_smooth)

    # Align TICC phase arrays to the joined df length
    df["ticc_interval_a_ps"]       = intv_a[:n]
    df["ticc_interval_b_ps"]       = intv_b[:n]
    df["ticc_cumphase_a_ps"]       = cum_a[:n]
    df["ticc_cumphase_b_ps"]       = cum_b[:n]
    df["ticc_cumphase_smooth_a_ps"] = sm_a[:n]
    df["ticc_cumphase_smooth_b_ps"] = sm_b[:n]

    # Select and order output columns
    out = df[[
        "utc_time",
        "qerr_top_ps",
        "qerr_bot_ps",
        "qerr_top_smooth_ps",
        "qerr_bot_smooth_ps",
        "qerr_diff_ps",
        "ticc_interval_a_ps",
        "ticc_interval_b_ps",
        "ticc_cumphase_a_ps",
        "ticc_cumphase_b_ps",
        "ticc_cumphase_smooth_a_ps",
        "ticc_cumphase_smooth_b_ps",
        "raw_diff_ns",
    ]].copy()
    out.insert(0, "epoch", range(len(out)))

    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows → {args.out}")
    print(f"  Columns: {', '.join(out.columns)}")
    print("Done.")


if __name__ == "__main__":
    main()
