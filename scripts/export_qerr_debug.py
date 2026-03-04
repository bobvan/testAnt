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

    Handles three CSV format generations:
      Gen 1: timestamp_s, channel
      Gen 2: host_timestamp, timestamp_s, channel
      Gen 3: host_timestamp, ref_sec, ref_ps, channel

    Returns DataFrame with int64 columns chA_ref_sec, chA_ref_ps,
    chB_ref_sec, chB_ref_ps, raw_diff_ps, raw_diff_ns, and
    optionally host_sec.
    """
    _BOUNDARY_GUARD_S = 100e-9

    df = pd.read_csv(path)
    cols = set(df.columns)

    if "ref_sec" in cols:                      # Gen 3
        df["ref_sec"] = df["ref_sec"].astype("int64")
        df["ref_ps"]  = df["ref_ps"].astype("int64")
        df["integer_sec"] = df["ref_sec"]
    else:                                       # Gen 1 or 2
        df["integer_sec"] = df["timestamp_s"].astype("int64")
        df["ref_sec"] = df["integer_sec"]
        df["ref_ps"]  = ((df["timestamp_s"] - df["integer_sec"]) * 1e12
                         ).round().astype("int64")

    if "host_timestamp" in cols:
        host_ts = pd.to_datetime(df["host_timestamp"], utc=True)
        df["host_sec"] = (host_ts.astype("int64") // 1_000_000_000).astype(int)

    frac_s = df["ref_ps"] / 1e12
    bad = (frac_s < _BOUNDARY_GUARD_S) | (frac_s > 1.0 - _BOUNDARY_GUARD_S)
    if bad.any():
        raise ValueError(f"TICC: {bad.sum()} edge(s) within 100 ns of second boundary")

    piv_sec = (df.pivot_table(index="integer_sec", columns="channel",
                               values="ref_sec", aggfunc="first")
                 .rename(columns={"chA": "chA_ref_sec", "chB": "chB_ref_sec"}))
    piv_ps  = (df.pivot_table(index="integer_sec", columns="channel",
                               values="ref_ps",  aggfunc="first")
                 .rename(columns={"chA": "chA_ref_ps",  "chB": "chB_ref_ps"}))
    piv = (pd.concat([piv_sec, piv_ps], axis=1)
             .dropna()
             .reset_index()
             .sort_values("integer_sec")
             .reset_index(drop=True))
    for col in ("chA_ref_sec", "chB_ref_sec", "chA_ref_ps", "chB_ref_ps"):
        piv[col] = piv[col].astype("int64")

    piv["raw_diff_ps"] = ((piv["chA_ref_sec"] - piv["chB_ref_sec"])
                          * 1_000_000_000_000
                          + piv["chA_ref_ps"] - piv["chB_ref_ps"])
    piv["raw_diff_ns"] = piv["raw_diff_ps"].astype(float) * 1e-3

    if "host_sec" in df.columns:
        hs_map = df.groupby("integer_sec")["host_sec"].first()
        piv["host_sec"] = piv["integer_sec"].map(hs_map)
    return piv


def load_timtp(path: Path) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["tow_s"] = (df["tow_ms"] // 1000).astype(int)
    df["utc_s"] = (df["timestamp"].astype("int64") // 1_000_000_000).astype(int)
    return {
        rx: grp.sort_values("timestamp").reset_index(drop=True)
        for rx, grp in df.groupby("receiver")
    }


# ── join helpers (mirrors analyze_pps.py) ────────────────────────────────── #

def gps_join(ticc: pd.DataFrame,
             top_df: pd.DataFrame,
             bot_df: pd.DataFrame,
             gps_offset: int) -> pd.DataFrame:
    """GPS-second join fallback (for TICC CSV without host_timestamp)."""
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


def utc_join(ticc: pd.DataFrame,
             top_df: pd.DataFrame,
             bot_df: pd.DataFrame) -> pd.DataFrame:
    """UTC-second join when host_timestamp column is present (preferred)."""
    df = ticc.copy()
    top_q  = top_df.set_index("utc_s")["qerr_ps"]
    bot_q  = bot_df.set_index("utc_s")["qerr_ps"]
    top_ts = top_df.set_index("utc_s")["timestamp"]

    corr_utc = df["host_sec"] - 1
    df["qerr_top_ps"] = corr_utc.map(top_q)
    df["qerr_bot_ps"] = corr_utc.map(bot_q)
    df["utc_time"]    = corr_utc.map(top_ts)
    return df.dropna(subset=["qerr_top_ps", "qerr_bot_ps"]).reset_index(drop=True)


def best_gps_offset(ticc: pd.DataFrame,
                    timtp: dict[str, pd.DataFrame]) -> tuple[int, int]:
    """
    Return (gps_offset, sign) that minimises corrected-diff std.
    Uses UTC join when host_timestamp present; GPS offset search otherwise.
    """
    top = timtp["TOP"]
    bot = timtp["BOT"]

    if "host_sec" in ticc.columns and "utc_s" in top.columns:
        joined = utc_join(ticc, top, bot)
        best_std = np.inf
        best_sign = +1
        for sign in (+1, -1):
            corr_ps = (joined["raw_diff_ps"]
                       + sign * (joined["qerr_top_ps"] - joined["qerr_bot_ps"]))
            std = float(corr_ps.std() * 1e-3)   # ps → ns
            if std < best_std:
                best_std = std
                best_sign = sign
        return 0, best_sign   # offset unused by utc_join, 0 is a sentinel

    naive = int(top["tow_s"].iloc[0]) - int(ticc["integer_sec"].iloc[0])
    best_std = np.inf
    best_offset, best_sign = naive, +1

    for delta in (-1, 0, +1, +2):
        joined = gps_join(ticc, top, bot, naive + delta)
        if joined.empty:
            continue
        for sign in (+1, -1):
            corr_ps = (joined["raw_diff_ps"]
                       + sign * (joined["qerr_top_ps"] - joined["qerr_bot_ps"]))
            std = float(corr_ps.std() * 1e-3)   # ps → ns
            if std < best_std:
                best_std = std
                best_offset, best_sign = naive + delta, sign

    return best_offset, best_sign


# ── TICC cumulative phase walk ───────────────────────────────────────────── #

def cumulative_phase(ref_sec: np.ndarray, ref_ps: np.ndarray,
                     long_window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    From TICC int64 ref_sec/ref_ps arrays (already sorted by epoch), compute:
      interval_ps   — per-epoch PPS interval deviation from 1 s (ps)
                      (NaN for the first epoch — no prior interval)
      cumphase_ps   — cumulative sum of interval_ps (phase walk, ps)
      smooth_ps     — rolling-median of cumphase_ps (anticipated sawtooth trend)

    All arrays have the same length as ref_sec/ref_ps.
    Using int64 arithmetic avoids float64 precision loss at long TICC uptimes.
    """
    # interval deviation from 1 s, in ps:
    # (ref_sec[n] - ref_sec[n-1] - 1) * 1e12 + (ref_ps[n] - ref_ps[n-1])
    sec_diff = np.diff(ref_sec.astype("int64"))    # expected: 1 for no gap
    ps_diff  = np.diff(ref_ps.astype("int64"))
    deviations_ps = (sec_diff - 1) * 1_000_000_000_000 + ps_diff  # int64

    # Prepend NaN so length matches input arrays
    dev_f = deviations_ps.astype(float)
    interval_ps = np.concatenate([[np.nan], dev_f])
    cumphase_ps = np.concatenate([[np.nan], np.cumsum(dev_f)])

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
    use_utc = "host_sec" in ticc.columns and "utc_s" in timtp["TOP"].columns
    if use_utc:
        print(f"  Best alignment: UTC join, sign={sign:+d}")
        df = utc_join(ticc, timtp["TOP"], timtp["BOT"])
    else:
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
    # ticc is already sorted by integer_sec from load_ticc(), so no re-sort needed.
    n = len(df)
    intv_a, cum_a, sm_a = cumulative_phase(
        ticc["chA_ref_sec"].values, ticc["chA_ref_ps"].values, args.long_smooth)
    intv_b, cum_b, sm_b = cumulative_phase(
        ticc["chB_ref_sec"].values, ticc["chB_ref_ps"].values, args.long_smooth)

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
