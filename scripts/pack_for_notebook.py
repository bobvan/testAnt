#!/usr/bin/env python3
"""
pack_for_notebook.py — Decimate and bundle summary data for notebook transfer.

Runs on PiPuss after summarize_for_report.py.  Takes the summary directory
and produces a compact .tar.gz bundle suitable for scp to the analysis Mac.

Decimation steps:
  - snr_1min.csv, rawx_1min.csv, slips.csv  → copy as-is (already small)
  - sky.csv         → decimate to 5-min medians per SV (75K → ~15K rows)
  - ticc.csv        → pre-compute windowed ADEV + keep 1-in-10 raw pairs
  - timtp.csv       → pair with ticc for qErr correction, then discard raw

Output: <tag>_notebook.tar.gz containing:
  - *_snr_1min.csv, *_rawx_1min.csv, *_slips.csv  (unchanged)
  - *_sky_5min.csv      (decimated sky positions)
  - *_adev_windowed.csv (pre-computed 10-min windowed ADEV at multiple taus)
  - *_timing_1hz.csv    (qErr-corrected TICC diffs at 1 Hz, for custom ADEV)

Usage:
    python pack_for_notebook.py data/patch1-bot_patch2-top_20260309T193726 \
        [--outdir /tmp/bundles]
"""

import argparse
import csv
import os
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import allantools
    HAS_ALLAN = True
except ImportError:
    HAS_ALLAN = False


def decimate_sky(sky_path, out_path):
    """Decimate sky.csv to 5-minute medians per (receiver, sv_id)."""
    sky = pd.read_csv(sky_path, parse_dates=["minute"])
    sky["minute"] = pd.to_datetime(sky["minute"], utc=True)
    sky["bin5"] = sky["minute"].dt.floor("5min")

    agg = sky.groupby(["bin5", "receiver", "sv_id"]).agg(
        elev_deg=("elev_deg", "median"),
        azim_deg=("azim_deg", "median"),
        cno_dBHz=("cno_dBHz", "median"),
    ).reset_index()
    agg = agg.rename(columns={"bin5": "minute"})
    agg.to_csv(out_path, index=False)
    return len(sky), len(agg)


def compute_timing(ticc_path, timtp_path, out_adev_path, out_timing_path):
    """Compute qErr-corrected timing diffs and windowed ADEV."""
    ticc = pd.read_csv(ticc_path)

    has_host_ts = "host_timestamp" in ticc.columns
    if has_host_ts:
        ticc["host_timestamp"] = pd.to_datetime(ticc["host_timestamp"], utc=True)
        ticc["utc_s"] = ticc["host_timestamp"].dt.floor("s")
        cha = ticc[ticc["channel"] == "chA"].groupby("utc_s").first().reset_index()
        chb = ticc[ticc["channel"] == "chB"].groupby("utc_s").first().reset_index()
        pairs = cha.merge(chb, on="utc_s", suffixes=("_a", "_b"))
        pairs["diff_ps"] = (
            pairs["ref_sec_a"].astype(np.int64) * 1_000_000_000_000
            + pairs["ref_ps_a"].astype(np.int64)
            - pairs["ref_sec_b"].astype(np.int64) * 1_000_000_000_000
            - pairs["ref_ps_b"].astype(np.int64)
        )
    else:
        ticc["ts_ps"] = (ticc["timestamp_s"] * 1e12).round().astype(np.int64)
        cha = ticc[ticc["channel"] == "chA"].reset_index(drop=True)
        chb = ticc[ticc["channel"] == "chB"].reset_index(drop=True)
        n = min(len(cha), len(chb))
        cha, chb = cha.iloc[:n], chb.iloc[:n]
        pairs = pd.DataFrame({"diff_ps": cha["ts_ps"].values - chb["ts_ps"].values})
        timtp_tmp = pd.read_csv(timtp_path, parse_dates=["timestamp"])
        timtp_top = timtp_tmp[timtp_tmp["receiver"] == "TOP"].sort_values("timestamp")
        if len(timtp_top) >= n:
            pairs["utc_s"] = timtp_top["timestamp"].iloc[:n].dt.floor("s").values

    pairs["diff_ns"] = pairs["diff_ps"] / 1000.0

    # qErr correction
    timtp = pd.read_csv(timtp_path, parse_dates=["timestamp"])
    timtp["utc_s"] = timtp["timestamp"].dt.floor("s")

    if "utc_s" in pairs.columns:
        pairs["utc_s"] = pd.to_datetime(pairs["utc_s"], utc=True)
        for label in ["TOP", "BOT"]:
            q = timtp[timtp["receiver"] == label][["utc_s", "qerr_ps"]].copy()
            q["utc_s"] = q["utc_s"] + pd.Timedelta(seconds=1)
            q = q.rename(columns={"qerr_ps": f"qerr_{label.lower()}"})
            pairs = pairs.merge(q, on="utc_s", how="left")
        pairs = pairs.dropna(subset=["qerr_top", "qerr_bot"])
        corr_p = pairs["diff_ns"] + (pairs["qerr_top"] - pairs["qerr_bot"]) / 1000
        corr_m = pairs["diff_ns"] - (pairs["qerr_top"] - pairs["qerr_bot"]) / 1000
        pairs["corr_ns"] = corr_p if corr_p.std() < corr_m.std() else corr_m
    else:
        pairs["corr_ns"] = pairs["diff_ns"]

    # Outlier rejection
    med = pairs["corr_ns"].median()
    mad = (pairs["corr_ns"] - med).abs().median() * 1.4826
    pairs = pairs[(pairs["corr_ns"] - med).abs() < max(mad * 5, 50)].reset_index(drop=True)

    # Write 1 Hz timing CSV (the key transfer file — replaces raw ticc+timtp)
    if "utc_s" in pairs.columns:
        timing_out = pairs[["utc_s", "diff_ns", "corr_ns"]].copy()
    else:
        timing_out = pairs[["diff_ns", "corr_ns"]].copy()
    timing_out.to_csv(out_timing_path, index=False)

    # Windowed ADEV at multiple taus
    if not HAS_ALLAN:
        print("  WARNING: allantools not installed, skipping ADEV", file=sys.stderr)
        pd.DataFrame(columns=["window_h", "tau", "adev_ns"]).to_csv(
            out_adev_path, index=False)
        return len(ticc), len(timing_out), 0

    if "utc_s" in pairs.columns:
        t0 = pairs["utc_s"].iloc[0]
        pairs["hours"] = (pairs["utc_s"] - t0).dt.total_seconds() / 3600
    else:
        pairs["hours"] = np.arange(len(pairs)) / 3600

    freq_ns = pairs["corr_ns"].values - pairs["corr_ns"].mean()
    taus = [1, 2, 5, 10, 30, 60, 300, 900]

    rows = []
    # Overall ADEV
    try:
        _, ad, _, _ = allantools.adev(freq_ns, rate=1.0, data_type="freq", taus=taus)
        for i, t in enumerate(taus[:len(ad)]):
            rows.append({"window_h": -1, "tau": t, "adev_ns": ad[i]})
    except Exception:
        pass

    # 10-minute windowed ADEV at tau=1s
    pairs["window"] = (pairs["hours"] * 6).astype(int)
    for wid, g in pairs.groupby("window"):
        if len(g) < 60:
            continue
        y = g["corr_ns"].values - g["corr_ns"].mean()
        try:
            _, ad, _, _ = allantools.adev(y, rate=1.0, data_type="freq", taus=[1])
            if len(ad):
                rows.append({
                    "window_h": g["hours"].mean(),
                    "tau": 1,
                    "adev_ns": ad[0],
                })
        except Exception:
            pass

    pd.DataFrame(rows).to_csv(out_adev_path, index=False)
    return len(ticc), len(timing_out), len(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Decimate and bundle summary data for notebook transfer")
    ap.add_argument("prefix",
                    help="Data prefix (e.g. data/patch1-bot_patch2-top_20260309T193726)")
    ap.add_argument("--outdir", default=None,
                    help="Output directory (default: same as prefix parent)")
    args = ap.parse_args()

    prefix = args.prefix.rstrip("/")
    tag = os.path.basename(prefix)
    src_dir = os.path.dirname(prefix) or "."
    out_dir = args.outdir or src_dir

    # Locate summary files
    required = {
        "snr":   f"{prefix}_snr_1min.csv",
        "rawx":  f"{prefix}_rawx_1min.csv",
        "sky":   f"{prefix}_sky.csv",
        "ticc":  f"{prefix}_ticc.csv",
        "timtp": f"{prefix}_timtp.csv",
    }
    optional = {
        "slips": f"{prefix}_slips.csv",
    }

    for name, path in required.items():
        if not os.path.exists(path):
            print(f"ERROR: missing {name}: {path}", file=sys.stderr)
            sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    print(f"Packing {tag} → {out_dir}/", file=sys.stderr)

    # Files to include in bundle
    bundle_files = []

    # Copy small files as-is
    for name in ["snr", "rawx"]:
        bundle_files.append(required[name])
    if os.path.exists(optional.get("slips", "")):
        bundle_files.append(optional["slips"])

    # Decimate sky
    sky_out = os.path.join(out_dir, f"{tag}_sky_5min.csv")
    n_raw, n_dec = decimate_sky(required["sky"], sky_out)
    print(f"  sky: {n_raw} → {n_dec} rows (5-min medians)", file=sys.stderr)
    bundle_files.append(sky_out)

    # Compute timing
    adev_out = os.path.join(out_dir, f"{tag}_adev_windowed.csv")
    timing_out = os.path.join(out_dir, f"{tag}_timing_1hz.csv")
    n_ticc, n_timing, n_adev = compute_timing(
        required["ticc"], required["timtp"], adev_out, timing_out)
    print(f"  ticc: {n_ticc} → {n_timing} corrected pairs + {n_adev} ADEV points",
          file=sys.stderr)
    bundle_files.append(adev_out)
    bundle_files.append(timing_out)

    # Create tar.gz bundle
    tar_path = os.path.join(out_dir, f"{tag}_notebook.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        for f in bundle_files:
            tar.add(f, arcname=os.path.basename(f))
    size_mb = os.path.getsize(tar_path) / 1e6
    print(f"\n  Bundle: {tar_path} ({size_mb:.1f} MB)", file=sys.stderr)
    print(f"\n  scp pipuss:{os.path.abspath(tar_path)} .", file=sys.stderr)


if __name__ == "__main__":
    main()
