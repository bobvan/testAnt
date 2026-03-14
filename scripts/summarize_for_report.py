#!/usr/bin/env python3
"""
summarize_for_report.py — Summarize raw testAnt CSVs into small files
suitable for report_card.py on a visualization host.

Runs on the data-collection host (e.g. PiPuss) where the multi-GB raw
CSVs live.  Produces ~100 KB summary files that can be scp'd back.

Usage:
    python summarize_for_report.py <prefix> [--rx BOT] [--outdir /tmp/summaries]

    prefix: path stem like data/ufo-top_patch2-bot_20260303T225657
            expects <prefix>.csv, <prefix>_rawx.csv, <prefix>_ticc.csv,
            <prefix>_timtp.csv

Outputs (in outdir):
    <tag>_snr_1min.csv    — per-minute per-receiver C/N0 + sat count
    <tag>_rawx_1min.csv   — per-minute per-SV-detrended CMC std
    <tag>_sky.csv         — per-minute per-SV sky positions (elev, azim, C/N0)
    <tag>_slips.csv       — quality-filtered cycle slip events
    <tag>_ticc.csv        — symlink or copy of TICC CSV (already small)
    <tag>_timtp.csv       — symlink or copy of TIM-TP CSV (already small)
"""

import argparse
import csv
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

C = 299792458.0
WAVELENGTH = {
    "GPS-L1CA":  C / 1575420000.0,
    "GAL-E1C":   C / 1575420000.0,
    "GAL-E1B":   C / 1575420000.0,
    "GPS-L5I":   C / 1176450000.0,
    "GPS-L5Q":   C / 1176450000.0,
    "GAL-E5aI":  C / 1176450000.0,
    "GAL-E5aQ":  C / 1176450000.0,
    "BDS-B1I":   C / 1561098000.0,
    "BDS-B2aI":  C / 1176450000.0,
    "GPS-L2CL":  C / 1227600000.0,
    "GPS-L2CM":  C / 1227600000.0,
    "GAL-E5bI":  C / 1207140000.0,
    "GAL-E5bQ":  C / 1207140000.0,
    "BDS-B2I":   C / 1207140000.0,
    "QZSS-L1CA": C / 1575420000.0,
    "SBAS-L1CA": C / 1575420000.0,
    "BDS-B1C":   C / 1575420000.0,
    "BDS-B1C-Q": C / 1575420000.0,
    "GLO-L1OF":  C / 1602000000.0,
    "GLO-L2OF":  C / 1246000000.0,
}


def summarize_snr(snr_csv, out_path):
    """Per-minute per-receiver: mean C/N0, sat count, used count, mean elev."""
    print(f"  SNR: {snr_csv}", file=sys.stderr)
    buckets = defaultdict(lambda: {"cno_sum": 0, "count": 0, "svs": set(),
                                   "used": 0, "elev_sum": 0})
    with open(snr_csv) as f:
        for row in csv.DictReader(f):
            ts = row["timestamp"][:16]
            rx = row["receiver"]
            b = buckets[(ts, rx)]
            b["cno_sum"] += float(row["cno_dBHz"])
            b["count"] += 1
            b["svs"].add(row["sv_id"])
            b["used"] += int(row["used"])
            b["elev_sum"] += float(row["elev_deg"])

    with open(out_path, "w") as f:
        f.write("minute,receiver,mean_cno,sat_count,used_count,mean_elev\n")
        for (ts, rx), b in sorted(buckets.items()):
            n = b["count"]
            f.write(f"{ts},{rx},{b['cno_sum']/n:.1f},{len(b['svs'])},"
                    f"{b['used']},{b['elev_sum']/n:.1f}\n")
    print(f"    → {out_path} ({len(buckets)} rows)", file=sys.stderr)


def summarize_sky(snr_csv, out_path):
    """Per-minute per-receiver per-SV: mean elev, azim, C/N0."""
    print(f"  Sky: {snr_csv}", file=sys.stderr)
    buckets = defaultdict(lambda: [0.0, 0.0, 0.0, 0])

    with open(snr_csv) as f:
        for row in csv.DictReader(f):
            ts = row["timestamp"][:16]
            rx = row["receiver"]
            sv = row["sv_id"]
            b = buckets[(ts, rx, sv)]
            b[0] += float(row["elev_deg"])
            b[1] += float(row["azim_deg"])
            b[2] += float(row["cno_dBHz"])
            b[3] += 1

    with open(out_path, "w") as f:
        f.write("minute,receiver,sv_id,elev_deg,azim_deg,cno_dBHz\n")
        for (ts, rx, sv), (es, azs, cs, n) in sorted(buckets.items()):
            f.write(f"{ts},{rx},{sv},{es/n:.1f},{azs/n:.1f},{cs/n:.1f}\n")
    print(f"    → {out_path} ({len(buckets)} rows)", file=sys.stderr)


def summarize_rawx(rawx_csv, out_path):
    """Per-minute per-receiver: median per-SV detrended CMC std (two-pass)."""
    print(f"  RAWX CMC (pass 1 — arc baselines): {rawx_csv}", file=sys.stderr)

    last_lock = {}
    arc_base = {}
    sv_buckets = defaultdict(list)

    with open(rawx_csv) as f:
        for row in csv.DictReader(f):
            sig = row["signal_id"]
            wl = WAVELENGTH.get(sig)
            if wl is None:
                continue
            if row.get("pr_valid", "1") != "1" or row.get("cp_valid", "1") != "1":
                continue
            try:
                pr = float(row["pseudorange_m"])
                cp = float(row["carrier_phase_cy"])
                lock = float(row["lock_duration_ms"])
            except (ValueError, KeyError):
                continue

            cmc = pr - wl * cp
            rx, sv = row["receiver"], row["sv_id"]
            key = (rx, sv, sig)

            prev = last_lock.get(key, -1)
            if lock < prev or key not in arc_base:
                arc_base[key] = cmc
            last_lock[key] = lock

            ts = row["timestamp"][:16]
            sv_buckets[(ts, rx, sv, sig)].append(cmc - arc_base[key])

    print(f"  RAWX CMC (pass 2 — per-minute stats)", file=sys.stderr)
    minute_rx_stds = defaultdict(list)
    for (ts, rx, sv, sig), vals in sv_buckets.items():
        if len(vals) < 3:
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        minute_rx_stds[(ts, rx)].append(math.sqrt(var))

    with open(out_path, "w") as f:
        f.write("minute,receiver,cmc_std_m,n_svs\n")
        for (ts, rx), stds in sorted(minute_rx_stds.items()):
            if len(stds) < 3:
                continue
            stds.sort()
            median_std = stds[len(stds) // 2]
            f.write(f"{ts},{rx},{median_std:.4f},{len(stds)}\n")
    print(f"    → {out_path} ({len(minute_rx_stds)} rows)", file=sys.stderr)


def summarize_slips(rawx_csv, out_path, min_lock_ms=10000, min_cno=25):
    """Quality-filtered cycle slip events."""
    print(f"  Slips: {rawx_csv} (lock>{min_lock_ms}ms, C/N0>{min_cno})",
          file=sys.stderr)
    last_lock = {}
    n_slips = 0

    with open(out_path, "w") as fout:
        fout.write("timestamp,receiver,sv_id,signal_id,cno_dBHz,prev_lock_ms\n")
        with open(rawx_csv) as f:
            for row in csv.DictReader(f):
                rx, sv, sig = row["receiver"], row["sv_id"], row["signal_id"]
                key = (rx, sv, sig)
                try:
                    lock = float(row["lock_duration_ms"])
                    cno = float(row["cno_dBHz"])
                except (ValueError, KeyError):
                    continue
                prev = last_lock.get(key)
                if prev is not None and lock < prev:
                    if prev >= min_lock_ms and cno >= min_cno:
                        ts = row["timestamp"][:19]
                        fout.write(f"{ts},{rx},{sv},{sig},{cno:.0f},{prev:.0f}\n")
                        n_slips += 1
                last_lock[key] = lock

    print(f"    → {out_path} ({n_slips} slips)", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(
        description="Summarize raw testAnt CSVs for report_card.py")
    ap.add_argument("prefix", help="Path stem (e.g. data/ufo-top_patch2-bot_20260303T225657)")
    ap.add_argument("--outdir", default="/tmp/ta_summaries",
                    help="Output directory (default /tmp/ta_summaries)")
    ap.add_argument("--tag", default=None,
                    help="Output file prefix (default: basename of prefix)")
    args = ap.parse_args()

    prefix = args.prefix
    tag = args.tag or Path(prefix).name
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    snr_csv = f"{prefix}.csv"
    rawx_csv = f"{prefix}_rawx.csv"
    ticc_csv = f"{prefix}_ticc.csv"
    timtp_csv = f"{prefix}_timtp.csv"

    # Check files exist
    for f in [snr_csv, rawx_csv, ticc_csv, timtp_csv]:
        if not os.path.exists(f):
            print(f"ERROR: {f} not found", file=sys.stderr)
            sys.exit(1)

    # SNR summaries
    summarize_snr(snr_csv, outdir / f"{tag}_snr_1min.csv")
    summarize_sky(snr_csv, outdir / f"{tag}_sky.csv")

    # RAWX summaries
    summarize_rawx(rawx_csv, outdir / f"{tag}_rawx_1min.csv")
    summarize_slips(rawx_csv, outdir / f"{tag}_slips.csv")

    # Copy small files
    for suffix in ["_ticc.csv", "_timtp.csv"]:
        src = f"{prefix}{suffix}"
        dst = outdir / f"{tag}{suffix}"
        shutil.copy2(src, dst)
        print(f"  Copied {src} → {dst}", file=sys.stderr)

    print(f"\nDone. Summaries in {outdir}/", file=sys.stderr)
    print(f"Transfer with: scp {outdir}/{tag}_* <vizhost>:data/", file=sys.stderr)


if __name__ == "__main__":
    main()
