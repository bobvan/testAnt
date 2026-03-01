#!/usr/bin/env python3
"""
analyze_snr.py — Summarize a C/N0 log CSV; produce text report + plots.

Usage:
    python scripts/analyze_snr.py --csv data/snr_overnight.csv --out data/overnight
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def epoch_stats(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["timestamp", "receiver"])
        .agg(mean_cno=("cno_dBHz", "mean"),
             std_cno=("cno_dBHz", "std"),
             sat_count=("sv_id", "count"),
             used_count=("used", "sum"))
        .reset_index()
    )


def write_report(df: pd.DataFrame, epoch: pd.DataFrame, out_stem: Path) -> None:
    lines = []
    a = lines.append

    dur_h = (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 3600
    a("=" * 60)
    a("  testAnt C/N0 report")
    a("=" * 60)
    a(f"  Start    : {df['timestamp'].min()}")
    a(f"  End      : {df['timestamp'].max()}")
    a(f"  Duration : {dur_h:.2f} h")
    a(f"  Epochs   : {epoch['timestamp'].nunique()}")
    a("")

    a("── Overall C/N0 per receiver ───────────────────────────────")
    for rx, g in epoch.groupby("receiver"):
        a(f"  {rx:6s}  mean={g['mean_cno'].mean():.2f} dBHz  "
          f"σ={g['mean_cno'].std():.2f}  "
          f"avg_sats={g['sat_count'].mean():.1f}")
    a("")

    ref = epoch[epoch["receiver"] == "REF"].set_index("timestamp")["mean_cno"]
    dut = epoch[epoch["receiver"] == "DUT"].set_index("timestamp")["mean_cno"]
    delta = (ref - dut).dropna()
    if not delta.empty:
        a("── REF − DUT delta ─────────────────────────────────────────")
        a(f"  mean={delta.mean():+.3f}  σ={delta.std():.3f}  "
          f"min={delta.min():+.3f}  max={delta.max():+.3f} dBHz")
        a(f"  +ve → REF stronger;  −ve → DUT stronger")
        a("")

    a("── Mean C/N0 by constellation ──────────────────────────────")
    # σ = std of per-epoch means (temporal variability), not std of raw obs
    epoch_cst = (df.groupby(["timestamp", "receiver", "gnss_id"])["cno_dBHz"]
                   .mean().reset_index())
    cst_mean = (epoch_cst.groupby(["receiver", "gnss_id"])["cno_dBHz"]
                         .agg(mean_cno="mean", std_cno="std", n="count")
                         .reset_index()
                         .sort_values(["gnss_id", "receiver"]))
    for _, row in cst_mean.iterrows():
        a(f"  {row['receiver']:6s}  {row['gnss_id']:12s}  "
          f"mean={row['mean_cno']:.2f} ±{row['std_cno']:.2f} dBHz  n_epochs={int(row['n'])}")
    a("")

    a("── Hourly mean C/N0 ────────────────────────────────────────")
    epoch2 = epoch.copy()
    epoch2["hour"] = epoch2["timestamp"].dt.floor("h")
    hrly = (epoch2.groupby(["hour", "receiver"])
                  .agg(mean_cno=("mean_cno", "mean"), sat_count=("sat_count", "mean"))
                  .reset_index()
                  .sort_values(["hour", "receiver"]))
    for _, row in hrly.iterrows():
        a(f"  {row['hour'].strftime('%H:%M')}  {row['receiver']:6s}  "
          f"C/N0={row['mean_cno']:.2f} dBHz  sats={row['sat_count']:.1f}")
    a("")
    a("=" * 60)

    path = out_stem.parent / (out_stem.name + "_report.txt")
    path.write_text("\n".join(lines) + "\n")
    print(f"Report  → {path}")


def plot_cno(epoch: pd.DataFrame, out_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    for rx, g in epoch.groupby("receiver"):
        g = g.sort_values("timestamp")
        ax.plot(g["timestamp"], g["mean_cno"], label=rx, linewidth=0.6, alpha=0.85)
        ax.fill_between(g["timestamp"],
                        g["mean_cno"] - g["std_cno"],
                        g["mean_cno"] + g["std_cno"],
                        alpha=0.15)
    ax.set_title("Mean C/N0 over time (shaded band = ±1σ across tracked satellites)")
    ax.set_ylabel("C/N0 (dBHz)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    path = out_stem.parent / (out_stem.name + "_cno.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"Plot    → {path}")


def plot_delta(epoch: pd.DataFrame, out_stem: Path) -> None:
    # REF and DUT are stamped independently; floor to 1s so the join works.
    e = epoch.copy()
    e["timestamp"] = e["timestamp"].dt.floor("s")
    e = e.groupby(["timestamp", "receiver"], as_index=False)["mean_cno"].mean()
    ref = e[e["receiver"] == "REF"].set_index("timestamp")["mean_cno"]
    dut = e[e["receiver"] == "DUT"].set_index("timestamp")["mean_cno"]
    delta = (ref - dut).dropna().sort_index()
    if delta.empty:
        return
    smooth = delta.rolling("5min").median()
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(delta.index, delta.values, 0,
                    where=delta.values >= 0, alpha=0.25, color="steelblue", label="REF > DUT")
    ax.fill_between(delta.index, delta.values, 0,
                    where=delta.values < 0,  alpha=0.25, color="tomato",    label="DUT > REF")
    ax.plot(smooth.index, smooth.values, color="navy", linewidth=1.0, label="5-min median")
    ax.set_title("REF − DUT mean C/N0")
    ax.set_ylabel("ΔC/N0 (dBHz)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    path = out_stem.parent / (out_stem.name + "_delta.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"Plot    → {path}")


def plot_satcount(epoch: pd.DataFrame, out_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 3))
    for rx, g in epoch.groupby("receiver"):
        g = g.sort_values("timestamp")
        ax.plot(g["timestamp"], g["sat_count"], label=rx, linewidth=0.6, alpha=0.85)
    ax.set_title("Tracked satellite count")
    ax.set_ylabel("Satellites")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    path = out_stem.parent / (out_stem.name + "_satcount.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"Plot    → {path}")


def plot_by_constellation(df: pd.DataFrame, out_stem: Path) -> None:
    # Compute per-epoch per-constellation means, then derive per-constellation
    # stats for REF and DUT. Plot REF−DUT delta with quadrature error bars:
    # σ_diff = sqrt(σ_REF² + σ_DUT²). Error bar crossing zero = not significant.
    epoch_cst = (df.groupby(["timestamp", "receiver", "gnss_id"])["cno_dBHz"]
                   .mean()
                   .reset_index())
    stats = (epoch_cst.groupby(["receiver", "gnss_id"])["cno_dBHz"]
                      .agg(["mean", "std"])
                      .reset_index())

    consts = sorted(df["gnss_id"].unique())
    deltas, errs = [], []
    for c in consts:
        r = stats[(stats["receiver"] == "REF") & (stats["gnss_id"] == c)]
        d = stats[(stats["receiver"] == "DUT") & (stats["gnss_id"] == c)]
        if r.empty or d.empty:
            deltas.append(0); errs.append(0)
            continue
        deltas.append(float(r["mean"].iloc[0] - d["mean"].iloc[0]))
        errs.append(float((r["std"].iloc[0]**2 + d["std"].iloc[0]**2) ** 0.5))

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["steelblue" if d >= 0 else "tomato" for d in deltas]
    ax.bar(consts, deltas, color=colors, alpha=0.8,
           yerr=errs, capsize=6, error_kw={"linewidth": 1.5, "color": "black"})
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("REF − DUT C/N0 (dBHz)")
    ax.set_title("REF − DUT mean C/N0 by constellation\n"
                 "error bars = ±1σ quadrature (√(σ_REF² + σ_DUT²)); "
                 "bar crossing zero = not significant")
    ax.grid(True, alpha=0.3, axis="y"); fig.tight_layout()
    path = out_stem.parent / (out_stem.name + "_by_constellation.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"Plot    → {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True, help="Output filename stem")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_stem = Path(args.out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {csv_path} …")
    df = load(csv_path)
    print(f"  {len(df):,} rows  {df['timestamp'].nunique()} epochs")

    epoch = epoch_stats(df)
    write_report(df, epoch, out_stem)
    plot_cno(epoch, out_stem)
    plot_delta(epoch, out_stem)
    plot_satcount(epoch, out_stem)
    plot_by_constellation(df, out_stem)
    print("Done.")


if __name__ == "__main__":
    main()
