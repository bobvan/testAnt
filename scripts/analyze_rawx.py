#!/usr/bin/env python3
"""
analyze_rawx.py — Code-minus-carrier (CMC) analysis from a RAWX CSV.

Usage:
    python scripts/analyze_rawx.py --csv data/foo_rawx.csv --out data/foo

Outputs (all suffixed onto <out>):
    _rawx_report.txt    — per-signal CMC std dev and lock time stats
    _cmc_by_signal.png  — CMC time series per signal (receivers overlaid)
    _cmc_diff.png       — CMC_A − CMC_B per signal (reveals receiver noise/bias)
    _locktime.png       — carrier phase lock time CDF per signal
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ── Signal wavelengths (c/f, metres) ────────────────────────────────── #
# c = 299 792 458 m/s
_C = 299_792_458.0

_WAVELENGTH: dict[str, float] = {
    "GPS-L1CA":    _C / 1_575_420_000.0,   # 0.19029 m
    "GAL-E1C":     _C / 1_575_420_000.0,
    "GAL-E1B":     _C / 1_575_420_000.0,
    "QZSS-L1CA":   _C / 1_575_420_000.0,
    "SBAS-L1CA":   _C / 1_575_420_000.0,
    "GPS-L5I":     _C / 1_176_450_000.0,   # 0.25480 m
    "GPS-L5Q":     _C / 1_176_450_000.0,
    "GAL-E5aI":    _C / 1_176_450_000.0,
    "GAL-E5aQ":    _C / 1_176_450_000.0,
    "BDS-B2aI":    _C / 1_176_450_000.0,
    "BDS-B2aQ":    _C / 1_176_450_000.0,
    "BDS-B1I":     _C / 1_561_098_000.0,   # 0.19203 m
    "BDS-B1I-D2":  _C / 1_561_098_000.0,
    "BDS-B1C":     _C / 1_575_420_000.0,   # 0.19029 m (same as GPS L1)
    "BDS-B1C-Q":   _C / 1_575_420_000.0,
    "BDS-B2I":     _C / 1_207_140_000.0,   # 0.24834 m
    "BDS-B2I-D2":  _C / 1_207_140_000.0,
    "GAL-E5bI":    _C / 1_207_140_000.0,
    "GAL-E5bQ":    _C / 1_207_140_000.0,
    "GPS-L2CL":    _C / 1_227_600_000.0,   # 0.24421 m
    "GPS-L2CM":    _C / 1_227_600_000.0,
    "GLO-L1OF":    _C / 1_602_000_000.0,   # 0.18740 m (centre, ignores slot offset)
    "GLO-L2OF":    _C / 1_246_000_000.0,   # 0.24051 m (centre)
}


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def add_cmc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'cmc_m' column: CMC = pseudorange_m − wavelength × carrier_phase_cy.

    Only rows with a known wavelength, cp_valid=1, and half_cyc=1
    (half-cycle resolved) get a finite CMC value; others are NaN.
    """
    df = df.copy()
    wl = df["signal_id"].map(_WAVELENGTH)       # NaN for unknown signals
    cp_ok = (df["cp_valid"] == 1) & (df["half_cyc"] == 1)
    df["cmc_m"] = np.where(
        cp_ok & wl.notna(),
        df["pseudorange_m"] - wl * df["carrier_phase_cy"],
        np.nan,
    )
    return df


def write_report(df: pd.DataFrame, out_stem: Path) -> None:
    lines = []
    a = lines.append

    dur_h = (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 3600
    a("=" * 62)
    a("  testAnt RAWX / CMC report")
    a("=" * 62)
    a(f"  Start    : {df['timestamp'].min()}")
    a(f"  End      : {df['timestamp'].max()}")
    a(f"  Duration : {dur_h:.2f} h")
    a(f"  Total rows (pr_valid): {len(df):,}")
    a("")

    cmc_ok = df.dropna(subset=["cmc_m"])

    a("── CMC std dev by signal & receiver (noise floor) ──────────────")
    a(f"  {'Signal':16s}  {'Receiver':8s}  {'N':>6s}  "
      f"{'mean_m':>9s}  {'std_m':>8s}")
    stats = (cmc_ok.groupby(["signal_id", "receiver"])["cmc_m"]
               .agg(["count", "mean", "std"]).reset_index()
               .sort_values(["signal_id", "receiver"]))
    for _, row in stats.iterrows():
        a(f"  {row['signal_id']:16s}  {row['receiver']:8s}  "
          f"{int(row['count']):>6d}  "
          f"{row['mean']:>+9.3f}  {row['std']:>8.3f} m")
    a("")

    a("── Lock time stats by signal & receiver (ms) ───────────────────")
    a(f"  {'Signal':16s}  {'Receiver':8s}  {'median_ms':>10s}  {'p95_ms':>8s}")
    lt_stats = (df.groupby(["signal_id", "receiver"])["locktime_ms"]
                  .agg(median="median", p95=lambda x: x.quantile(0.95))
                  .reset_index()
                  .sort_values(["signal_id", "receiver"]))
    for _, row in lt_stats.iterrows():
        a(f"  {row['signal_id']:16s}  {row['receiver']:8s}  "
          f"{row['median']:>10.0f}  {row['p95']:>8.0f}")
    a("")
    a("=" * 62)

    path = out_stem.parent / (out_stem.name + "_rawx_report.txt")
    path.write_text("\n".join(lines) + "\n")
    print(f"Report  → {path}")


def plot_cmc_by_signal(df: pd.DataFrame, out_stem: Path) -> None:
    """CMC time series per signal, receivers overlaid."""
    cmc_ok = df.dropna(subset=["cmc_m"])
    signals = sorted(cmc_ok["signal_id"].unique())
    if not signals:
        return

    n = len(signals)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True, squeeze=False)

    receivers = sorted(cmc_ok["receiver"].unique())
    colors = ["steelblue", "tomato", "seagreen", "darkorange"]

    for ax, sig in zip(axes[:, 0], signals):
        sub = cmc_ok[cmc_ok["signal_id"] == sig]
        # Per-epoch mean CMC across all SVs, per receiver
        epoch_mean = (sub.groupby(["timestamp", "receiver"])["cmc_m"]
                        .median().reset_index())
        for color, rx in zip(colors, receivers):
            g = epoch_mean[epoch_mean["receiver"] == rx].sort_values("timestamp")
            if g.empty:
                continue
            ax.plot(g["timestamp"], g["cmc_m"], label=rx,
                    linewidth=0.6, alpha=0.85, color=color)
        ax.set_ylabel("CMC (m)")
        ax.set_title(sig)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    fig.suptitle("Code-minus-carrier (CMC) by signal — epoch median across all SVs",
                 fontsize=11, y=1.002)
    fig.tight_layout()
    path = out_stem.parent / (out_stem.name + "_cmc_by_signal.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot    → {path}")


def plot_cmc_diff(df: pd.DataFrame, out_stem: Path) -> None:
    """CMC_A − CMC_B per signal per SV — reveals receiver noise and bias."""
    cmc_ok = df.dropna(subset=["cmc_m"])
    receivers = sorted(cmc_ok["receiver"].unique())
    if len(receivers) < 2:
        return
    rx_a, rx_b = receivers[0], receivers[1]

    signals = sorted(cmc_ok["signal_id"].unique())
    if not signals:
        return

    n = len(signals)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True, squeeze=False)

    for ax, sig in zip(axes[:, 0], signals):
        sub = cmc_ok[cmc_ok["signal_id"] == sig]
        # Per-epoch median CMC per receiver
        ep = (sub.groupby(["timestamp", "receiver"])["cmc_m"]
                 .median().reset_index())
        ep["ts_s"] = ep["timestamp"].dt.floor("s")
        ep2 = ep.groupby(["ts_s", "receiver"], as_index=False)["cmc_m"].median()

        a = ep2[ep2["receiver"] == rx_a].set_index("ts_s")["cmc_m"]
        b = ep2[ep2["receiver"] == rx_b].set_index("ts_s")["cmc_m"]
        diff = (a - b).dropna().sort_index()
        if diff.empty:
            ax.set_title(f"{sig} — no overlapping epochs")
            continue

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.fill_between(diff.index, diff.values, 0,
                        where=diff.values >= 0, alpha=0.25, color="steelblue")
        ax.fill_between(diff.index, diff.values, 0,
                        where=diff.values < 0,  alpha=0.25, color="tomato")
        ax.plot(diff.index, diff.rolling(60, min_periods=10).median(),
                color="navy", linewidth=1.0, label="60-s median")
        ax.set_ylabel("ΔCMC (m)")
        ax.set_title(f"{sig}  —  {rx_a} − {rx_b}  CMC diff  "
                     f"(std={diff.std():.3f} m)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    fig.suptitle(f"CMC difference ({rx_a} − {rx_b}) by signal\n"
                 "common-mode iono/clock cancelled; residual = receiver noise + multipath",
                 fontsize=11, y=1.002)
    fig.tight_layout()
    path = out_stem.parent / (out_stem.name + "_cmc_diff.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot    → {path}")


def plot_locktime(df: pd.DataFrame, out_stem: Path) -> None:
    """Carrier phase lock time CDF per signal."""
    signals = sorted(df["signal_id"].dropna().unique())
    if not signals:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10.colors

    for color, sig in zip(colors, signals):
        sub = df[df["signal_id"] == sig]["locktime_ms"].dropna()
        if sub.empty:
            continue
        x = np.sort(sub.values)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, label=sig, linewidth=1.2, color=color)

    ax.set_xlabel("Lock time (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Carrier phase lock time CDF by signal")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    path = out_stem.parent / (out_stem.name + "_locktime.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Plot    → {path}")


def main():
    ap = argparse.ArgumentParser(
        description="CMC multipath analysis from a RAWX CSV"
    )
    ap.add_argument("--csv",  required=True, help="Input _rawx.csv file")
    ap.add_argument("--out",  required=True, help="Output filename stem")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_stem = Path(args.out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {csv_path} …")
    df = load(csv_path)
    print(f"  {len(df):,} rows  "
          f"{df['timestamp'].nunique()} epochs  "
          f"{df['signal_id'].nunique()} signals")

    df = add_cmc(df)
    n_cmc = df["cmc_m"].notna().sum()
    print(f"  {n_cmc:,} rows with valid CMC (cp_valid=1, half_cyc=1, known wavelength)")

    write_report(df, out_stem)
    plot_cmc_by_signal(df, out_stem)
    plot_cmc_diff(df, out_stem)
    plot_locktime(df, out_stem)
    print("Done.")


if __name__ == "__main__":
    main()
