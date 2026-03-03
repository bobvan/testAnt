#!/usr/bin/env python3
"""
analyze_snr.py — Summarize a C/N0 log CSV; produce text report + plots.

Usage:
    python scripts/analyze_snr.py --csv data/snr_overnight.csv --out data/overnight
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
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


def make_rx_label(df: pd.DataFrame) -> dict[str, str]:
    """Map each receiver → 'Antenna @ site (RECEIVER)' for labels and reports."""
    result = {}
    for rx, g in df.groupby("receiver"):
        ant_str = (g["antenna_mount"].dropna().mode().iloc[0]
                   if "antenna_mount" in g.columns and g["antenna_mount"].notna().any()
                   else rx)
        site_str = ""
        if "mount_site" in g.columns:
            site = g["mount_site"].dropna().mode()
            if not site.empty and str(site.iloc[0]):
                site_str = f" @ {site.iloc[0]}"
        result[rx] = f"{ant_str}{site_str} ({rx})"
    return result


def write_report(df: pd.DataFrame, epoch: pd.DataFrame, out_stem: Path) -> None:
    lines = []
    a = lines.append
    labels = make_rx_label(df)

    dur_h = (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 3600
    a("=" * 60)
    a("  testAnt C/N0 report")
    a("=" * 60)
    a(f"  Start    : {df['timestamp'].min()}")
    a(f"  End      : {df['timestamp'].max()}")
    a(f"  Duration : {dur_h:.2f} h")
    a(f"  Epochs   : {epoch['timestamp'].nunique()}")
    a("")

    a("── Overall C/N0 per antenna ────────────────────────────────")
    for rx, g in epoch.groupby("receiver"):
        lbl = labels.get(rx, rx)
        a(f"  {lbl:30s}  mean={g['mean_cno'].mean():.2f} dBHz  "
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

    sig_col = "signal_id" if "signal_id" in df.columns else "gnss_id"
    a(f"── Mean C/N0 by signal ({'signal_id' if sig_col == 'signal_id' else 'gnss_id — old CSV'}) ──")
    # σ = std of per-epoch means (temporal variability), not std of raw obs
    epoch_sig = (df.groupby(["timestamp", "receiver", sig_col])["cno_dBHz"]
                   .mean().reset_index())
    sig_mean = (epoch_sig.groupby(["receiver", sig_col])["cno_dBHz"]
                         .agg(mean_cno="mean", std_cno="std", n="count")
                         .reset_index()
                         .sort_values([sig_col, "receiver"]))
    for _, row in sig_mean.iterrows():
        lbl = labels.get(row['receiver'], row['receiver'])
        a(f"  {lbl:30s}  {row[sig_col]:14s}  "
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
        lbl = labels.get(row['receiver'], row['receiver'])
        a(f"  {row['hour'].strftime('%H:%M')}  {lbl:30s}  "
          f"C/N0={row['mean_cno']:.2f} dBHz  sats={row['sat_count']:.1f}")
    a("")
    a("=" * 60)

    path = out_stem.parent / (out_stem.name + "_report.txt")
    path.write_text("\n".join(lines) + "\n")
    print(f"Report  → {path}")


def _add_cno_shading(ax) -> None:
    """Add light red/yellow/green background shading for C/N0 quality zones."""
    ax.axhspan(0,  30, color="red",    alpha=0.06, zorder=0)
    ax.axhspan(30, 40, color="gold",   alpha=0.07, zorder=0)
    ax.axhspan(40, 90, color="green",  alpha=0.05, zorder=0)


def plot_cno(epoch: pd.DataFrame, out_stem: Path, labels: dict,
             filter_note: str = "") -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    _add_cno_shading(ax)
    for rx, g in epoch.groupby("receiver"):
        g = g.sort_values("timestamp")
        ax.plot(g["timestamp"], g["mean_cno"], label=labels.get(rx, rx),
                linewidth=0.6, alpha=0.85)
        ax.fill_between(g["timestamp"],
                        g["mean_cno"] - g["std_cno"],
                        g["mean_cno"] + g["std_cno"],
                        alpha=0.15)
    title = ("Mean C/N0 over time (shaded band = ±1σ across tracked satellites)\n"
             "red < 30 dBHz (poor) · yellow 30–40 (marginal) · green > 40 (good)")
    if filter_note:
        title += f"\n{filter_note}"
    ax.set_title(title)
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


def plot_satcount(epoch: pd.DataFrame, out_stem: Path, labels: dict,
                  filter_note: str = "") -> None:
    fig, ax = plt.subplots(figsize=(14, 3))
    for rx, g in epoch.groupby("receiver"):
        g = g.sort_values("timestamp")
        ax.plot(g["timestamp"], g["sat_count"], label=labels.get(rx, rx),
                linewidth=0.6, alpha=0.85)
    title = "Tracked satellite count (common signals only)" if filter_note else "Tracked satellite count"
    if filter_note:
        title += f"\n{filter_note}"
    ax.set_title(title)
    ax.set_ylabel("Satellites")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    path = out_stem.parent / (out_stem.name + "_satcount.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"Plot    → {path}")


def plot_by_signal(df: pd.DataFrame, out_stem: Path,
                   filter_note: str = "") -> None:
    # Compute per-epoch per-signal means, then plot antenna-A minus antenna-B
    # delta with quadrature error bars.  Works with any two-receiver CSV.
    sig_col = "signal_id" if "signal_id" in df.columns else "gnss_id"
    receivers = sorted(df["receiver"].unique())
    rx_a, rx_b = receivers[0], receivers[1] if len(receivers) > 1 else (receivers[0], receivers[0])

    labels = make_rx_label(df)
    label_a = labels.get(rx_a, rx_a)
    label_b = labels.get(rx_b, rx_b)
    # Short antenna names for the y-axis label
    ant_a = (df[df["receiver"] == rx_a]["antenna_mount"].dropna().mode()
             if "antenna_mount" in df.columns else pd.Series([rx_a]))
    ant_b = (df[df["receiver"] == rx_b]["antenna_mount"].dropna().mode()
             if "antenna_mount" in df.columns else pd.Series([rx_b]))
    ant_a = ant_a.iloc[0] if not ant_a.empty else rx_a
    ant_b = ant_b.iloc[0] if not ant_b.empty else rx_b

    epoch_sig = (df.groupby(["timestamp", "receiver", sig_col])["cno_dBHz"]
                   .mean().reset_index())
    stats = (epoch_sig.groupby(["receiver", sig_col])["cno_dBHz"]
                      .agg(["mean", "std"]).reset_index())

    # Only include signals present in both receivers
    rx_a_sigs = set(stats[stats["receiver"] == rx_a][sig_col])
    rx_b_sigs = set(stats[stats["receiver"] == rx_b][sig_col])
    common_sigs = rx_a_sigs & rx_b_sigs
    signals = sorted(common_sigs)
    if not signals:
        print("Skip    → by_signal (no common signals between receivers)")
        return

    deltas, errs = [], []
    for s in signals:
        ra = stats[(stats["receiver"] == rx_a) & (stats[sig_col] == s)]
        rb = stats[(stats["receiver"] == rx_b) & (stats[sig_col] == s)]
        deltas.append(float(ra["mean"].iloc[0] - rb["mean"].iloc[0]))
        errs.append(float((ra["std"].iloc[0]**2 + rb["std"].iloc[0]**2) ** 0.5))

    fig, ax = plt.subplots(figsize=(max(8, len(signals) * 1.4), 4))
    colors = ["steelblue" if d >= 0 else "tomato" for d in deltas]
    ax.bar(signals, deltas, color=colors, alpha=0.8,
           yerr=errs, capsize=6, error_kw={"linewidth": 1.5, "color": "black"})
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel(f"{ant_a} − {ant_b}  ΔC/N0 (dBHz)")
    title = (f"{label_a}  vs  {label_b} — mean C/N0 by signal\n"
             "error bars = ±1σ quadrature (√(σ_A² + σ_B²)); "
             "bar crossing zero = not significant")
    if filter_note:
        title += f"\n{filter_note}"
    ax.set_title(title)
    plt.xticks(rotation=20, ha="right")
    ax.grid(True, alpha=0.3, axis="y"); fig.tight_layout()
    path = out_stem.parent / (out_stem.name + "_by_signal.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"Plot    → {path}")


_EL_BINS = list(range(0, 95, 5))   # [0, 5, 10, …, 90]
_MIN_OBS = 10                      # minimum observations per bin to plot


def _add_el_bin(df: pd.DataFrame) -> pd.DataFrame:
    """Add el_bin_deg = lower edge of the 5° elevation bin."""
    d = df.copy()
    d["el_bin_deg"] = (d["elev_deg"] // 5 * 5).clip(0, 85).astype(int)
    return d


def plot_skyplot(df: pd.DataFrame, out_stem: Path) -> None:
    """
    Azimuth / elevation polar scatter coloured by C/N0.
    One subplot per antenna_mount.  Zenith at centre, North up, clockwise.
    """
    if "elev_deg" not in df.columns or "azim_deg" not in df.columns:
        return
    sub = df.dropna(subset=["elev_deg", "azim_deg", "cno_dBHz"])
    if sub.empty:
        return

    mounts = sorted(sub["antenna_mount"].unique())
    vmin = sub["cno_dBHz"].quantile(0.02)
    vmax = sub["cno_dBHz"].quantile(0.98)

    fig, axes = plt.subplots(1, len(mounts),
                             figsize=(6 * len(mounts), 6),
                             subplot_kw={"projection": "polar"})
    if len(mounts) == 1:
        axes = [axes]

    for ax, mount in zip(axes, mounts):
        s = sub[sub["antenna_mount"] == mount]
        theta = np.radians(s["azim_deg"].values)
        r     = 90.0 - s["elev_deg"].values        # 0=zenith, 90=horizon
        sc = ax.scatter(theta, r, c=s["cno_dBHz"].values,
                        cmap="viridis", s=2, alpha=0.5,
                        vmin=vmin, vmax=vmax)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)                  # clockwise
        ax.set_ylim(0, 90)
        ax.set_yticks([0, 30, 60, 90])
        ax.set_yticklabels(["90°", "60°", "30°", "0°"], fontsize=7)
        ax.set_title(mount, fontsize=12, pad=15)
        plt.colorbar(sc, ax=ax, label="C/N0 (dBHz)", fraction=0.046, pad=0.04)

    fig.suptitle("Skyplot — satellite tracks coloured by C/N0\n"
                 "(zenith at centre, N up, clockwise)", fontsize=11)
    fig.tight_layout()
    path = out_stem.parent / (out_stem.name + "_skyplot.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot    → {path}")


def plot_cno_vs_elevation(df: pd.DataFrame, out_stem: Path) -> None:
    """
    Mean C/N0 ± 1σ vs elevation in 5° bins, per signal, antennas overlaid.
    Isolates antenna gain pattern from sky geometry.
    """
    if "elev_deg" not in df.columns:
        return
    sig_col = "signal_id" if "signal_id" in df.columns else "gnss_id"
    sub = _add_el_bin(df.dropna(subset=["elev_deg", "cno_dBHz"]))
    if sub.empty:
        return

    mounts  = sorted(sub["antenna_mount"].unique())
    signals = sorted(sub[sig_col].dropna().unique())
    colors  = ["steelblue", "tomato", "seagreen", "darkorange"]

    # Global y range so all signal panels share the same scale
    all_stats = (sub.groupby([sig_col, "antenna_mount", "el_bin_deg"])["cno_dBHz"]
                   .agg(mean="mean", std="std", n="count")
                   .reset_index())
    all_stats = all_stats[all_stats["n"] >= _MIN_OBS]
    if not all_stats.empty:
        ymin = (all_stats["mean"] - all_stats["std"]).min()
        ymax = (all_stats["mean"] + all_stats["std"]).max()
        pad  = (ymax - ymin) * 0.08
        ylim = (ymin - pad, ymax + pad)
    else:
        ylim = None

    fig, axes = plt.subplots(len(signals), 1,
                             figsize=(10, 3.5 * max(len(signals), 1)),
                             sharex=True, squeeze=False)
    for ax, sig in zip(axes[:, 0], signals):
        for color, mount in zip(colors, mounts):
            g = sub[(sub[sig_col] == sig) & (sub["antenna_mount"] == mount)]
            stats = (g.groupby("el_bin_deg")["cno_dBHz"]
                      .agg(mean="mean", std="std", n="count")
                      .reset_index())
            stats = stats[stats["n"] >= _MIN_OBS]
            if stats.empty:
                continue
            ax.errorbar(stats["el_bin_deg"] + 2.5, stats["mean"],
                        yerr=stats["std"], label=mount, color=color,
                        linewidth=1.2, marker="o", markersize=3, capsize=3)
        if ylim:
            ax.set_ylim(ylim)
        _add_cno_shading(ax)
        ax.set_ylabel("C/N0 (dBHz)")
        ax.set_title(sig)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Elevation (°)")
    fig.suptitle("C/N0 vs elevation (5° bins, mean ± 1σ)\n"
                 "Isolates antenna gain pattern from sky geometry",
                 fontsize=11, y=1.002)
    fig.tight_layout()
    path = out_stem.parent / (out_stem.name + "_cno_vs_elev.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot    → {path}")


def plot_used_vs_elevation(df: pd.DataFrame, out_stem: Path) -> None:
    """
    Fraction of tracked satellites included in the timing solution vs elevation.
    Reveals the effective elevation mask applied by the receiver.
    """
    if "elev_deg" not in df.columns or "used" not in df.columns:
        return
    if not df["used"].any():
        print("Skip    → used_vs_elev (used flag always 0 — needs UBX NAV-SAT output enabled)")
        return
    sub = _add_el_bin(df.dropna(subset=["elev_deg"]))
    if sub.empty:
        return

    mounts = sorted(sub["antenna_mount"].unique())
    colors = ["steelblue", "tomato", "seagreen", "darkorange"]

    fig, ax = plt.subplots(figsize=(10, 4))
    for color, mount in zip(colors, mounts):
        g = sub[sub["antenna_mount"] == mount]
        stats = (g.groupby("el_bin_deg")["used"]
                   .agg(frac="mean", n="count")
                   .reset_index())
        stats = stats[stats["n"] >= _MIN_OBS]
        if stats.empty:
            continue
        ax.plot(stats["el_bin_deg"] + 2.5, stats["frac"] * 100,
                label=mount, color=color, linewidth=1.2, marker="o", markersize=3)

    ax.set_xlabel("Elevation (°)")
    ax.set_ylabel("Used (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Fraction of visible satellites used in timing solution vs elevation\n"
                 "(effective elevation mask applied by receiver)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_stem.parent / (out_stem.name + "_used_vs_elev.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Plot    → {path}")


def _load_toml(path: Path) -> dict:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]
    with open(path, "rb") as f:
        return tomllib.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True, help="Output filename stem")
    ap.add_argument("--receivers", default=None,
                    help="Receiver config (receivers.toml) to restrict "
                         "comparisons to the signal intersection")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_stem = Path(args.out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {csv_path} …")
    df = load(csv_path)
    print(f"  {len(df):,} rows  {df['timestamp'].nunique()} epochs")

    # Filter to signal intersection if receiver config provided
    filter_note = ""
    if args.receivers:
        from testant.signals import (
            load_receiver_signals, signal_intersection, exclusion_note,
        )
        rx_cfg = _load_toml(Path(args.receivers))
        rx_signals = load_receiver_signals(rx_cfg)
        common = signal_intersection(rx_signals)
        if common and "gnss_id" in df.columns:
            filter_note = exclusion_note(rx_signals, common)
            before = len(df)
            df = df[df["gnss_id"].isin(common)].copy()
            dropped = before - len(df)
            if dropped:
                print(f"  Signal intersection filter: dropped {dropped:,} rows")
            if filter_note:
                print(f"  {filter_note}")

    epoch = epoch_stats(df)
    labels = make_rx_label(df)
    write_report(df, epoch, out_stem)
    plot_cno(epoch, out_stem, labels, filter_note)
    plot_delta(epoch, out_stem)
    plot_satcount(epoch, out_stem, labels, filter_note)
    plot_by_signal(df, out_stem, filter_note)
    plot_skyplot(df, out_stem)
    plot_cno_vs_elevation(df, out_stem)
    plot_used_vs_elevation(df, out_stem)
    print("Done.")


if __name__ == "__main__":
    main()
