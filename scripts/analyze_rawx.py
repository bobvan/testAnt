#!/usr/bin/env python3
"""
analyze_rawx.py — Code-minus-carrier (CMC) and cycle-slip analysis from RAWX CSV.

Usage:
    python scripts/analyze_rawx.py --csv data/foo_rawx.csv --out data/foo \
        [--snr data/foo.csv]

The optional --snr argument supplies the companion SNR/C/N0 CSV (produced by
log_snr.py) which provides elevation and azimuth for each satellite.  Without
it, elevation-dependent plots are skipped.

Outputs (all suffixed onto <out>):
    _rawx_report.txt      — CMC noise floor, lock stats, cycle slip summary
    _cmc_by_signal.png    — detrended CMC time series per signal
    _cmc_diff.png         — CMC_A − CMC_B per signal (receiver noise floor)
    _lock_duration.png    — carrier phase lock duration CDF per signal
    _cmc_vs_elev.png      — detrended CMC std vs elevation (requires --snr)
    _cmc_skyplot.png      — |CMC| polar map by direction (requires --snr)
    _slip_timeline.png    — cycle slip raster (colour=signal, shape=antenna, size=severity)
    _slip_quality.png     — slip rate vs C/N0 and elevation bins (normalised by exposure)
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


# ── Signal wavelengths (c/f, metres) ────────────────────────────────── #
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
    "BDS-B1C":     _C / 1_575_420_000.0,
    "BDS-B1C-Q":   _C / 1_575_420_000.0,
    "BDS-B2I":     _C / 1_207_140_000.0,   # 0.24834 m
    "BDS-B2I-D2":  _C / 1_207_140_000.0,
    "GAL-E5bI":    _C / 1_207_140_000.0,
    "GAL-E5bQ":    _C / 1_207_140_000.0,
    "GPS-L2CL":    _C / 1_227_600_000.0,   # 0.24421 m
    "GPS-L2CM":    _C / 1_227_600_000.0,
    "GLO-L1OF":    _C / 1_602_000_000.0,
    "GLO-L2OF":    _C / 1_246_000_000.0,
}

_EL_BINS = list(range(0, 95, 5))   # 0, 5, 10, …, 90
_MIN_OBS = 20                      # minimum obs per bin to plot


# ── label helpers ────────────────────────────────────────────────────── #

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


# ── loading ──────────────────────────────────────────────────────────── #

def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if "locktime_ms" in df.columns:
        df = df.rename(columns={"locktime_ms": "lock_duration_ms"})
    return df


def load_snr(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def add_elevation(rawx: pd.DataFrame, snr: pd.DataFrame) -> pd.DataFrame:
    """
    Join RAWX with the SNR CSV to add elev_deg and azim_deg columns.

    Joins on floor(timestamp, 1s) × receiver × gnss_id × sv_id × signal_id.
    Rows with no matching SNR entry keep NaN elevation/azimuth.
    """
    rawx = rawx.copy()
    snr  = snr.copy()
    rawx["_ts_s"] = rawx["timestamp"].dt.floor("s")
    snr["_ts_s"]  = snr["timestamp"].dt.floor("s")

    el_lut = (snr.groupby(["_ts_s", "receiver", "gnss_id", "sv_id", "signal_id"])
                 .agg(elev_deg=("elev_deg", "median"),
                      azim_deg=("azim_deg", "median"))
                 .reset_index())

    rawx = rawx.merge(el_lut,
                      on=["_ts_s", "receiver", "gnss_id", "sv_id", "signal_id"],
                      how="left")
    rawx = rawx.drop(columns=["_ts_s"])
    n_matched = rawx["elev_deg"].notna().sum()
    print(f"  Elevation join: {n_matched:,} / {len(rawx):,} rows matched")
    return rawx


# ── CMC ──────────────────────────────────────────────────────────────── #

def add_cmc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cmc_m = pseudorange_m − wavelength × carrier_phase_cy.

    Only rows with cp_valid=1 AND half_cyc=1 (half-cycle resolved) and a
    known signal wavelength get a finite value; others are NaN.
    """
    df = df.copy()
    wl    = df["signal_id"].map(_WAVELENGTH)
    cp_ok = (df["cp_valid"] == 1) & (df["half_cyc"] == 1)
    df["cmc_m"] = np.where(
        cp_ok & wl.notna(),
        df["pseudorange_m"] - wl * df["carrier_phase_cy"],
        np.nan,
    )
    return df


def add_cmc_detrended(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cmc_detrended_m by subtracting each SV's mean CMC (per receiver arc).

    Removes the integer-ambiguity offset (N·λ) and receiver hardware bias,
    leaving short-term multipath variability and code noise.
    """
    df = df.copy()
    per_sv_mean = (df.groupby(["receiver", "signal_id", "sv_id"])["cmc_m"]
                     .transform("mean"))
    df["cmc_detrended_m"] = df["cmc_m"] - per_sv_mean
    return df


# ── cycle slip detection ─────────────────────────────────────────────── #

def detect_cycle_slips(df: pd.DataFrame,
                       max_gap_s: float   = 1.5,
                       min_drop_ms: float = 500.0) -> pd.DataFrame:
    """
    Detect carrier-phase cycle slips from resets in lock_duration_ms.

    lock_duration_ms counts up from 0 when the receiver acquires phase lock.
    A cycle slip (or loss-of-lock) causes it to reset.  Within consecutive
    epochs (time gap ≤ max_gap_s), a drop of > min_drop_ms from the previous
    epoch is recorded as a slip.

    Parameters
    ----------
    max_gap_s   : seconds — maximum inter-epoch gap still treated as consecutive
    min_drop_ms : ms — minimum decrease in lock_duration to count as a slip
                  (500 ms = half an epoch at 1 Hz, tolerates quantisation)

    Returns a DataFrame with one row per slip:
        timestamp, receiver, antenna_mount, gnss_id, sv_id, signal_id,
        lock_before_ms, lock_after_ms, drop_ms
    """
    if "lock_duration_ms" not in df.columns:
        return pd.DataFrame()

    group_keys = ["receiver", "antenna_mount", "gnss_id", "sv_id", "signal_id"]
    slips = []

    for keys, grp in df.groupby(group_keys, sort=False):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        ts   = grp["timestamp"]
        lock = grp["lock_duration_ms"]

        gap_s  = (ts - ts.shift(1)).dt.total_seconds()
        drop   = lock.shift(1) - lock   # positive = lock decreased

        is_slip = (gap_s <= max_gap_s) & (drop > min_drop_ms) & drop.notna()

        for idx in grp[is_slip].index:
            s = dict(zip(group_keys, keys))
            s["timestamp"]      = grp["timestamp"].iloc[idx]
            s["lock_before_ms"] = float(lock.iloc[idx - 1])
            s["lock_after_ms"]  = float(lock.iloc[idx])
            s["drop_ms"]        = float(drop.iloc[idx])
            # Signal quality at the last good epoch before the slip
            s["cno_before"] = (float(grp["cno_dBHz"].iloc[idx - 1])
                               if "cno_dBHz" in grp.columns else float("nan"))
            s["elev_before"] = (float(grp["elev_deg"].iloc[idx - 1])
                                if "elev_deg" in grp.columns
                                and not pd.isna(grp["elev_deg"].iloc[idx - 1])
                                else float("nan"))
            slips.append(s)

    return pd.DataFrame(slips)


# ── report ───────────────────────────────────────────────────────────── #

def write_report(df: pd.DataFrame, slips: pd.DataFrame,
                 out_stem: Path) -> None:
    lines = []
    a = lines.append
    labels = make_rx_label(df)

    dur_h = (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 3600
    a("=" * 62)
    a("  testAnt RAWX / CMC report")
    a("=" * 62)
    a(f"  Start    : {df['timestamp'].min()}")
    a(f"  End      : {df['timestamp'].max()}")
    a(f"  Duration : {dur_h:.2f} h")
    a(f"  Total rows (pr_valid): {len(df):,}")
    a("")

    cmc_ok = df.dropna(subset=["cmc_detrended_m"])
    a("── Detrended CMC std dev by signal & antenna (noise floor) ─────")
    a(f"  {'Signal':16s}  {'Antenna':24s}  {'N_sv':>5s}  {'N_obs':>6s}  {'std_m':>8s}")
    sv_counts = (cmc_ok.groupby(["receiver", "signal_id"])["sv_id"]
                       .nunique().rename("n_sv").reset_index())
    stats = (cmc_ok.groupby(["signal_id", "receiver"])["cmc_detrended_m"]
               .agg(n_obs="count", std="std").reset_index()
               .sort_values(["signal_id", "receiver"]))
    stats = stats.merge(sv_counts, on=["receiver", "signal_id"])
    for _, row in stats.iterrows():
        lbl = labels.get(row['receiver'], row['receiver'])
        a(f"  {row['signal_id']:16s}  {lbl:24s}  "
          f"{int(row['n_sv']):>5d}  {int(row['n_obs']):>6d}  "
          f"{row['std']:>8.3f} m")
    a("")

    a("── Lock duration stats by signal & antenna (ms) ────────────────")
    a(f"  {'Signal':16s}  {'Antenna':24s}  {'median_ms':>10s}  {'p95_ms':>8s}")
    lt_stats = (df.groupby(["signal_id", "receiver"])["lock_duration_ms"]
                  .agg(median="median", p95=lambda x: x.quantile(0.95))
                  .reset_index()
                  .sort_values(["signal_id", "receiver"]))
    for _, row in lt_stats.iterrows():
        lbl = labels.get(row['receiver'], row['receiver'])
        a(f"  {row['signal_id']:16s}  {lbl:24s}  "
          f"{row['median']:>10.0f}  {row['p95']:>8.0f}")
    a("")

    # ── cycle slip section ──
    a("── Cycle slip summary ──────────────────────────────────────────")
    a("  Detection method: lock_duration_ms drop > 500 ms in consecutive epochs")
    a("  Note: u-blox caps lock_duration at 64,500 ms — only slips that cause")
    a("  a visible reset within ≤1.5 s are counted; subtle within-lock-arc slips")
    a("  are not detectable from this field alone.")
    if slips.empty:
        a("  No cycle slips detected.")
    else:
        # Total SV-seconds tracked per (receiver, signal) to compute rate
        track_s = (df.groupby(["receiver", "signal_id"])
                     .size()
                     .rename("n_epochs")
                     .reset_index())
        # Slips per (receiver, signal)
        slip_counts = (slips.groupby(["receiver", "signal_id"])
                            .size()
                            .rename("n_slips")
                            .reset_index())
        summary = track_s.merge(slip_counts, on=["receiver", "signal_id"], how="left")
        summary["n_slips"] = summary["n_slips"].fillna(0).astype(int)
        # Rate = slips / tracking_hours * 24  →  slips per 24 h
        summary["rate_per_24h"] = (summary["n_slips"]
                                   / (summary["n_epochs"] / 3600.0)
                                   * 24.0)

        a(f"  {'Signal':16s}  {'Antenna':24s}  {'Slips':>6s}  "
          f"{'Track-h':>7s}  {'Rate/24h':>9s}")
        for _, row in summary.sort_values(["signal_id", "receiver"]).iterrows():
            lbl = labels.get(row['receiver'], row['receiver'])
            a(f"  {row['signal_id']:16s}  {lbl:24s}  "
              f"{int(row['n_slips']):>6d}  "
              f"{row['n_epochs']/3600:>7.2f}  "
              f"{row['rate_per_24h']:>9.1f}")
        a("")
        total_slips = int(slips["drop_ms"].count())
        total_track_h = track_s["n_epochs"].sum() / 3600.0
        overall_rate  = total_slips / total_track_h * 24.0 if total_track_h > 0 else 0.0
        a(f"  Total: {total_slips} slips in {total_track_h:.2f} SV-hours  "
          f"→  {overall_rate:.1f} slips/24 h (all signals combined)")
        a("")
        # Quality-filtered slip counts
        if "cno_before" in slips.columns:
            hi_cno  = slips["cno_before"] > 35.0
            hi_elev = (slips["elev_before"] > 30.0
                       if "elev_before" in slips.columns
                       else pd.Series(True, index=slips.index))
            hi_both = hi_cno & hi_elev.fillna(True)
            a("── Slips on strong / high-elevation SVs ─────────────────────")
            a("  (C/N0 > 35 dBHz AND elevation > 30° — these matter most)")
            a(f"  {'Antenna':12s}  {'Total':>6s}  {'>35dBHz':>8s}  {'>30° elev':>10s}  {'Both':>6s}")
            for ant in sorted(slips["antenna_mount"].dropna().unique()):
                m = slips["antenna_mount"] == ant
                a(f"  {ant:12s}  "
                  f"{m.sum():>6d}  "
                  f"{(m & hi_cno).sum():>8d}  "
                  f"{(m & hi_elev.fillna(False)).sum():>10d}  "
                  f"{(m & hi_both).sum():>6d}")
            a("")

        a("  10 largest slips:")
        a(f"  {'Timestamp':26s}  {'Antenna':22s}  {'Sig':14s}  SV  "
          f"{'Before ms':>10s}  {'After ms':>9s}  {'Drop ms':>8s}")
        for _, row in slips.nlargest(10, "drop_ms").iterrows():
            ant_lbl = (f"{row['antenna_mount']} ({row['receiver']})"
                       if "antenna_mount" in row.index else row['receiver'])
            a(f"  {str(row['timestamp']):26s}  "
              f"{ant_lbl:22s}  {row['signal_id']:14s}  "
              f"{int(row['sv_id']):>2d}  "
              f"{row['lock_before_ms']:>10.0f}  "
              f"{row['lock_after_ms']:>9.0f}  "
              f"{row['drop_ms']:>8.0f}")
    a("")
    a("=" * 62)

    path = out_stem.parent / (out_stem.name + "_rawx_report.txt")
    path.write_text("\n".join(lines) + "\n")
    print(f"Report  → {path}")


# ── plots ────────────────────────────────────────────────────────────── #

def plot_cmc_by_signal(df: pd.DataFrame, out_stem: Path) -> None:
    """Detrended CMC time series per signal, receivers overlaid."""
    cmc_ok = df.dropna(subset=["cmc_detrended_m"])
    signals = sorted(cmc_ok["signal_id"].unique())
    if not signals:
        return

    receivers = sorted(cmc_ok["receiver"].unique())
    colors    = ["steelblue", "tomato", "seagreen", "darkorange"]
    labels    = make_rx_label(cmc_ok)

    fig, axes = plt.subplots(len(signals), 1,
                             figsize=(14, 3 * len(signals)),
                             sharex=True, squeeze=False)
    for ax, sig in zip(axes[:, 0], signals):
        sub = cmc_ok[cmc_ok["signal_id"] == sig]
        epoch_med = (sub.groupby(["timestamp", "receiver"])["cmc_detrended_m"]
                       .median().reset_index())
        for color, rx in zip(colors, receivers):
            g = epoch_med[epoch_med["receiver"] == rx].sort_values("timestamp")
            if g.empty:
                continue
            ax.plot(g["timestamp"], g["cmc_detrended_m"], label=labels.get(rx, rx),
                    linewidth=0.6, alpha=0.85, color=color)
        ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
        ax.set_ylabel("CMC (m)")
        ax.set_title(sig)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    fig.suptitle("Detrended CMC by signal — epoch median across SVs\n"
                 "(per-SV mean subtracted; integer ambiguity + hardware bias removed)",
                 fontsize=11, y=1.002)
    fig.tight_layout()
    path = out_stem.parent / (out_stem.name + "_cmc_by_signal.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot    → {path}")


def plot_cmc_diff(df: pd.DataFrame, out_stem: Path,
                  filter_note: str = "") -> None:
    """
    Detrended CMC difference (rx_a − rx_b) matched per SV per epoch.
    Cancels per-SV integer ambiguity; residual = receiver noise + diff multipath.
    """
    cmc_ok = df.dropna(subset=["cmc_detrended_m"]).copy()
    receivers = sorted(cmc_ok["receiver"].unique())
    if len(receivers) < 2:
        return
    rx_a, rx_b = receivers[0], receivers[1]

    signals = sorted(cmc_ok["signal_id"].dropna().unique())
    if not signals:
        return

    cmc_ok["ts_s"] = cmc_ok["timestamp"].dt.floor("s")
    labels = make_rx_label(cmc_ok)
    label_a = labels.get(rx_a, rx_a)
    label_b = labels.get(rx_b, rx_b)
    ant_a = (cmc_ok[cmc_ok["receiver"] == rx_a]["antenna_mount"].dropna().mode()
             if "antenna_mount" in cmc_ok.columns else pd.Series([rx_a]))
    ant_b = (cmc_ok[cmc_ok["receiver"] == rx_b]["antenna_mount"].dropna().mode()
             if "antenna_mount" in cmc_ok.columns else pd.Series([rx_b]))
    ant_a = ant_a.iloc[0] if not ant_a.empty else rx_a
    ant_b = ant_b.iloc[0] if not ant_b.empty else rx_b

    fig, axes = plt.subplots(len(signals), 1,
                             figsize=(14, 3 * len(signals)),
                             sharex=True, squeeze=False)
    for ax, sig in zip(axes[:, 0], signals):
        sub = cmc_ok[cmc_ok["signal_id"] == sig]
        piv = sub.pivot_table(
            index=["ts_s", "sv_id"],
            columns="receiver",
            values="cmc_detrended_m",
            aggfunc="median",
        )
        if rx_a not in piv.columns or rx_b not in piv.columns:
            ax.set_title(f"{sig} — one receiver absent")
            continue
        piv = piv[[rx_a, rx_b]].dropna()
        if piv.empty:
            ax.set_title(f"{sig} — no matched SV/epoch pairs")
            continue

        piv["diff"] = piv[rx_a] - piv[rx_b]
        epoch_diff  = piv.groupby("ts_s")["diff"].median().sort_index()
        n_sv        = int(piv.groupby("ts_s")["diff"].count().median())

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.fill_between(epoch_diff.index, epoch_diff.values, 0,
                        where=epoch_diff.values >= 0, alpha=0.25, color="steelblue")
        ax.fill_between(epoch_diff.index, epoch_diff.values, 0,
                        where=epoch_diff.values < 0,  alpha=0.25, color="tomato")
        ax.plot(epoch_diff.index,
                epoch_diff.rolling(60, min_periods=10).median(),
                color="navy", linewidth=1.0, label="60-s median")
        ax.set_ylabel(f"{ant_a} − {ant_b}  ΔCMC (m)")
        ax.set_title(f"{sig}  —  {label_a} − {label_b}  "
                     f"(std={epoch_diff.std():.3f} m, median {n_sv} SVs/epoch)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    suptitle = (f"Detrended CMC difference ({label_a} − {label_b}) by signal\n"
                "per-SV matched; residual = receiver noise + differential multipath")
    if filter_note:
        suptitle += f"\n{filter_note}"
    fig.suptitle(suptitle, fontsize=11, y=1.002)
    fig.tight_layout()
    path = out_stem.parent / (out_stem.name + "_cmc_diff.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot    → {path}")


def plot_lock_duration(df: pd.DataFrame, out_stem: Path) -> None:
    """Carrier phase lock duration CDF per signal."""
    signals = sorted(df["signal_id"].dropna().unique())
    if not signals:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for color, sig in zip(plt.cm.tab10.colors, signals):
        sub = df[df["signal_id"] == sig]["lock_duration_ms"].dropna()
        if sub.empty:
            continue
        x = np.sort(sub.values)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, label=sig, linewidth=1.2, color=color)

    ax.set_xlabel("Lock duration (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Carrier phase lock duration CDF by signal")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    path = out_stem.parent / (out_stem.name + "_lock_duration.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Plot    → {path}")


def plot_cmc_vs_elevation(df: pd.DataFrame, out_stem: Path) -> None:
    """
    Detrended CMC noise (std dev) vs elevation in 5° bins, per signal.
    High CMC at low elevation = multipath; floor at high elevation = code noise.
    Requires elevation column (add_elevation must have been called).
    """
    if "elev_deg" not in df.columns:
        return
    cmc_ok = df.dropna(subset=["cmc_detrended_m", "elev_deg"]).copy()
    if cmc_ok.empty:
        return

    cmc_ok["el_bin_deg"] = (cmc_ok["elev_deg"] // 5 * 5).clip(0, 85).astype(int)

    signals   = sorted(cmc_ok["signal_id"].unique())
    receivers = sorted(cmc_ok["receiver"].unique())
    colors    = ["steelblue", "tomato", "seagreen", "darkorange"]
    labels    = make_rx_label(cmc_ok)

    fig, axes = plt.subplots(len(signals), 1,
                             figsize=(10, 3.5 * max(len(signals), 1)),
                             sharex=True, squeeze=False)
    for ax, sig in zip(axes[:, 0], signals):
        sub = cmc_ok[cmc_ok["signal_id"] == sig]
        for color, rx in zip(colors, receivers):
            g = sub[sub["receiver"] == rx]
            stats = (g.groupby("el_bin_deg")["cmc_detrended_m"]
                      .agg(std="std", n="count")
                      .reset_index())
            stats = stats[stats["n"] >= _MIN_OBS]
            if stats.empty:
                continue
            ax.plot(stats["el_bin_deg"] + 2.5, stats["std"],
                    label=labels.get(rx, rx), color=color,
                    linewidth=1.2, marker="o", markersize=3)
        ax.set_ylabel("CMC std (m)")
        ax.set_title(sig)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Elevation (°)")
    fig.suptitle("Detrended CMC noise vs elevation (5° bins)\n"
                 "Rising towards horizon = multipath;  "
                 "floor at zenith = code noise",
                 fontsize=11, y=1.002)
    fig.tight_layout()
    path = out_stem.parent / (out_stem.name + "_cmc_vs_elev.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot    → {path}")


def plot_cmc_skyplot(df: pd.DataFrame, out_stem: Path) -> None:
    """
    Polar skyplot of |detrended CMC|, one subplot per receiver.
    Bright spots indicate multipath hotspots (reflecting surfaces in that direction).
    Requires elevation/azimuth columns.
    """
    if "elev_deg" not in df.columns or "azim_deg" not in df.columns:
        return
    cmc_ok = df.dropna(subset=["cmc_detrended_m", "elev_deg", "azim_deg"]).copy()
    if cmc_ok.empty:
        return

    panels = sorted(cmc_ok["antenna_mount"].unique())
    vmax = float(cmc_ok["cmc_detrended_m"].abs().quantile(0.95))

    fig, axes = plt.subplots(1, len(panels),
                             figsize=(7 * len(panels), 7),
                             subplot_kw={"projection": "polar"})
    if len(panels) == 1:
        axes = [axes]

    for ax, ant in zip(axes, panels):
        sub   = cmc_ok[cmc_ok["antenna_mount"] == ant]
        theta = np.radians(sub["azim_deg"].values)
        r     = 90.0 - sub["elev_deg"].values      # 0=zenith, 90=horizon
        sc = ax.scatter(theta, r,
                        c=sub["cmc_detrended_m"].abs().values,
                        cmap="hot", s=3, alpha=0.5,
                        vmin=0, vmax=vmax)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 90)
        ax.set_yticks([0, 30, 60, 90])
        ax.set_yticklabels(["90°", "60°", "30°", "0°"], fontsize=7)
        # Build title: "antenna_mount @ mount_site (RECEIVER)"
        if "mount_site" in sub.columns:
            site = sub["mount_site"].dropna()
            site_str = f" @ {site.iloc[0]}" if not site.empty and site.iloc[0] else ""
        else:
            site_str = ""
        rx = sub["receiver"].dropna()
        rx_str = f" ({rx.iloc[0]})" if not rx.empty else ""
        ax.set_title(f"{ant}{site_str}{rx_str}", fontsize=12, pad=15)
        plt.colorbar(sc, ax=ax, label="|CMC| (m)", fraction=0.046, pad=0.04)

    fig.suptitle("CMC multipath skyplot — |detrended CMC| by direction\n"
                 "(zenith at centre, N up, clockwise; bright = high multipath)",
                 fontsize=11)
    fig.tight_layout()
    path = out_stem.parent / (out_stem.name + "_cmc_skyplot.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot    → {path}")


def plot_cycle_slip_timeline(slips: pd.DataFrame, out_stem: Path) -> None:
    """
    Timeline of cycle slip events.

    Top panel — raster plot, one horizontal lane per (gnss_id, sv_id):
      colour = signal_id       (which frequency slipped)
      shape  = antenna_mount   (which antenna)
      size   = drop_ms         (how much lock time was lost; larger = worse)
    Red shading marks windows where ≥3 SVs slip within 5 s — potential
    interference burst, ionospheric event, or receiver reset.

    Bottom panel — 1-minute stacked histogram of slip counts by signal.

    Patterns to look for:
      • Vertical stripe (many SVs at once) → interference / ionospheric event /
        receiver reset.  Both antennas same time = common-mode; one only = local.
      • Same SV, both antennas → satellite-side anomaly (manoeuvre, signal anomaly)
      • Same SV, one antenna only → local multipath or blockage at that mount
      • Cluster near run start → initial lock acquisition (not real slips)
      • L5 slips >> L1 slips on same antenna → antenna poorly suited for L5
    """
    if slips.empty:
        return

    slips = slips.copy()

    # ── SV labels and y-positions ──────────────────────────────────────
    _PREFIX = {"GPS": "G", "GAL": "E", "BDS": "C", "GLO": "R",
               "QZSS": "J", "SBAS": "S"}
    all_svs = (slips[["gnss_id", "sv_id"]]
               .drop_duplicates()
               .sort_values(["gnss_id", "sv_id"])
               .reset_index(drop=True))
    all_svs["label"] = all_svs.apply(
        lambda r: f"{_PREFIX.get(r['gnss_id'], r['gnss_id'][0])}{int(r['sv_id']):02d}",
        axis=1)
    all_svs["y"] = all_svs.index
    sv_y     = {(r.gnss_id, r.sv_id): r.y for r in all_svs.itertuples()}
    sv_label = dict(zip(all_svs["y"], all_svs["label"]))
    slips["_y"] = slips.apply(lambda r: sv_y.get((r["gnss_id"], r["sv_id"]), np.nan), axis=1)

    # ── Colour by signal_id ────────────────────────────────────────────
    signals   = sorted(slips["signal_id"].dropna().unique())
    cmap10    = matplotlib.colormaps.get_cmap("tab10").resampled(max(len(signals), 1))
    sig_color = {s: cmap10(i) for i, s in enumerate(signals)}

    # ── Marker shape by antenna_mount ──────────────────────────────────
    _MARKERS  = ["o", "s", "^", "D"]
    ants      = sorted(slips["antenna_mount"].dropna().unique())
    ant_marker = {a: _MARKERS[i % len(_MARKERS)] for i, a in enumerate(ants)}

    # ── Marker size proportional to drop_ms ───────────────────────────
    drop = slips["drop_ms"].clip(upper=64500)
    dr   = drop.max() - drop.min()
    slips["_sz"] = 40 + 260 * ((drop - drop.min()) / dr) if dr > 0 else 120

    # ── Layout ────────────────────────────────────────────────────────
    n_svs  = len(all_svs)
    fig_h  = max(3, 0.35 * n_svs)
    fig, (ax_r, ax_h) = plt.subplots(
        2, 1, figsize=(14, fig_h + 2.5),
        gridspec_kw={"height_ratios": [fig_h, 2]},
        sharex=True)

    # ── Raster panel ──────────────────────────────────────────────────
    # Faint horizontal grid lines between SV lanes
    for y in sv_label:
        ax_r.axhline(y, color="lightgrey", linewidth=0.4, zorder=1)

    sig_handles = {}
    ant_handles = {}
    for ant in ants:
        for sig in signals:
            sub = slips[(slips["antenna_mount"] == ant) & (slips["signal_id"] == sig)]
            if sub.empty:
                continue
            ax_r.scatter(sub["timestamp"], sub["_y"],
                         c=[sig_color[sig]] * len(sub),
                         marker=ant_marker[ant],
                         s=sub["_sz"], alpha=0.85, zorder=3)
            if sig not in sig_handles:
                sig_handles[sig] = plt.Line2D(
                    [0], [0], marker="o", color="w",
                    markerfacecolor=sig_color[sig], markersize=8, label=sig)
            if ant not in ant_handles:
                ant_handles[ant] = plt.Line2D(
                    [0], [0], marker=ant_marker[ant], color="grey",
                    markersize=8, linestyle="none", label=ant)

    # Red shading for simultaneous bursts (≥3 SVs within 5 s)
    slip_5s = slips.set_index("timestamp").resample("5s").size()
    for t_start in slip_5s[slip_5s >= 3].index:
        ax_r.axvspan(t_start, t_start + pd.Timedelta(seconds=5),
                     alpha=0.10, color="red", zorder=0)

    ax_r.set_yticks(list(sv_label.keys()))
    ax_r.set_yticklabels(list(sv_label.values()), fontsize=7)
    ax_r.yaxis.set_tick_params(length=0)
    ax_r.set_ylabel("Satellite")
    ax_r.grid(axis="x", alpha=0.3)

    all_handles = list(sig_handles.values()) + list(ant_handles.values())
    if all_handles:
        ax_r.legend(handles=all_handles, fontsize=7, loc="upper right",
                    ncol=max(1, len(all_handles) // 5))

    # ── Histogram panel ───────────────────────────────────────────────
    t0   = slips["timestamp"].min().floor("min")
    t1   = slips["timestamp"].max().ceil("min") + pd.Timedelta(minutes=1)
    bins = pd.date_range(t0, t1, freq="1min")
    if len(bins) > 1:
        bin_mids = bins[:-1] + pd.Timedelta(seconds=30)
        width_days = 0.85 / 1440          # 0.85 min in matplotlib date units
        bottom = np.zeros(len(bin_mids))
        for sig in signals:
            sub = slips[slips["signal_id"] == sig]
            counts, _ = np.histogram(
                sub["timestamp"].values.astype("datetime64[ns]").astype(np.int64),
                bins=bins.values.astype("datetime64[ns]").astype(np.int64))
            ax_h.bar(mdates.date2num(bin_mids), counts, bottom=bottom,
                     width=width_days, color=sig_color[sig], alpha=0.85, label=sig)
            bottom += counts
    ax_h.set_ylabel("Slips / min")
    ax_h.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=4))
    ax_h.grid(axis="y", alpha=0.3)
    ax_h.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()

    fig.suptitle(
        "Cycle slip timeline\n"
        "colour = signal,  shape = antenna,  size ∝ lock lost (ms);  "
        "red band = ≥3 SVs within 5 s",
        fontsize=10, y=1.01)
    fig.tight_layout()
    path = out_stem.parent / (out_stem.name + "_slip_timeline.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot    → {path}")


def plot_slip_quality(df: pd.DataFrame, slips: pd.DataFrame,
                     out_stem: Path) -> None:
    """
    Slip rate normalised by tracking exposure, broken down by C/N0 and
    elevation bins.  Answers: are the excess slips on weak / low-elevation
    SVs (expected, low impact) or on strong / high-elevation SVs (real
    antenna quality problem)?

    Panel layout (2 × 2):
      [0,0] Slip rate (per SV-hour) vs C/N0 bin — bars per antenna
      [0,1] Scatter of (elevation, C/N0) at slip time — one point per slip
      [1,0] Slip rate (per SV-hour) vs elevation bin — bars per antenna
      [1,1] Cumulative slip fraction vs elevation — steep rise at low
             elevation is expected; a high-elevation tail is a red flag
    Bottom row requires elevation data (--snr); omitted otherwise.
    """
    if slips.empty or "cno_before" not in slips.columns:
        return

    ants      = sorted(slips["antenna_mount"].dropna().unique())
    _COLORS   = ["steelblue", "tomato", "seagreen", "darkorange"]
    _MARKERS  = ["o", "s", "^", "D"]
    ant_color  = {a: _COLORS[i % len(_COLORS)]  for i, a in enumerate(ants)}
    ant_marker = {a: _MARKERS[i % len(_MARKERS)] for i, a in enumerate(ants)}

    has_elev = ("elev_before" in slips.columns
                and slips["elev_before"].notna().any()
                and "elev_deg" in df.columns)

    # ── binning definitions ───────────────────────────────────────────
    cno_edges  = [0,  25, 30, 35, 40, 45, 100]
    cno_labels = ["<25", "25-30", "30-35", "35-40", "40-45", ">45"]
    el_edges   = [0, 10, 20, 30, 45, 60, 91]
    el_labels  = ["0-10", "10-20", "20-30", "30-45", "45-60", ">60"]

    def bar_rate(ax, merged, bin_col, labels, ant_list):
        """Draw grouped bars of slip rate per SV-hour."""
        x     = np.arange(len(labels))
        width = 0.8 / max(len(ant_list), 1)
        for i, ant in enumerate(ant_list):
            sub   = merged[merged["antenna_mount"] == ant].set_index(bin_col)
            rates = [float(sub.loc[lbl, "rate"]) if lbl in sub.index else 0.0
                     for lbl in labels]
            ax.bar(x + i * width - (len(ant_list) - 1) * width / 2,
                   rates, width=width * 0.9,
                   color=ant_color[ant], label=ant, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("Slips / SV-hour")

    # ── C/N0 exposure and slip counts ─────────────────────────────────
    df_c                = df.copy()
    df_c["cno_bin"]     = pd.cut(df_c["cno_dBHz"],        cno_edges,
                                 labels=cno_labels, right=False)
    slips_c             = slips.dropna(subset=["cno_before"]).copy()
    slips_c["cno_bin"]  = pd.cut(slips_c["cno_before"],   cno_edges,
                                 labels=cno_labels, right=False)
    exp_cno   = (df_c.groupby(["antenna_mount", "cno_bin"], observed=True)
                     .size().rename("n_epochs").reset_index())
    cnt_cno   = (slips_c.groupby(["antenna_mount", "cno_bin"], observed=True)
                        .size().rename("n_slips").reset_index())
    mgd_cno   = exp_cno.merge(cnt_cno, on=["antenna_mount", "cno_bin"], how="left")
    mgd_cno["n_slips"] = mgd_cno["n_slips"].fillna(0)
    mgd_cno["rate"]    = mgd_cno["n_slips"] / (mgd_cno["n_epochs"] / 3600.0)

    n_rows = 2 if has_elev else 1
    fig, axes = plt.subplots(n_rows, 2,
                             figsize=(14, 5 * n_rows),
                             squeeze=False)

    # ── [0,0] slip rate vs C/N0 ───────────────────────────────────────
    bar_rate(axes[0, 0], mgd_cno, "cno_bin", cno_labels, ants)
    axes[0, 0].set_xlabel("C/N0 at last good epoch before slip (dBHz)")
    axes[0, 0].set_title("Slip rate vs C/N0\n(normalised by tracking exposure)")

    # ── [0,1] scatter (elevation vs C/N0) at slip time ────────────────
    ax_sc = axes[0, 1]
    for ant in ants:
        sub = slips_c[slips_c["antenna_mount"] == ant]
        if has_elev:
            sub = sub.dropna(subset=["elev_before"])
            ax_sc.scatter(sub["elev_before"], sub["cno_before"],
                          color=ant_color[ant], marker=ant_marker[ant],
                          s=15, alpha=0.4, label=ant)
            ax_sc.set_xlabel("Elevation at slip (°)")
        else:
            ax_sc.scatter(np.arange(len(sub)), sub["cno_before"],
                          color=ant_color[ant], marker=ant_marker[ant],
                          s=15, alpha=0.4, label=ant)
            ax_sc.set_xlabel("Slip index")
    ax_sc.set_ylabel("C/N0 at slip (dBHz)")
    ax_sc.set_title("(elevation, C/N0) at slip time\nupper-right = strong signal, high elevation = significant slip")
    ax_sc.axhline(35, color="grey", linestyle="--", linewidth=0.8)
    if has_elev:
        ax_sc.axvline(30, color="grey", linestyle="--", linewidth=0.8)
    ax_sc.legend(fontsize=8)
    ax_sc.grid(True, alpha=0.3)

    if has_elev:
        # ── elevation exposure and slip counts ────────────────────────
        df_e               = df.dropna(subset=["elev_deg"]).copy()
        df_e["el_bin"]     = pd.cut(df_e["elev_deg"],          el_edges,
                                    labels=el_labels, right=False)
        slips_e            = slips.dropna(subset=["elev_before"]).copy()
        slips_e["el_bin"]  = pd.cut(slips_e["elev_before"],    el_edges,
                                    labels=el_labels, right=False)
        exp_el  = (df_e.groupby(["antenna_mount", "el_bin"], observed=True)
                       .size().rename("n_epochs").reset_index())
        cnt_el  = (slips_e.groupby(["antenna_mount", "el_bin"], observed=True)
                          .size().rename("n_slips").reset_index())
        mgd_el  = exp_el.merge(cnt_el, on=["antenna_mount", "el_bin"], how="left")
        mgd_el["n_slips"] = mgd_el["n_slips"].fillna(0)
        mgd_el["rate"]    = mgd_el["n_slips"] / (mgd_el["n_epochs"] / 3600.0)

        # ── [1,0] slip rate vs elevation ──────────────────────────────
        bar_rate(axes[1, 0], mgd_el, "el_bin", el_labels, ants)
        axes[1, 0].set_xlabel("Elevation at last good epoch before slip (°)")
        axes[1, 0].set_title("Slip rate vs elevation\n(normalised by tracking exposure)")

        # ── [1,1] cumulative slip fraction vs elevation ───────────────
        ax_cum = axes[1, 1]
        for ant in ants:
            sub = (slips_e[slips_e["antenna_mount"] == ant]
                   .sort_values("elev_before"))
            if sub.empty:
                continue
            y_cum = np.arange(1, len(sub) + 1) / len(sub) * 100
            ax_cum.plot(sub["elev_before"].values, y_cum,
                        color=ant_color[ant], label=ant, linewidth=1.5)
        ax_cum.axvline(30, color="grey", linestyle="--",
                       linewidth=0.8, label="30° mask")
        ax_cum.set_xlabel("Elevation (°)")
        ax_cum.set_ylabel("Cumulative slip fraction (%)")
        ax_cum.set_title("Cumulative slip fraction vs elevation\n"
                         "steep rise at low elevation = expected; "
                         "high-elevation tail = antenna problem")
        ax_cum.legend(fontsize=8)
        ax_cum.grid(True, alpha=0.3)
        ax_cum.set_ylim(0, 105)

    fig.suptitle("Cycle slip quality analysis — slips normalised by tracking exposure\n"
                 "Excess slips at low C/N0 / low elevation = expected;  "
                 "excess at high C/N0 / high elevation = real antenna problem",
                 fontsize=10)
    fig.tight_layout()
    path = out_stem.parent / (out_stem.name + "_slip_quality.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot    → {path}")


# ── main ─────────────────────────────────────────────────────────────── #

def _load_toml(path: Path) -> dict:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]
    with open(path, "rb") as f:
        return tomllib.load(f)


def main():
    ap = argparse.ArgumentParser(
        description="CMC and cycle-slip analysis from a RAWX CSV"
    )
    ap.add_argument("--csv",  required=True, help="Input _rawx.csv file")
    ap.add_argument("--out",  required=True, help="Output filename stem")
    ap.add_argument("--snr",  default=None,
                    help="Companion SNR CSV for elevation/azimuth join")
    ap.add_argument("--receivers", default=None,
                    help="Receiver config (receivers.toml) to restrict "
                         "comparisons to the signal intersection")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_stem = Path(args.out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading RAWX  : {csv_path}")
    df = load(csv_path)
    print(f"  {len(df):,} rows  "
          f"{df['timestamp'].nunique()} epochs  "
          f"{df['signal_id'].nunique()} signals")

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

    if args.snr:
        print(f"Loading SNR   : {args.snr}")
        snr_df = load_snr(Path(args.snr))
        # Apply same signal intersection filter to SNR data
        if filter_note and "gnss_id" in snr_df.columns:
            snr_df = snr_df[snr_df["gnss_id"].isin(common)].copy()
        df = add_elevation(df, snr_df)

    df = add_cmc(df)
    n_cmc = df["cmc_m"].notna().sum()
    print(f"  {n_cmc:,} rows with valid CMC (cp_valid=1, half_cyc=1)")
    df = add_cmc_detrended(df)

    print("Detecting cycle slips …")
    slips = detect_cycle_slips(df)
    print(f"  {len(slips)} cycle slips detected")

    write_report(df, slips, out_stem)
    plot_cmc_by_signal(df, out_stem)
    plot_cmc_diff(df, out_stem, filter_note)
    plot_lock_duration(df, out_stem)
    plot_cmc_vs_elevation(df, out_stem)
    plot_cmc_skyplot(df, out_stem)
    plot_cycle_slip_timeline(slips, out_stem)
    plot_slip_quality(df, slips, out_stem)
    print("Done.")


if __name__ == "__main__":
    main()
