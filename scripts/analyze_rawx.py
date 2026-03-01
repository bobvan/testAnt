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
            slips.append(s)

    return pd.DataFrame(slips)


# ── report ───────────────────────────────────────────────────────────── #

def write_report(df: pd.DataFrame, slips: pd.DataFrame,
                 out_stem: Path) -> None:
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

    cmc_ok = df.dropna(subset=["cmc_detrended_m"])
    a("── Detrended CMC std dev by signal & receiver (noise floor) ────")
    a(f"  {'Signal':16s}  {'Receiver':8s}  {'N_sv':>5s}  {'N_obs':>6s}  {'std_m':>8s}")
    sv_counts = (cmc_ok.groupby(["receiver", "signal_id"])["sv_id"]
                       .nunique().rename("n_sv").reset_index())
    stats = (cmc_ok.groupby(["signal_id", "receiver"])["cmc_detrended_m"]
               .agg(n_obs="count", std="std").reset_index()
               .sort_values(["signal_id", "receiver"]))
    stats = stats.merge(sv_counts, on=["receiver", "signal_id"])
    for _, row in stats.iterrows():
        a(f"  {row['signal_id']:16s}  {row['receiver']:8s}  "
          f"{int(row['n_sv']):>5d}  {int(row['n_obs']):>6d}  "
          f"{row['std']:>8.3f} m")
    a("")

    a("── Lock duration stats by signal & receiver (ms) ───────────────")
    a(f"  {'Signal':16s}  {'Receiver':8s}  {'median_ms':>10s}  {'p95_ms':>8s}")
    lt_stats = (df.groupby(["signal_id", "receiver"])["lock_duration_ms"]
                  .agg(median="median", p95=lambda x: x.quantile(0.95))
                  .reset_index()
                  .sort_values(["signal_id", "receiver"]))
    for _, row in lt_stats.iterrows():
        a(f"  {row['signal_id']:16s}  {row['receiver']:8s}  "
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

        a(f"  {'Signal':16s}  {'Receiver':8s}  {'Slips':>6s}  "
          f"{'Track-h':>7s}  {'Rate/24h':>9s}")
        for _, row in summary.sort_values(["signal_id", "receiver"]).iterrows():
            a(f"  {row['signal_id']:16s}  {row['receiver']:8s}  "
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
        a("  10 largest slips:")
        a(f"  {'Timestamp':26s}  {'Rx':5s}  {'Sig':14s}  SV  "
          f"{'Before ms':>10s}  {'After ms':>9s}  {'Drop ms':>8s}")
        for _, row in slips.nlargest(10, "drop_ms").iterrows():
            a(f"  {str(row['timestamp']):26s}  "
              f"{row['receiver']:5s}  {row['signal_id']:14s}  "
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
            ax.plot(g["timestamp"], g["cmc_detrended_m"], label=rx,
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


def plot_cmc_diff(df: pd.DataFrame, out_stem: Path) -> None:
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
        ax.set_ylabel("ΔCMC (m)")
        ax.set_title(f"{sig}  —  {rx_a} − {rx_b}  "
                     f"(std={epoch_diff.std():.3f} m, median {n_sv} SVs/epoch)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    fig.suptitle(f"Detrended CMC difference ({rx_a} − {rx_b}) by signal\n"
                 "per-SV matched; residual = receiver noise + differential multipath",
                 fontsize=11, y=1.002)
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
                    label=rx, color=color,
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

    receivers = sorted(cmc_ok["receiver"].unique())
    vmax = float(cmc_ok["cmc_detrended_m"].abs().quantile(0.95))

    fig, axes = plt.subplots(1, len(receivers),
                             figsize=(7 * len(receivers), 7),
                             subplot_kw={"projection": "polar"})
    if len(receivers) == 1:
        axes = [axes]

    for ax, rx in zip(axes, receivers):
        sub   = cmc_ok[cmc_ok["receiver"] == rx]
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
        ax.set_title(rx, fontsize=12, pad=15)
        plt.colorbar(sc, ax=ax, label="|CMC| (m)", fraction=0.046, pad=0.04)

    fig.suptitle("CMC multipath skyplot — |detrended CMC| by direction\n"
                 "(zenith at centre, N up, clockwise; bright = high multipath)",
                 fontsize=11)
    fig.tight_layout()
    path = out_stem.parent / (out_stem.name + "_cmc_skyplot.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot    → {path}")


# ── main ─────────────────────────────────────────────────────────────── #

def main():
    ap = argparse.ArgumentParser(
        description="CMC and cycle-slip analysis from a RAWX CSV"
    )
    ap.add_argument("--csv",  required=True, help="Input _rawx.csv file")
    ap.add_argument("--out",  required=True, help="Output filename stem")
    ap.add_argument("--snr",  default=None,
                    help="Companion SNR CSV for elevation/azimuth join")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_stem = Path(args.out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading RAWX  : {csv_path}")
    df = load(csv_path)
    print(f"  {len(df):,} rows  "
          f"{df['timestamp'].nunique()} epochs  "
          f"{df['signal_id'].nunique()} signals")

    if args.snr:
        print(f"Loading SNR   : {args.snr}")
        snr_df = load_snr(Path(args.snr))
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
    plot_cmc_diff(df, out_stem)
    plot_lock_duration(df, out_stem)
    plot_cmc_vs_elevation(df, out_stem)
    plot_cmc_skyplot(df, out_stem)
    print("Done.")


if __name__ == "__main__":
    main()
