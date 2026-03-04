#!/usr/bin/env python3
"""
analyze_pps.py — PPS timing analysis from TICC and TIM-TP CSV files.

Usage:
    python scripts/analyze_pps.py \
        --ticc  data/foo_ticc.csv  \
        --timtp data/foo_timtp.csv \
        --out   data/foo           \
        [--rawx data/foo_rawx.csv]

Outputs:
    _pps_report.txt   — statistics and ADEV/TDEV summary
    _pps_diff.png     — raw vs qErr-corrected A−B time series
    _pps_adev.png     — ADEV(τ): raw and corrected
    _pps_tdev.png     — TDEV(τ): raw and corrected
    _pps_cmc_corr.png — rolling CMC noise vs PPS jitter (requires --rawx)

Alignment notes:
  - TIM-TP message N reports the predicted qErr for PPS edge N+1.
    So TICC pair i is corrected by TIM-TP[i-1].qerr_ps (shift by 1).
    The first TICC pair has no prior qErr and is dropped.
  - chA = TOP receiver, chB = BOT receiver.
  - Alignment by UTC wall-clock second (preferred): TICC host_timestamp
    floored to integer second S joins TIM-TP row at utc_s = S-1.
    This is unambiguous and requires no GPS offset search.
  - Fallback (old TICC CSVs without host_timestamp): GPS offset arithmetic
    with ±1 search, which has inherent ±1-epoch uncertainty.
    UTC wall-clock times come from the TOP TIM-TP timestamps.
"""

import argparse
import sys
from pathlib import Path

import allantools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Wavelength table reused for optional CMC correlation
_C = 299_792_458.0
_WAVELENGTH: dict[str, float] = {
    "GPS-L1CA":   _C / 1_575_420_000.0,
    "GAL-E1C":    _C / 1_575_420_000.0,
    "GAL-E1B":    _C / 1_575_420_000.0,
    "GPS-L5I":    _C / 1_176_450_000.0,
    "GPS-L5Q":    _C / 1_176_450_000.0,
    "GAL-E5aI":   _C / 1_176_450_000.0,
    "GAL-E5aQ":   _C / 1_176_450_000.0,
    "BDS-B2aI":   _C / 1_176_450_000.0,
    "BDS-B1I":    _C / 1_561_098_000.0,
    "BDS-B1C":    _C / 1_575_420_000.0,
    "GPS-L2CL":   _C / 1_227_600_000.0,
    "GPS-L2CM":   _C / 1_227_600_000.0,
}


# ── loading ──────────────────────────────────────────────────────────── #

def load_ticc(path: Path) -> pd.DataFrame:
    """
    Load TICC CSV and pair chA/chB edges by integer second.

    Asserts that no edge timestamp is within 100 ns of a second boundary —
    if both channels are > 100 ns clear, and the inter-channel delay is
    < 100 ns (as expected for our setup), they will always share the same
    integer second and the pivot is unambiguous.

    Returns DataFrame with columns:
        integer_sec, chA_ts, chB_ts, raw_diff_s
    Sorted by integer_sec; only seconds where both channels arrived.
    """
    _BOUNDARY_GUARD_S = 100e-9   # 100 ns minimum distance from integer boundary

    df = pd.read_csv(path)
    df["integer_sec"] = df["timestamp_s"].astype(int)

    # host_timestamp: UTC wall-clock captured immediately on serial receipt.
    # Enables unambiguous UTC-second join with TIM-TP (no GPS offset search).
    if "host_timestamp" in df.columns:
        host_ts = pd.to_datetime(df["host_timestamp"], utc=True)
        df["host_sec"] = (host_ts.astype("int64") // 1_000_000_000).astype(int)

    frac = df["timestamp_s"] - df["integer_sec"]
    bad  = (frac < _BOUNDARY_GUARD_S) | (frac > 1.0 - _BOUNDARY_GUARD_S)
    if bad.any():
        raise ValueError(
            f"TICC: {bad.sum()} edge(s) within {_BOUNDARY_GUARD_S*1e9:.0f} ns "
            f"of a second boundary — possible straddling artefact:\n"
            f"{df[bad][['timestamp_s', 'channel']].to_string()}"
        )

    piv = (
        df.pivot_table(index="integer_sec", columns="channel",
                       values="timestamp_s", aggfunc="first")
          .rename(columns={"chA": "chA_ts", "chB": "chB_ts"})
          .dropna()
          .reset_index()
          .sort_values("integer_sec")
          .reset_index(drop=True)
    )
    piv["raw_diff_s"] = piv["chA_ts"] - piv["chB_ts"]
    if "host_sec" in df.columns:
        hs_map = df.groupby("integer_sec")["host_sec"].first()
        piv["host_sec"] = piv["integer_sec"].map(hs_map)
    return piv


def load_timtp(path: Path) -> dict[str, pd.DataFrame]:
    """
    Load TIM-TP CSV.  Returns dict keyed by receiver ('TOP', 'BOT'),
    each a DataFrame sorted by timestamp with columns:
        timestamp, qerr_ps, tow_ms, tow_s, week
    tow_s is the integer GPS second (tow_ms // 1000).
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["tow_s"] = (df["tow_ms"] // 1000).astype(int)
    # utc_s: integer UTC second when message was logged (for UTC-second join).
    df["utc_s"] = (df["timestamp"].astype("int64") // 1_000_000_000).astype(int)
    return {
        rx: grp.sort_values("timestamp").reset_index(drop=True)
        for rx, grp in df.groupby("receiver")
    }


# ── GPS-second join ──────────────────────────────────────────────────── #

def _gps_join(ticc: pd.DataFrame,
              top_df: pd.DataFrame,
              bot_df: pd.DataFrame,
              gps_offset: int) -> pd.DataFrame:
    """
    Join TICC pairs with TIM-TP qErr by GPS second.

    TICC integer_sec + gps_offset = GPS second S.
    TIM-TP at tow_s = S−1 predicted the PPS edge at GPS second S,
    so it supplies the qErr that corrects the TICC pair at GPS second S.

    Returns a copy of ticc with added columns:
        gps_sec, qerr_top_ps, qerr_bot_ps, utc_time
    Rows where no matching TIM-TP entry exists are dropped.
    """
    df = ticc.copy()
    df["gps_sec"] = df["integer_sec"] + gps_offset

    top_q  = top_df.set_index("tow_s")["qerr_ps"]
    bot_q  = bot_df.set_index("tow_s")["qerr_ps"]
    top_ts = top_df.set_index("tow_s")["timestamp"]

    # tow_s = GPS_second − 1 is the TIM-TP that predicted this PPS edge
    corr_tow = df["gps_sec"] - 1
    df["qerr_top_ps"] = corr_tow.map(top_q)
    df["qerr_bot_ps"] = corr_tow.map(bot_q)
    df["utc_time"]    = corr_tow.map(top_ts)

    return df.dropna(subset=["qerr_top_ps", "qerr_bot_ps"]).reset_index(drop=True)


def _utc_join(ticc: pd.DataFrame,
              top_df: pd.DataFrame,
              bot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join TICC pairs with TIM-TP qErr by UTC wall-clock second.

    TICC host_sec = S  (integer UTC second when edge arrived at host).
    TIM-TP utc_s  = S−1 (when TIM-TP message was logged; it predicted PPS at S).

    No GPS offset arithmetic; no ±1 search uncertainty.
    Requires host_timestamp column in TICC CSV (logged since 2026-03-04).
    """
    df = ticc.copy()
    top_q  = top_df.set_index("utc_s")["qerr_ps"]
    bot_q  = bot_df.set_index("utc_s")["qerr_ps"]
    top_ts = top_df.set_index("utc_s")["timestamp"]

    corr_utc = df["host_sec"] - 1   # TIM-TP at S-1 predicts PPS edge at S
    df["qerr_top_ps"] = corr_utc.map(top_q)
    df["qerr_bot_ps"] = corr_utc.map(bot_q)
    df["utc_time"]    = corr_utc.map(top_ts)

    return df.dropna(subset=["qerr_top_ps", "qerr_bot_ps"]).reset_index(drop=True)


# ── qErr alignment validation ────────────────────────────────────────── #

def validate_alignment(ticc: pd.DataFrame,
                       timtp: dict[str, pd.DataFrame]) -> tuple:
    """
    Confirm qErr sign and GPS-second alignment.

    When TICC CSV contains host_timestamp (logged since 2026-03-04):
      Uses direct UTC-second join.  Only sign (+1/-1) is searched.
      Returns join_method="utc".

    Fallback (old data without host_timestamp):
      GPS offset arithmetic with delta search −1…+2.
      Expected: delta=0, sign=+1.
      Returns join_method="gps".

    Returns (raw_std_ns, results, naive_offset, join_method)
      results: list of (delta, sign, std_ns, n_pairs)
    """
    top = timtp.get("TOP", pd.DataFrame(columns=["qerr_ps", "tow_s"]))
    bot = timtp.get("BOT", pd.DataFrame(columns=["qerr_ps", "tow_s"]))
    raw_std_ns = float(ticc["raw_diff_s"].dropna().std() * 1e9)

    if top.empty or bot.empty:
        return raw_std_ns, [], 0, "gps"

    # UTC join: unambiguous when host_timestamp column is present.
    if "host_sec" in ticc.columns and "utc_s" in top.columns:
        joined = _utc_join(ticc, top, bot)
        n = len(joined)
        results = []
        for sign in (+1, -1):
            corr = ((joined["chA_ts"] + sign * joined["qerr_top_ps"] * 1e-12) -
                    (joined["chB_ts"] + sign * joined["qerr_bot_ps"] * 1e-12))
            std_ns = float(corr.std() * 1e9) if n > 1 else np.inf
            results.append((0, sign, std_ns, n))
        return raw_std_ns, results, 0, "utc"

    # GPS offset fallback: search ±1 around naive estimate.
    naive = int(top["tow_s"].iloc[0]) - int(ticc["integer_sec"].iloc[0])
    results = []
    for delta in (-1, 0, +1, +2):
        joined = _gps_join(ticc, top, bot, naive + delta)
        n = len(joined)
        for sign in (+1, -1):
            corr = ((joined["chA_ts"] + sign * joined["qerr_top_ps"] * 1e-12) -
                    (joined["chB_ts"] + sign * joined["qerr_bot_ps"] * 1e-12))
            std_ns = float(corr.std() * 1e9) if n > 1 else np.inf
            results.append((delta, sign, std_ns, n))
    return raw_std_ns, results, naive, "gps"


# ── qErr correction ──────────────────────────────────────────────────── #

def apply_qerr(ticc: pd.DataFrame,
               timtp: dict[str, pd.DataFrame],
               gps_offset: int,
               sign: int = +1) -> pd.DataFrame:
    """
    Correct TICC timestamps with qErr from TIM-TP.

    Uses UTC-second join when host_timestamp is available (preferred);
    falls back to GPS-second arithmetic with the provided gps_offset.

    Sign convention: corrected = measured + sign * qerr_ps * 1e-12.
    sign=+1: positive qErr means the pulse fired early; adding qerr moves
    the timestamp to the true second boundary.

    Adds columns:
        qerr_top_ps, qerr_bot_ps  — corrections applied (ps)
        corr_diff_s                — qErr-corrected A−B (s)
        utc_time                   — wall-clock UTC from TOP TIM-TP
    """
    top = timtp.get("TOP", pd.DataFrame(columns=["qerr_ps", "tow_s", "timestamp"]))
    bot = timtp.get("BOT", pd.DataFrame(columns=["qerr_ps", "tow_s"]))

    use_utc = ("host_sec" in ticc.columns and not top.empty
               and "utc_s" in top.columns)
    if use_utc:
        df = _utc_join(ticc, top, bot)
    else:
        df = _gps_join(ticc, top, bot, gps_offset)

    df["corr_diff_s"] = (
        (df["chA_ts"] + sign * df["qerr_top_ps"] * 1e-12) -
        (df["chB_ts"] + sign * df["qerr_bot_ps"] * 1e-12)
    )
    return df


# ── ADEV / TDEV ──────────────────────────────────────────────────────── #

def compute_stability(phase_s: np.ndarray) -> dict:
    """
    Compute ADEV and TDEV from a 1-Hz phase time series (seconds).
    Returns dict with keys: taus_adev, adev, taus_tdev, tdev.
    Returns empty dict if the series is too short.
    """
    phase_s = phase_s[~np.isnan(phase_s)]
    if len(phase_s) < 8:
        return {}
    taus_a, adev, _, _ = allantools.adev(
        phase_s, rate=1.0, data_type="phase", taus="decade")
    taus_t, tdev, _, _ = allantools.tdev(
        phase_s, rate=1.0, data_type="phase", taus="decade")
    return {"taus_adev": taus_a, "adev": adev,
            "taus_tdev": taus_t, "tdev": tdev}


# ── report ───────────────────────────────────────────────────────────── #

def write_report(df: pd.DataFrame,
                 raw_stab: dict, corr_stab: dict,
                 alignment: tuple, out_stem: Path) -> None:
    lines = []
    a = lines.append

    raw_ns  = df["raw_diff_s"]  * 1e9
    corr_ns = df["corr_diff_s"] * 1e9
    dur_h   = (df["utc_time"].max() - df["utc_time"].min()).total_seconds() / 3600

    a("=" * 62)
    a("  testAnt PPS / TICC report")
    a("=" * 62)
    a(f"  Start    : {df['utc_time'].min()}")
    a(f"  End      : {df['utc_time'].max()}")
    a(f"  Duration : {dur_h:.2f} h")
    a(f"  Pairs    : {len(df)}  (chA=TOP, chB=BOT)")
    a("")

    raw_std_ns, align_results, naive_offset, join_method = alignment
    best = min(align_results, key=lambda x: x[2])
    if join_method == "utc":
        a("── qErr alignment (UTC host_timestamp join) ─────────────────────")
        a(f"  Raw std  : {raw_std_ns:.3f} ns")
        a(f"  {'Sign':>5s}  {'N pairs':>7s}  {'std (ns)':>10s}")
        for delta, sign, std_ns, n_pairs in align_results:
            tag = "  ← best" if (delta == best[0] and sign == best[1]) else ""
            a(f"  {sign:>+5d}  {n_pairs:>7d}  {std_ns:>10.3f}{tag}")
    else:
        a("── qErr alignment (GPS offset search; no host_timestamp) ─────────")
        a(f"  Raw std  : {raw_std_ns:.3f} ns")
        a(f"  {'GPS Δ':>6s}  {'Sign':>5s}  {'N pairs':>7s}  {'std (ns)':>10s}")
        for delta, sign, std_ns, n_pairs in align_results:
            tags = []
            if delta == 0 and sign == +1:
                tags.append("← naive")
            if delta == best[0] and sign == best[1]:
                tags.append("← best")
            tag = "  " + " ".join(tags) if tags else ""
            a(f"  {delta:>+6d}  {sign:>+5d}  {n_pairs:>7d}  {std_ns:>10.3f}{tag}")
        if best[0] != 0 or best[1] != +1:
            a(f"  *** GPS offset delta={best[0]:+d} sign={best[1]:+d} "
              f"(GPS offset={naive_offset + best[0]}) ***")
    a("")

    a("── Raw A−B difference (chA − chB) ───────────────────────")
    a(f"  Mean : {raw_ns.mean():+.3f} ns")
    a(f"  Std  : {raw_ns.std():.3f} ns")
    a("")

    a("── qErr-corrected A−B difference ────────────────────────")
    a(f"  Mean : {corr_ns.mean():+.3f} ns")
    a(f"  Std  : {corr_ns.std():.3f} ns")
    a("")

    key_taus = [1, 10, 100, 1000]

    for label, stab in [("Raw", raw_stab), ("Corrected", corr_stab)]:
        if not stab:
            continue
        a(f"── ADEV(τ) — {label} ─────────────────────────────────────")
        a(f"  {'τ (s)':>8s}  {'ADEV (ns)':>12s}")
        for tau in key_taus:
            idx = np.searchsorted(stab["taus_adev"], tau)
            if idx < len(stab["adev"]):
                a(f"  {tau:>8d}  {stab['adev'][idx]*1e9:>12.3f}")
        a("")

    for label, stab in [("Raw", raw_stab), ("Corrected", corr_stab)]:
        if not stab:
            continue
        a(f"── TDEV(τ) — {label} ─────────────────────────────────────")
        a(f"  {'τ (s)':>8s}  {'TDEV (ns)':>12s}")
        for tau in key_taus:
            idx = np.searchsorted(stab["taus_tdev"], tau)
            if idx < len(stab["tdev"]):
                a(f"  {tau:>8d}  {stab['tdev'][idx]*1e9:>12.3f}")
        a("")

    a("=" * 62)

    path = out_stem.parent / (out_stem.name + "_pps_report.txt")
    path.write_text("\n".join(lines) + "\n")
    print(f"Report  → {path}")


# ── plots ────────────────────────────────────────────────────────────── #

def plot_diff(df: pd.DataFrame, out_stem: Path) -> None:
    """Raw vs corrected A−B time series."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    t = df["utc_time"]
    raw_ns  = df["raw_diff_s"]  * 1e9
    corr_ns = df["corr_diff_s"] * 1e9

    axes[0].plot(t, raw_ns, color="steelblue", linewidth=0.7, label="raw")
    axes[0].axhline(raw_ns.mean(), color="navy", linewidth=0.8,
                    linestyle="--", label=f"mean {raw_ns.mean():+.1f} ns")
    axes[0].set_ylabel("A−B (ns)")
    axes[0].set_title(f"Raw  (std = {raw_ns.std():.2f} ns)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, corr_ns, color="tomato", linewidth=0.7, label="qErr-corrected")
    axes[1].axhline(corr_ns.mean(), color="darkred", linewidth=0.8,
                    linestyle="--", label=f"mean {corr_ns.mean():+.1f} ns")
    axes[1].set_ylabel("A−B (ns)")
    axes[1].set_title(f"qErr-corrected  (std = {corr_ns.std():.2f} ns)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    fig.suptitle("PPS A−B time difference  (chA=TOP, chB=BOT)\n"
                 "mean offset = cable delay + receiver bias",
                 fontsize=11)
    fig.tight_layout()
    path = out_stem.parent / (out_stem.name + "_pps_diff.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot    → {path}")


def _stability_plot(raw_stab: dict, corr_stab: dict,
                    key: str, ylabel: str, title: str,
                    out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    if raw_stab:
        ax.loglog(raw_stab[f"taus_{key}"], raw_stab[key] * 1e9,
                  color="steelblue", linewidth=1.2, label="raw")
    if corr_stab:
        ax.loglog(corr_stab[f"taus_{key}"], corr_stab[key] * 1e9,
                  color="tomato", linewidth=1.2, label="qErr-corrected")
    ax.set_xlabel("τ (s)")
    ax.set_ylabel(f"{ylabel} (ns)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Plot    → {out_path}")


def plot_adev(raw_stab: dict, corr_stab: dict, out_stem: Path) -> None:
    _stability_plot(
        raw_stab, corr_stab, "adev", "ADEV",
        "Allan deviation — TOP vs BOT PPS difference\n"
        "(differential; common-mode ionosphere/clock cancelled)",
        out_stem.parent / (out_stem.name + "_pps_adev.png"),
    )


def plot_tdev(raw_stab: dict, corr_stab: dict, out_stem: Path) -> None:
    _stability_plot(
        raw_stab, corr_stab, "tdev", "TDEV",
        "Time deviation — TOP vs BOT PPS difference",
        out_stem.parent / (out_stem.name + "_pps_tdev.png"),
    )


# ── optional CMC correlation ─────────────────────────────────────────── #

def plot_cmc_correlation(df: pd.DataFrame, rawx_path: Path,
                         out_stem: Path, window_s: int = 60) -> None:
    """
    Compare rolling CMC noise (per receiver) with rolling PPS jitter.

    Hypothesis: epochs with higher multipath (higher CMC std dev) should
    show more PPS timing variability.  The Kalman filter in the receiver
    low-passes the multipath-to-PPS transfer, so correlation is expected
    at timescales of ~100 s or longer.
    """
    rawx = pd.read_csv(rawx_path, parse_dates=["timestamp"])
    rawx["timestamp"] = pd.to_datetime(rawx["timestamp"], utc=True)
    if "locktime_ms" in rawx.columns:
        rawx = rawx.rename(columns={"locktime_ms": "lock_duration_ms"})

    # CMC (code-minus-carrier)
    wl = rawx["signal_id"].map(_WAVELENGTH)
    cp_ok = (rawx["cp_valid"] == 1) & (rawx["half_cyc"] == 1)
    rawx["cmc_m"] = np.where(
        cp_ok & wl.notna(),
        rawx["pseudorange_m"] - wl * rawx["carrier_phase_cy"],
        np.nan,
    )
    per_sv_mean = (rawx.groupby(["receiver", "signal_id", "sv_id"])["cmc_m"]
                       .transform("mean"))
    rawx["cmc_det"] = rawx["cmc_m"] - per_sv_mean

    # Floor to integer second, then compute per-epoch std across all SVs
    rawx["ts_s"] = rawx["timestamp"].dt.floor("s")
    epoch_cmc = (rawx.dropna(subset=["cmc_det"])
                     .groupby(["ts_s", "receiver"])["cmc_det"]
                     .std()
                     .reset_index()
                     .rename(columns={"cmc_det": "cmc_std_m"}))

    top_cmc = (epoch_cmc[epoch_cmc["receiver"] == "TOP"]
               .set_index("ts_s")["cmc_std_m"]
               .sort_index())
    bot_cmc = (epoch_cmc[epoch_cmc["receiver"] == "BOT"]
               .set_index("ts_s")["cmc_std_m"]
               .sort_index())

    # Rolling std of corrected PPS diff, indexed by utc_time
    pps = df.set_index("utc_time")["corr_diff_s"].sort_index() * 1e9  # ns
    pps_roll = pps.rolling(window=window_s, center=True, min_periods=window_s // 2).std()

    # Rolling mean CMC (smoothed) per receiver
    top_roll = top_cmc.rolling(window=window_s, center=True, min_periods=window_s // 2).mean()
    bot_roll = bot_cmc.rolling(window=window_s, center=True, min_periods=window_s // 2).mean()

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Top panel: CMC std per receiver
    ax = axes[0]
    ax.plot(top_roll.index, top_roll.values, color="steelblue",
            linewidth=0.8, label="TOP CMC noise")
    ax.plot(bot_roll.index, bot_roll.values, color="tomato",
            linewidth=0.8, label="BOT CMC noise")
    ax.set_ylabel("CMC std (m)")
    ax.set_title(f"Rolling {window_s}s CMC noise (detrended, all signals combined)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom panel: PPS jitter
    ax = axes[1]
    ax.plot(pps_roll.index, pps_roll.values, color="seagreen",
            linewidth=0.8, label="PPS jitter (A−B std)")
    ax.set_ylabel("PPS std (ns)")
    ax.set_title(f"Rolling {window_s}s PPS jitter (qErr-corrected A−B)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    fig.suptitle("CMC multipath noise vs PPS timing jitter\n"
                 "(correlation would indicate multipath influence on PPS)",
                 fontsize=11)
    fig.tight_layout()
    path = out_stem.parent / (out_stem.name + "_pps_cmc_corr.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot    → {path}")

    # Quantify: Pearson r on the overlapping time window
    common = pps_roll.index.intersection(top_roll.index)
    if len(common) > 10:
        r_top = np.corrcoef(pps_roll.reindex(common).dropna(),
                            top_roll.reindex(common).dropna())[0, 1]
        r_bot = np.corrcoef(pps_roll.reindex(common).dropna(),
                            bot_roll.reindex(common).dropna())[0, 1]
        print(f"  Pearson r(PPS jitter, TOP CMC noise) = {r_top:+.3f}")
        print(f"  Pearson r(PPS jitter, BOT CMC noise) = {r_bot:+.3f}")


# ── main ─────────────────────────────────────────────────────────────── #

def main():
    ap = argparse.ArgumentParser(
        description="PPS timing analysis from TICC and TIM-TP CSV files"
    )
    ap.add_argument("--ticc",  required=True, help="Input _ticc.csv file")
    ap.add_argument("--timtp", required=True, help="Input _timtp.csv file")
    ap.add_argument("--out",   required=True, help="Output filename stem")
    ap.add_argument("--rawx",  default=None,
                    help="Optional _rawx.csv for CMC correlation plot")
    args = ap.parse_args()

    out_stem = Path(args.out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading TICC  : {args.ticc}")
    ticc = load_ticc(Path(args.ticc))
    print(f"  {len(ticc)} complete pairs (chA + chB)")

    print(f"Loading TIM-TP: {args.timtp}")
    timtp = load_timtp(Path(args.timtp))
    for rx, grp in timtp.items():
        print(f"  {rx}: {len(grp)} rows  "
              f"qerr range [{grp['qerr_ps'].min():+d}, {grp['qerr_ps'].max():+d}] ps")

    print("Validating qErr alignment …")
    alignment = validate_alignment(ticc, timtp)
    raw_std_ns, align_results, naive_offset, join_method = alignment
    best = min(align_results, key=lambda x: x[2]) if align_results else (0, 1, raw_std_ns, 0)
    best_delta, best_sign, best_std, best_n = best
    best_gps_offset = naive_offset + best_delta
    print(f"  Method   : {join_method.upper()} join")
    print(f"  Raw std  : {raw_std_ns:.3f} ns")
    if join_method == "utc":
        print(f"  {'Sign':>5s}  {'N pairs':>7s}  {'std (ns)':>10s}")
        for delta, sign, std_ns, n_pairs in align_results:
            tag = "  ← best" if (delta == best_delta and sign == best_sign) else ""
            print(f"  {sign:>+5d}  {n_pairs:>7d}  {std_ns:>10.3f}{tag}")
    else:
        print(f"  {'GPS Δ':>6s}  {'Sign':>5s}  {'N pairs':>7s}  {'std (ns)':>10s}")
        for delta, sign, std_ns, n_pairs in align_results:
            tags = []
            if delta == 0 and sign == +1:
                tags.append("← naive")
            if delta == best_delta and sign == best_sign:
                tags.append("← best")
            tag = "  " + " ".join(tags) if tags else ""
            print(f"  {delta:>+6d}  {sign:>+5d}  {n_pairs:>7d}  {std_ns:>10.3f}{tag}")
        if best_delta != 0 or best_sign != +1:
            print(f"  *** ALIGNMENT: GPS offset delta={best_delta:+d}, sign={best_sign:+d} "
                  f"(using GPS offset={best_gps_offset}) ***", file=sys.stderr)

    print("Applying qErr correction …")
    df = apply_qerr(ticc, timtp, gps_offset=best_gps_offset, sign=best_sign)
    print(f"  {len(df)} pairs after correction (GPS offset={best_gps_offset}, sign={best_sign:+d})")
    raw_ns  = df["raw_diff_s"]  * 1e9
    corr_ns = df["corr_diff_s"] * 1e9
    print(f"  Raw  : mean={raw_ns.mean():+.2f} ns  std={raw_ns.std():.2f} ns")
    print(f"  Corr : mean={corr_ns.mean():+.2f} ns  std={corr_ns.std():.2f} ns")

    print("Computing ADEV/TDEV …")
    raw_stab  = compute_stability(df["raw_diff_s"].values)
    corr_stab = compute_stability(df["corr_diff_s"].values)

    write_report(df, raw_stab, corr_stab, alignment, out_stem)
    plot_diff(df, out_stem)
    plot_adev(raw_stab, corr_stab, out_stem)
    plot_tdev(raw_stab, corr_stab, out_stem)

    if args.rawx:
        print(f"Loading RAWX  : {args.rawx}")
        plot_cmc_correlation(df, Path(args.rawx), out_stem)

    print("Done.")


if __name__ == "__main__":
    main()
