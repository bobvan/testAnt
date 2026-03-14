#!/usr/bin/env python3
"""
report_card.py — One-page PDF antenna evaluation report card.

Usage:
    python scripts/report_card.py \
        --snr   data/choke_24h/choke_24h_snr_1min.csv \
        --rawx  data/choke_24h/choke_24h_rawx_1min.csv \
        --ticc  data/choke_24h/choke_24h_ticc.csv \
        --timtp data/choke_24h/choke_24h_timtp.csv \
        --sky   data/choke_24h/choke_24h_sky.csv \
        --slips data/choke_24h/choke_24h_slips.csv \
        --rx BOT \
        --antenna "Choke ring" --mount "Desk, indoors" \
        --receiver "u-blox ZED-F9T" \
        --out data/choke_report_card.pdf

Minimum test duration for a full (non-provisional) evaluation: 24 hours.
Shorter runs are marked PROVISIONAL with a completion percentage.
"""

import argparse
import sys
from pathlib import Path

import allantools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from report_plots import (
    VERSION, MINIMUM_HOURS, CMAP, METRICS, FIXED_Y, FIXED_X_HOURS,
    SERIES_KEYS, SERIES_YLABELS, CNO_VMIN, CNO_VMAX, SLIP_PCT_MAX,
    fraction, fmt_val, simple_formatter, apply_fixed_y,
    compute_lock_loss, plot_sparkline,
    polar_cno_heatmap, setup_polar_ax,
)


# ── Data loading ───────────────────────────────────────────────────── #

def load_data(snr_csv, rawx_csv, ticc_csv, timtp_csv,
              sky_csv=None, slips_csv=None, rx="BOT",
              antenna="Antenna", mount="", receiver="", notes=""):
    """Load pre-aggregated summaries + raw TICC/TIM-TP."""

    # ── SNR ────────────────────────────────────────────────── #
    snr = pd.read_csv(snr_csv, parse_dates=["minute"])
    snr["minute"] = pd.to_datetime(snr["minute"], utc=True)
    snr_rx = snr[snr["receiver"] == rx].sort_values("minute").reset_index(drop=True)
    t0 = snr_rx["minute"].iloc[0]
    snr_rx["hours"] = (snr_rx["minute"] - t0).dt.total_seconds() / 3600
    total_hours = (snr_rx["minute"].iloc[-1] - t0
                   + pd.Timedelta(minutes=1)).total_seconds() / 3600

    # ── RAWX / CMC ─────────────────────────────────────────── #
    rawx = pd.read_csv(rawx_csv, parse_dates=["minute"])
    rawx["minute"] = pd.to_datetime(rawx["minute"], utc=True)
    rawx_rx = rawx[rawx["receiver"] == rx].sort_values("minute").reset_index(drop=True)
    rawx_rx["hours"] = (rawx_rx["minute"] - t0).dt.total_seconds() / 3600

    # ── TICC + TIM-TP → ADEV ──────────────────────────────── #
    ticc = pd.read_csv(ticc_csv)
    has_host_ts = "host_timestamp" in ticc.columns

    if has_host_ts:
        # New format: pair by UTC second
        ticc["host_timestamp"] = pd.to_datetime(ticc["host_timestamp"], utc=True)
        ticc["utc_s"] = ticc["host_timestamp"].dt.floor("s")
        cha = ticc[ticc["channel"] == "chA"].groupby("utc_s").first().reset_index()
        chb = ticc[ticc["channel"] == "chB"].groupby("utc_s").first().reset_index()
        pairs = cha.merge(chb, on="utc_s", suffixes=("_a", "_b"))
        pairs["diff_ps"] = (pairs["ref_sec_a"].astype(np.int64) * 1_000_000_000_000
                            + pairs["ref_ps_a"].astype(np.int64)
                            - pairs["ref_sec_b"].astype(np.int64) * 1_000_000_000_000
                            - pairs["ref_ps_b"].astype(np.int64))
    else:
        # Old format: pair consecutive chA/chB rows by sequence
        ticc["ts_ps"] = (ticc["timestamp_s"] * 1e12).round().astype(np.int64)
        cha = ticc[ticc["channel"] == "chA"].reset_index(drop=True)
        chb = ticc[ticc["channel"] == "chB"].reset_index(drop=True)
        n = min(len(cha), len(chb))
        cha, chb = cha.iloc[:n], chb.iloc[:n]
        pairs = pd.DataFrame({
            "diff_ps": cha["ts_ps"].values - chb["ts_ps"].values,
        })
        # Use TIM-TP timestamps for time axis
        timtp_tmp = pd.read_csv(timtp_csv, parse_dates=["timestamp"])
        timtp_top = timtp_tmp[timtp_tmp["receiver"] == "TOP"].sort_values("timestamp")
        if len(timtp_top) >= n:
            pairs["utc_s"] = timtp_top["timestamp"].iloc[:n].dt.floor("s").values

    pairs["diff_ns"] = pairs["diff_ps"] / 1000.0

    # TIM-TP qErr correction (if utc_s available)
    timtp = pd.read_csv(timtp_csv, parse_dates=["timestamp"])
    timtp["utc_s"] = timtp["timestamp"].dt.floor("s")

    if "utc_s" in pairs.columns:
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
        # No UTC alignment — use raw diff
        pairs["corr_ns"] = pairs["diff_ns"]

    # Outlier rejection
    med = pairs["corr_ns"].median()
    mad = (pairs["corr_ns"] - med).abs().median() * 1.4826
    pairs = pairs[(pairs["corr_ns"] - med).abs() < max(mad * 5, 50)].reset_index(drop=True)

    if "utc_s" in pairs.columns:
        pairs["utc_s"] = pd.to_datetime(pairs["utc_s"], utc=True)
        pairs["hours"] = (pairs["utc_s"] - t0).dt.total_seconds() / 3600
    else:
        # Fallback: evenly spaced by index
        pairs["hours"] = np.arange(len(pairs)) / 3600
    freq_ns = pairs["corr_ns"].values - pairs["corr_ns"].mean()

    # Per-window ADEV
    pairs["window"] = (pairs["hours"] * 6).astype(int)
    adev_series = []
    for _, g in pairs.groupby("window"):
        if len(g) < 60:
            continue
        y = g["corr_ns"].values - g["corr_ns"].mean()
        _, ad, _, _ = allantools.adev(y, rate=1.0, data_type="freq", taus=[1])
        if len(ad):
            adev_series.append((g["hours"].mean(), ad[0]))
    adev_t = np.array([a[0] for a in adev_series])
    adev_v = np.array([a[1] for a in adev_series])

    _, adev_overall, _, _ = allantools.adev(freq_ns, rate=1.0,
                                            data_type="freq", taus=[1])

    # ── Carrier lock loss ─────────────────────────────────── #
    slip_elev = np.array([])
    slip_azim = np.array([])
    slip_cno = np.array([])

    # Load sky data (needed for both lock loss and polar plot)
    sky_data = None
    if sky_csv:
        sky = pd.read_csv(sky_csv, parse_dates=["minute"])
        sky["minute"] = pd.to_datetime(sky["minute"], utc=True)
        sky_data = sky[sky["receiver"] == rx].copy()

    slips_hours = np.array([])
    sky_hours = np.array([])

    if sky_data is not None:
        sky_hours = (sky_data["minute"] - t0).dt.total_seconds().values / 3600

    if slips_csv:
        sl = pd.read_csv(slips_csv, parse_dates=["timestamp"])
        sl["timestamp"] = pd.to_datetime(sl["timestamp"], utc=True)
        sl_rx = sl[sl["receiver"] == rx].copy()
        slips_hours = (sl_rx["timestamp"] - t0).dt.total_seconds().values / 3600

        # Join slips with sky positions for polar plot
        if sky_data is not None:
            sl_rx["minute"] = sl_rx["timestamp"].dt.floor("min")
            merged = sl_rx.merge(sky_data, on=["minute", "sv_id"], how="left",
                                 suffixes=("", "_sky"))
            merged = merged.dropna(subset=["elev_deg", "azim_deg"])
            slip_elev = merged["elev_deg"].values
            slip_azim = merged["azim_deg"].values
            slip_cno = merged["cno_dBHz"].values if "cno_dBHz" in merged else np.array([])

    t_centers, loss_pct, overall_pct = compute_lock_loss(
        slips_hours, sky_hours, total_hours)

    return {
        "antenna": antenna,
        "mount": mount,
        "receiver": receiver,
        "notes": notes,
        "start": t0.strftime("%Y-%m-%d %H:%M UTC"),
        "hours": total_hours,
        "summaries": {
            "cno_mean": float(snr_rx["mean_cno"].mean()),
            "cmc_std": float(rawx_rx["cmc_std_m"].median()),
            "adev_1s": float(adev_overall[0]) if len(adev_overall) else 1.0,
            "lock_loss_pct": overall_pct,
            "sat_count": float(snr_rx["sat_count"].mean()),
        },
        "series": {
            "cno":      (snr_rx["hours"].values, snr_rx["mean_cno"].values),
            "cmc":      (rawx_rx["hours"].values, rawx_rx["cmc_std_m"].values),
            "adev":     (adev_t, adev_v),
            "lockloss": (t_centers, loss_pct),
            "satcount": (snr_rx["hours"].values, snr_rx["sat_count"].values.astype(float)),
        },
        "sky": sky_data,
        "slip_sky": (slip_elev, slip_azim, slip_cno),
    }


# ── Report rendering ──────────────────────────────────────────────── #

def render_report(data, pdf_path):
    hours = data["hours"]
    is_provisional = hours < MINIMUM_HOURS
    completion_pct = min(100, hours / MINIMUM_HOURS * 100)
    s = data["summaries"]

    metric_values = [s["cno_mean"], s["cmc_std"], s["adev_1s"],
                     s["lock_loss_pct"], s["sat_count"]]

    fig = plt.figure(figsize=(8.5, 11))

    # ── Header ─────────────────────────────────────────────── #
    y = 0.97
    fig.text(0.05, y, data["antenna"],
             fontsize=16, fontweight="bold", va="top")

    # Equipment block
    info_parts = []
    if data["mount"]:
        info_parts.append(f"Mount: {data['mount']}")
    if data["receiver"]:
        info_parts.append(f"Receiver: {data['receiver']}")
    info_parts.append("TICC: TAPR (60 ps)")
    fig.text(0.05, y - 0.030, "    ".join(info_parts),
             fontsize=8, va="top", color="#555555")

    date_line = f"Start: {data['start']}    Duration: {hours:.1f} h"
    if data["notes"]:
        date_line += f"    Note: {data['notes']}"
    fig.text(0.05, y - 0.048, date_line,
             fontsize=8, va="top", color="#555555")

    # Badge
    if is_provisional:
        bx, by = 0.72, y - 0.008
        fig.patches.append(mpatches.FancyBboxPatch(
            (bx - 0.005, by - 0.030), 0.27, 0.036,
            boxstyle="round,pad=0.005",
            facecolor="#FFF3CD", edgecolor="#FFC107", linewidth=1.5,
            transform=fig.transFigure, zorder=10))
        fig.text(bx + 0.005, by - 0.012,
                 f"Provisional Result, Duration {hours:.0f} h "
                 f"({completion_pct:.0f}%) of full {MINIMUM_HOURS} h test",
                 fontsize=7, fontweight="bold", color="#856404", va="center",
                 zorder=11)
    else:
        fig.text(0.75, y - 0.018, "FULL EVALUATION",
                 fontsize=9, fontweight="bold", color="#155724",
                 bbox=dict(boxstyle="round,pad=0.3",
                           fc="#D4EDDA", ec="#28A745", lw=1.5))

    # Divider
    div_y = y - 0.072
    fig.add_artist(plt.Line2D([0.05, 0.95], [div_y, div_y],
                              color="#cccccc", lw=0.8,
                              transform=fig.transFigure, clip_on=False))

    # ── Metric rows ────────────────────────────────────────── #
    row_h = 0.105
    top = div_y - 0.045
    g_left, g_width = 0.05, 0.30
    sp_left, sp_width, sp_h = 0.42, 0.53, 0.055

    for i, (name, unit, lo, hi, log_scale) in enumerate(METRICS):
        yc = top - i * row_h
        val = metric_values[i]
        frac = fraction(val, lo, hi, log_scale)
        color = CMAP(frac)

        # Name + value
        fig.text(g_left, yc + 0.028, name,
                 fontsize=9, fontweight="bold", va="bottom",
                 transform=fig.transFigure)
        fig.text(g_left + g_width, yc + 0.028,
                 f"{fmt_val(val)} {unit}",
                 fontsize=9, va="bottom", ha="right", color="#333",
                 transform=fig.transFigure)

        # Gauge
        bar_y = yc + 0.008
        bar_h = 0.014
        fig.patches.append(mpatches.FancyBboxPatch(
            (g_left, bar_y), g_width, bar_h,
            boxstyle="round,pad=0.002", fc="#e8e8e8", ec="none",
            transform=fig.transFigure, zorder=1))
        fig.patches.append(mpatches.FancyBboxPatch(
            (g_left, bar_y), max(g_width * frac, 0.004), bar_h,
            boxstyle="round,pad=0.002", fc=color, ec="none",
            transform=fig.transFigure, zorder=2))
        fig.text(g_left, bar_y - 0.004, fmt_val(lo),
                 fontsize=5.5, va="top", color="#999",
                 transform=fig.transFigure)
        fig.text(g_left + g_width, bar_y - 0.004, fmt_val(hi),
                 fontsize=5.5, va="top", ha="right", color="#999",
                 transform=fig.transFigure)

        # Sparkline
        ax = fig.add_axes([sp_left, yc - 0.012, sp_width, sp_h])
        key = SERIES_KEYS[i]
        t, vals = data["series"][key]
        is_bar = (key == "lockloss")
        plot_sparkline(ax, t, vals, key, color,
                       ylabel=SERIES_YLABELS[i], is_bar=is_bar)

        if i == len(METRICS) - 1:
            ax.set_xlabel("Hours", fontsize=6, labelpad=1)
        else:
            ax.set_xticklabels([])

    # ── Bottom section: polar skyplot + elevation/C/N0 plot ── #
    bottom_top = top - len(METRICS) * row_h - 0.01
    polar_h = 0.25
    polar_w = 0.28

    # Left: Polar C/N0 heatmap — mean C/N0 per (az, el) cell
    ax_sky = fig.add_axes([0.08, bottom_top - polar_h, polar_w, polar_h],
                          projection="polar")
    setup_polar_ax(ax_sky)
    ax_sky.set_title("Sky C/N0 heatmap", fontsize=7, pad=8)

    sky = data.get("sky")
    if sky is not None and len(sky) > 0:
        pc, grid = polar_cno_heatmap(
            ax_sky, sky["azim_deg"].values, sky["elev_deg"].values,
            sky["cno_dBHz"].values)

        # C/N0 colorbar
        cbar_ax = fig.add_axes([0.08, bottom_top - polar_h - 0.02,
                                polar_w, 0.008])
        cb = fig.colorbar(pc, cax=cbar_ax, orientation="horizontal")
        cb.set_label("C/N0 (dBHz)", fontsize=5, labelpad=1)
        cb.ax.tick_params(labelsize=5, length=2, pad=1)

    # Overlay slip positions
    slip_elev, slip_azim, slip_cno = data["slip_sky"]
    if len(slip_elev) > 0:
        sa_rad = np.radians(slip_azim)
        sr = 90 - slip_elev
        ax_sky.scatter(sa_rad, sr, c="red", s=6, marker="x",
                       linewidths=0.5, alpha=0.6, zorder=5, label="Lock loss")
        ax_sky.legend(fontsize=5, loc="lower left",
                      bbox_to_anchor=(-0.05, -0.12), frameon=False)

    # Right: C/N0 vs elevation, slip density overlay
    ax_elev = fig.add_axes([0.52, bottom_top - polar_h + 0.02,
                            0.42, polar_h - 0.04])
    if sky is not None and len(sky) > 0:
        # Bin by 5° elevation
        bins = np.arange(0, 95, 5)
        bin_centers = bins[:-1] + 2.5
        cno_means = []
        cno_stds = []
        for b in range(len(bins) - 1):
            mask = (sky["elev_deg"] >= bins[b]) & (sky["elev_deg"] < bins[b + 1])
            vals = sky.loc[mask, "cno_dBHz"]
            cno_means.append(vals.mean() if len(vals) > 10 else np.nan)
            cno_stds.append(vals.std() if len(vals) > 10 else np.nan)
        cno_means = np.array(cno_means)
        cno_stds = np.array(cno_stds)

        ax_elev.fill_between(bin_centers, cno_means - cno_stds,
                             cno_means + cno_stds,
                             alpha=0.2, color="#4488cc")
        ax_elev.plot(bin_centers, cno_means, "o-", color="#4488cc",
                     markersize=3, lw=1.2, label="C/N0")
        ax_elev.set_xlabel("Elevation (°)", fontsize=7)
        ax_elev.set_ylabel("C/N0 (dBHz)", fontsize=7, color="#4488cc")
        ax_elev.tick_params(labelsize=6)
        ax_elev.set_xlim(0, 90)
        simple_formatter(ax_elev)

        # Overlay: lock loss per elevation bin as % (right axis, fixed range)
        if len(slip_elev) > 0:
            slip_counts, _ = np.histogram(slip_elev, bins=bins)
            # Each sky row = 1 sat-minute = 60 sat-seconds of exposure
            exposure_satsec, _ = np.histogram(sky["elev_deg"].values, bins=bins)
            exposure_satsec = exposure_satsec * 60.0
            with np.errstate(divide="ignore", invalid="ignore"):
                loss_pct = np.where(exposure_satsec > 0,
                                    slip_counts / exposure_satsec * 100, np.nan)
            ax2 = ax_elev.twinx()
            ax2.bar(bin_centers, loss_pct, width=4, alpha=0.3,
                    color="red", label="Lock loss %")
            ax2.set_ylabel("Lock loss (%)", fontsize=7, color="red")
            ax2.set_ylim(0, SLIP_PCT_MAX)
            ax2.tick_params(labelsize=6, colors="red")
            ax2.spines["right"].set_color("red")
            simple_formatter(ax2)

    ax_elev.set_title("C/N0 & lock loss vs elevation", fontsize=7, pad=6)
    ax_elev.spines["top"].set_visible(False)
    ax_elev.grid(True, axis="y", linewidth=0.3, alpha=0.4)
    # Secondary x-axis labels
    ax_elev.text(0.0, -0.18, "Horizon", fontsize=6, color="#888",
                 ha="left", transform=ax_elev.transAxes)
    ax_elev.text(1.0, -0.18, "Overhead", fontsize=6, color="#888",
                 ha="right", transform=ax_elev.transAxes)

    # ── Footer ─────────────────────────────────────────────── #
    fy = 0.025
    fig.add_artist(plt.Line2D([0.05, 0.95], [fy + 0.015, fy + 0.015],
                              color="#cccccc", lw=0.8,
                              transform=fig.transFigure, clip_on=False))
    fig.text(0.05, fy,
             f"Gauges: left = worst, right = best.  "
             f"Full evaluation requires {MINIMUM_HOURS} h "
             f"(one GPS constellation repeat).",
             fontsize=6.5, va="top", color="#888")
    fig.text(0.95, fy,
             f"Produced by testAnt v{VERSION} — "
             f"https://github.com/bobvan/testAnt",
             fontsize=6.5, va="top", ha="right", color="#888")

    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf_path}")


# ── CLI ────────────────────────────────────────────────────────────── #

def main():
    ap = argparse.ArgumentParser(description="One-page antenna evaluation PDF")
    ap.add_argument("--snr", type=str, required=True,
                    help="Per-minute SNR summary CSV")
    ap.add_argument("--rawx", type=str, required=True,
                    help="Per-minute RAWX/CMC summary CSV")
    ap.add_argument("--ticc", type=str, required=True,
                    help="Raw TICC CSV")
    ap.add_argument("--timtp", type=str, required=True,
                    help="Raw TIM-TP CSV")
    ap.add_argument("--sky", type=str, default=None,
                    help="Per-minute per-SV sky position CSV")
    ap.add_argument("--slips", type=str, default=None,
                    help="Cycle slip events CSV")
    ap.add_argument("--rx", type=str, default="BOT", choices=["TOP", "BOT"],
                    help="Which receiver channel to report on (default BOT)")
    ap.add_argument("--antenna", type=str, default="Antenna",
                    help="Antenna name for report title")
    ap.add_argument("--mount", type=str, default="",
                    help="Mount description")
    ap.add_argument("--receiver", type=str, default="",
                    help="Receiver model")
    ap.add_argument("--notes", type=str, default="",
                    help="Additional notes for header")
    ap.add_argument("--out", type=str, default="data/report_card.pdf",
                    help="Output PDF path")
    args = ap.parse_args()

    data = load_data(args.snr, args.rawx, args.ticc, args.timtp,
                     sky_csv=args.sky, slips_csv=args.slips,
                     rx=args.rx,
                     antenna=args.antenna, mount=args.mount,
                     receiver=args.receiver, notes=args.notes)
    render_report(data, args.out)


if __name__ == "__main__":
    main()
