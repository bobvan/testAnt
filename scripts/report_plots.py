#!/usr/bin/env python3
"""
report_plots.py — Shared constants and plot functions for testAnt reports.

Used by both report_card.py (PDF generation) and report_explore.ipynb (interactive).
Changes here propagate to both automatically.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter, FuncFormatter

# ── Version ───────────────────────────────────────────────────────── #

VERSION = "0.3.0"   # report format version — bump when layout/axes change
MINIMUM_HOURS = 24   # full evaluation threshold

# ── Colormap ──────────────────────────────────────────────────────── #
# RdYlGn with darkened yellows for white-background contrast.

CMAP = LinearSegmentedColormap.from_list("RdAmGn", [
    (0.0,  (0.843, 0.188, 0.153)),   # red
    (0.35, (0.940, 0.560, 0.180)),   # amber
    (0.50, (0.780, 0.680, 0.100)),   # dark gold
    (0.65, (0.500, 0.720, 0.240)),   # olive-green
    (1.0,  (0.102, 0.588, 0.255)),   # green
])

# ── Metric definitions ────────────────────────────────────────────── #
# (name, unit, lo=worst, hi=best, log_scale)

METRICS = [
    ("C/N0 mean",          "dBHz",  25,   50,     False),
    ("Multipath (CMC)",    "m",     1.0,  0.03,   True),
    ("ADEV @ τ=1 s",      "ns",    10,   0.1,    True),
    ("Carrier lock loss",  "%",     10,   0.01,   True),
    ("Satellite count",    "SVs",   4,    30,     False),
]

# ── Fixed sparkline Y-axis ranges ─────────────────────────────────── #
# key → (ymin, ymax) or (ymin, ymax, "log")

FIXED_Y = {
    "cno":       (5,    55),
    "cmc":       (0.02, 2.0,  "log"),
    "adev":      (0.1,  20,   "log"),
    "lockloss":  (0,    2),
    "satcount":  (0,    40),
}

SERIES_KEYS = ["cno", "cmc", "adev", "lockloss", "satcount"]
SERIES_YLABELS = ["dBHz", "m", "ns", "lock loss %", "SVs"]

# Fixed x-axis: one full constellation repeat
FIXED_X_HOURS = 24

# C/N0 heatmap range
CNO_VMIN, CNO_VMAX = 20, 50

# Slip rate fixed range on elevation plot
SLIP_PCT_MAX = 50

# ── GNSS constellation orbital parameters ─────────────────────────── #

CONSTELLATIONS = [
    ("GPS",     55.0,  20180),
    ("Galileo", 56.0,  23222),
    ("BDS MEO", 55.0,  21528),
    ("GLONASS", 64.8,  19130),
]
R_EARTH = 6371.0  # km


# ── Helpers ───────────────────────────────────────────────────────── #

def fraction(value, lo, hi, log_scale):
    """Map value to 0–1 where 0 = worst (lo), 1 = best (hi)."""
    if log_scale:
        value = np.log10(max(value, 1e-12))
        lo = np.log10(max(lo, 1e-12))
        hi = np.log10(max(hi, 1e-12))
    return float(np.clip((value - lo) / (hi - lo), 0, 1))


def fmt_val(value):
    """Format a metric value with appropriate decimal places."""
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def simple_formatter(ax, axis="y"):
    """Replace scientific notation with plain decimal labels."""
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


def log_tick_fmt(v, _pos):
    """Plain decimal labels for log-scale axes."""
    if v <= 0:
        return ""
    if v < 0.01:
        return f"{v:.3f}"
    if v < 0.1:
        return f"{v:.2f}"
    if v < 1:
        return f"{v:.1f}"
    if v < 10:
        return f"{v:.1f}"
    return f"{v:.0f}"


def apply_fixed_y(ax, key):
    """Apply fixed Y-axis range from FIXED_Y dict.

    Handles log-scale entries (ymin, ymax, "log") and linear (ymin, ymax).
    For log scale, sets nice tick marks within the range.
    """
    fy = FIXED_Y.get(key)
    if not fy:
        return
    if len(fy) == 3 and fy[2] == "log":
        ax.set_yscale("log")
        ax.set_ylim(fy[0], fy[1])
        lo_exp = np.floor(np.log10(fy[0]))
        hi_exp = np.ceil(np.log10(fy[1]))
        candidates = []
        for exp in range(int(lo_exp) - 1, int(hi_exp) + 2):
            for mult in [1, 2, 5]:
                candidates.append(mult * 10**exp)
        ticks = [c for c in candidates if fy[0] <= c <= fy[1]]
        while len(ticks) > 5:
            ticks = ticks[::2]
        if ticks:
            ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(FuncFormatter(log_tick_fmt))
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
    else:
        simple_formatter(ax)
        ax.set_ylim(fy[0], fy[1])


# ── Lock loss computation ─────────────────────────────────────────── #

def compute_lock_loss(slips_hours, sky_hours, total_hours, n_bins=None):
    """Compute lock loss percentage per time bin and overall.

    Args:
        slips_hours: array of slip event times in hours
        sky_hours: array of sky observation times in hours (1 per sat-minute)
        total_hours: total observation duration in hours
        n_bins: number of 1-hour bins (default: int(total_hours))

    Returns:
        t_centers: bin centers in hours
        loss_pct: lock loss % per bin
        overall_pct: overall lock loss %
    """
    if n_bins is None:
        n_bins = max(1, int(total_hours))

    bin_edges = np.arange(n_bins + 1)
    slip_counts, _ = np.histogram(slips_hours, bins=bin_edges)

    # Exposure: satellite-minutes per hour bin from sky data
    # Each sky row = 1 satellite tracked for 1 minute = 60 sat-seconds
    sky_counts, _ = np.histogram(sky_hours, bins=bin_edges)
    sat_seconds = sky_counts * 60.0

    with np.errstate(divide="ignore", invalid="ignore"):
        loss_pct = np.where(sat_seconds > 0,
                            slip_counts / sat_seconds * 100, 0)

    # Overall
    total_sat_seconds = len(sky_hours) * 60.0
    overall_pct = (len(slips_hours) / total_sat_seconds * 100
                   if total_sat_seconds > 0 else 0)

    t_centers = np.arange(n_bins) + 0.5
    return t_centers, loss_pct, overall_pct


# ── Sparkline plot ────────────────────────────────────────────────── #

def plot_sparkline(ax, t, vals, key, color, ylabel=None, is_bar=False):
    """Render a single sparkline on the given axes.

    Args:
        ax: matplotlib axes
        t: time array (hours)
        vals: value array
        key: FIXED_Y key for axis range
        color: line/bar color
        ylabel: Y-axis label (default from SERIES_YLABELS)
        is_bar: if True, render as bar chart instead of line
    """
    if is_bar:
        ax.bar(t, vals, width=0.8, color=color, alpha=0.7, ec="none")
    else:
        ax.plot(t, vals, color=color, lw=0.7, alpha=0.85)
        ax.fill_between(t, vals, alpha=0.12, color=color)

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=6, labelpad=1)

    apply_fixed_y(ax, key)
    ax.set_xlim(0, FIXED_X_HOURS)
    ax.tick_params(labelsize=5.5, length=2, pad=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)


# ── Polar C/N0 heatmap ───────────────────────────────────────────── #

def polar_cno_heatmap(ax, azim, elev, cno, az_bin=10, el_bin=5,
                      min_samples=3, show_contours=True):
    """Render a polar C/N0 heatmap.

    Args:
        ax: polar axes
        azim, elev, cno: arrays of azimuth, elevation, C/N0
        az_bin, el_bin: bin size in degrees
        min_samples: minimum samples per bin to show
        show_contours: overlay white contour lines

    Returns:
        pc: pcolormesh object (for colorbar)
        grid: the binned C/N0 grid
    """
    az_edges = np.linspace(0, 360, int(360 / az_bin) + 1)
    el_edges = np.linspace(0, 90, int(90 / el_bin) + 1)
    grid = np.full((len(el_edges) - 1, len(az_edges) - 1), np.nan)

    for i in range(len(el_edges) - 1):
        for j in range(len(az_edges) - 1):
            mask = ((elev >= el_edges[i]) & (elev < el_edges[i + 1]) &
                    (azim >= az_edges[j]) & (azim < az_edges[j + 1]))
            v = cno[mask]
            if len(v) >= min_samples:
                grid[i, j] = np.mean(v)

    AZ, R = np.meshgrid(np.radians(az_edges), 90 - el_edges)
    pc = ax.pcolormesh(AZ, R, grid, cmap=CMAP, vmin=CNO_VMIN, vmax=CNO_VMAX,
                       shading="flat", rasterized=True)

    if show_contours:
        az_c = np.radians((az_edges[:-1] + az_edges[1:]) / 2)
        r_c = 90 - (el_edges[:-1] + el_edges[1:]) / 2
        AZC, RC = np.meshgrid(az_c, r_c)
        gf = grid.copy()
        for i in range(gf.shape[0]):
            ring_mean = np.nanmean(gf[i])
            if np.isfinite(ring_mean):
                gf[i] = np.where(np.isfinite(gf[i]), gf[i], ring_mean)
        if np.any(np.isfinite(gf)):
            ax.contour(AZC, RC, gf, levels=np.arange(CNO_VMIN, CNO_VMAX + 1, 5),
                       colors="white", linewidths=0.4, alpha=0.6)

    return pc, grid


def setup_polar_ax(ax):
    """Configure a polar axes for sky plots (N at top, CW, 0–90 radius)."""
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 90)
    ax.set_yticks([0, 30, 60, 90])
    ax.set_yticklabels(["90°", "60°", "30°", "0°"], fontsize=5, color="#888")
    ax.set_rlabel_position(45)
    ax.tick_params(axis="x", labelsize=5)
    ax.grid(True, linewidth=0.3, alpha=0.5)


# ── Theoretical sky mask ──────────────────────────────────────────── #

def theoretical_sky_mask(obs_lat_deg, incl_deg, alt_km, az_bin=2):
    """Compute theoretical sky coverage for a GNSS constellation.

    Samples all possible sub-satellite positions for an orbit at the given
    inclination and altitude, computes (az, el) from the observer, and bins
    into an output grid. Uses fine internal sampling (0.25° lat × 0.5° lon)
    to avoid aliasing near-zenith passes.

    Returns:
        az_centers: azimuth bin centers (deg)
        max_el: max elevation at each azimuth (deg)
        az_bins, el_bins: bin edges
        mask: bool grid of reachable (el, az) cells
    """
    phi = np.radians(obs_lat_deg)
    r_sat = R_EARTH + alt_km

    obs = np.array([R_EARTH * np.cos(phi), 0.0, R_EARTH * np.sin(phi)])
    e_east  = np.array([0.0, 1.0, 0.0])
    e_north = np.array([-np.sin(phi), 0.0, np.cos(phi)])
    e_up    = np.array([np.cos(phi), 0.0, np.sin(phi)])

    lats = np.arange(-incl_deg, incl_deg + 0.1, 0.25)
    lons = np.arange(0, 360, 0.5)

    LAT, LON = np.meshgrid(np.radians(lats), np.radians(lons), indexing="ij")
    lat_f = LAT.ravel()
    lon_f = LON.ravel()

    cos_lat = np.cos(lat_f)
    sx = r_sat * cos_lat * np.cos(lon_f)
    sy = r_sat * cos_lat * np.sin(lon_f)
    sz = r_sat * np.sin(lat_f)

    dx = sx - obs[0]
    dy = sy - obs[1]
    dz = sz - obs[2]

    e = e_east[1] * dy
    n = e_north[0] * dx + e_north[2] * dz
    u = e_up[0] * dx + e_up[2] * dz

    horiz = np.sqrt(e**2 + n**2)
    el_deg = np.degrees(np.arctan2(u, horiz))
    az_deg_all = np.degrees(np.arctan2(e, n)) % 360

    above = el_deg >= 0
    el_deg = el_deg[above]
    az_deg_all = az_deg_all[above]

    az_bins = np.arange(0, 360 + az_bin, az_bin)
    el_bins = np.arange(0, 91, 5)
    mask = np.zeros((len(el_bins) - 1, len(az_bins) - 1), dtype=bool)

    az_idx = np.clip((az_deg_all / az_bin).astype(int), 0, len(az_bins) - 2)
    el_idx = np.clip((el_deg / 5).astype(int), 0, len(el_bins) - 2)
    mask[el_idx, az_idx] = True

    max_el = np.full(len(az_bins) - 1, np.nan)
    for j in range(len(az_bins) - 1):
        in_bin = (az_deg_all >= az_bins[j]) & (az_deg_all < az_bins[j + 1])
        if np.any(in_bin):
            max_el[j] = np.max(el_deg[in_bin])

    az_centers = (az_bins[:-1] + az_bins[1:]) / 2
    return az_centers, max_el, az_bins, el_bins, mask
