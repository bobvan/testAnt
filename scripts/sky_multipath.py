#!/usr/bin/env python3
"""Map code multipath over the sky (azimuth x elevation) to locate reflectors.

Elevation-only binning averages a reflector around the compass and understates
it.  A vertical obstruction -- a wall, a mast, a building face -- lives at one
azimuth, and shows up here as a sector where MP is high and, below the
obstruction's top edge, where satellites are missing entirely.

    ./sky_multipath.py --nav BRDC.rnx --rx-ecef X,Y,Z log.rtcm3

Reads the same formats as obs_quality.py.  Reports, per azimuth sector:
  * MP RMS within a chosen elevation band (default 30-50 deg)
  * observation count, normalised -- a shadowed sector is *missing data*, which
    is independent evidence and does not rely on the multipath metric at all.
"""
import argparse, collections, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from obs_quality import (load_epochs, load_nav, elevations, PAIRS, C,
                         SLIP_M, GAP_FACTOR, epoch_interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--nav", required=True)
    ap.add_argument("--rx-ecef", default=None)
    ap.add_argument("--max-mb", type=float, default=60.0)
    ap.add_argument("--sectors", type=int, default=12, help="azimuth sectors")
    ap.add_argument("--band", default="30,50", help="elevation band, deg")
    a = ap.parse_args()
    lo_el, hi_el = (float(x) for x in a.band.split(","))

    ep, fmt, rx = load_epochs(a.path, int(a.max_mb * 1e6))
    if a.rx_ecef:
        rx = tuple(float(x) for x in a.rx_ecef.replace(",", " ").split())
    if not rx:
        sys.exit("need --rx-ecef for this format")
    nav = load_nav(a.nav)
    iv = epoch_interval(ep)
    gap = max(1.5, GAP_FACTOR * iv)
    min_arc = max(10, int(round(300 / max(iv, 1e-9))))

    sky = {t: elevations(nav, rx, ep[t].keys(), t) for t in ep}

    # MP series per (sv, signal), carrying elevation and azimuth
    mp = collections.defaultdict(list)
    cnt = collections.Counter()          # (az_sector, el_band) observation count
    W = 360.0 / a.sectors
    for tow in sorted(ep):
        for sv, sig in ep[tow].items():
            ea = sky.get(tow, {}).get(sv)
            if not ea:
                continue
            el, az = ea
            if el < 0:
                continue
            cnt[(int(az // W), int(el // 10))] += 1
            for s1, s2 in PAIRS:
                if s1 in sig and s2 in sig:
                    P1, L1, f1, _ = sig[s1]
                    P2, L2, f2, _ = sig[s2]
                    k = 2.0 / ((f1 / f2) ** 2 - 1.0)
                    mp[(sv, s1)].append((tow, P1 - (1 + k) * L1 + k * L2, el, az))

    cells = collections.defaultdict(list)
    for _key, pts in mp.items():
        t = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
        el = np.array([p[2] for p in pts]); az = np.array([p[3] for p in pts])
        brk = np.where((np.diff(t) > gap) | (np.abs(np.diff(y)) > SLIP_M))[0]
        for seg in np.split(np.arange(len(y)), brk + 1):
            if len(seg) < min_arc:
                continue
            r = y[seg] - y[seg].mean()
            m = (el[seg] >= lo_el) & (el[seg] < hi_el)
            if m.sum() < 20:
                continue
            for sec in range(a.sectors):
                mm = m & (az[seg] // W == sec)
                if mm.sum() >= 20:
                    cells[sec].append(np.std(r[mm]))

    print(f"== {os.path.basename(a.path)}  ({fmt}, {len(ep)} epochs @ {iv:g}s)")
    print(f"   MP RMS by azimuth sector, elevation {lo_el:g}-{hi_el:g} deg\n")
    med = np.median([v for vs in cells.values() for v in vs]) if cells else float("nan")
    print(f"   {'sector':>14s} {'MP RMS':>8s} {'vs med':>7s}  {'n':>4s}  low-el obs count")
    peak = (None, -1)
    for sec in range(a.sectors):
        c0, c1 = sec * W, (sec + 1) * W
        v = cells.get(sec)
        # observations at 0-30 deg in this sector: shadow shows as a deficit
        low = sum(cnt[(sec, b)] for b in range(0, 3))
        if not v:
            print(f"   {c0:5.0f}-{c1:3.0f} deg {'   n/a':>8s} {'':>7s}  {0:4d}  {low:6d}")
            continue
        m = float(np.mean(v))
        bar = "#" * int(round(m / max(med, 1e-9) * 12))
        print(f"   {c0:5.0f}-{c1:3.0f} deg {m:8.3f} {m/med:6.2f}x  {len(v):4d}  {low:6d}  {bar}")
        if m > peak[1]:
            peak = (sec, m)
    if peak[0] is not None:
        c0 = peak[0] * W
        print(f"\n   PEAK sector: {c0:.0f}-{c0+W:.0f} deg  (centre {c0+W/2:.0f} deg)  MP {peak[1]:.3f} m")
    lows = {sec: sum(cnt[(sec, b)] for b in range(0, 3)) for sec in range(a.sectors)}
    if lows:
        mn = min(lows, key=lows.get)
        print(f"   MIN low-elevation observations: sector {mn*W:.0f}-{(mn+1)*W:.0f} deg "
              f"({lows[mn]} obs vs median {int(np.median(list(lows.values())))})")


if __name__ == "__main__":
    main()
