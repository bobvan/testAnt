#!/usr/bin/env python3
"""Absolute signal-quality metrics from raw GNSS observations.

Computes code multipath and carrier-phase noise from a logged observation
stream, in a form that is comparable **across stations, receivers and sites** —
which is what a mean-removed code-minus-carrier figure is not.

    ./obs_quality.py --label PTBB  logs/ptbb_20260826_00.rtcm3
    ./obs_quality.py --label UFO1  logs/prod/ufo1_msm7_20260826_02.rtcm3

Input format (RTCM3 MSM or Septentrio SBF) is auto-detected.

WHAT IT MEASURES

*Code multipath* uses the TEQC-style dual-frequency MP combination

    MP1 = P1 - (1 + 2/(a-1))*L1 + (2/(a-1))*L2,     a = (f1/f2)^2

which cancels geometry, satellite and receiver clocks, troposphere **and the
ionosphere exactly**, leaving multipath + code noise + a per-arc constant.  Only
the arc constant is removed, so the result is absolute.

Contrast with `cmc_std` in bobvan/testAnt, which removes a per-arc *mean* from a
single-frequency P - lambda*phi.  That retains the arc's ionospheric trend.  It
is the right metric for a simultaneous A/B antenna comparison (iono is
common-mode and cancels in the difference) and the wrong one to quote as an
absolute number or compare against a fixed threshold.

*Phase noise* uses the 1 s scatter of the geometry-free combination L1-L2.  Over
one second the ionosphere is effectively frozen, so the epoch-to-epoch
difference is dominated by phase noise on the two signals.

ALWAYS REPORT THE MASK.  The same antenna measured on one dataset gives 0.79 m
unweighted over all arcs and 0.19 m restricted to C/N0 >= 50 dBHz -- a factor of
four from itself.  A code-multipath figure without its elevation or C/N0 cut is
not comparable to anything, which is why the C/N0 breakdown is printed by
default rather than being an option.

See docs/antenna-quality-metrics-and-timing.md.
"""
import argparse, collections, io, os, sys

import numpy as np

# ── Observation decoders live in PePPAR-Fix, deliberately not vendored ──────
# The RTCM3 MSM and Septentrio SBF decoders this needs are maintained in
# bobvan/PePPAR-Fix (`scripts/peppar_fix/`).  They are *referenced*, not copied,
# because two copies of a decoder drift and the drift is silent -- you get
# plausible numbers from stale bit-unpacking.
#
# This is a known wart: it makes testAnt depend on a sibling checkout.  The
# proper fix is for the shared ingest layer to become its own package when the
# metric work gets its own repo; see docs/future-work-signal-quality.md.
#
# Override with PEPPAR_FIX_SCRIPTS=/path/to/PePPAR-Fix/scripts
def _find_peppar_fix():
    env = os.environ.get("PEPPAR_FIX_SCRIPTS")
    cands = [env] if env else []
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cands += [os.path.join(os.path.dirname(here), n, "scripts")
              for n in ("PePPAR-Fix", "peppar-fix", "PePPAR-Fix-charlie")]
    cands.append(os.path.expanduser("~/peppar-fix/scripts"))
    cands.append(os.path.expanduser("~/git/PePPAR-Fix/scripts"))
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "peppar_fix")):
            return c
    sys.exit(
        "cannot find the PePPAR-Fix decoders.\n"
        "  This script reads RTCM3 MSM and Septentrio SBF via\n"
        "  scripts/peppar_fix/{rtcm_msm_obs,sbf_obs}.py in bobvan/PePPAR-Fix.\n"
        "  Point at a checkout with:  export PEPPAR_FIX_SCRIPTS=/path/to/PePPAR-Fix/scripts")

sys.path.insert(0, _find_peppar_fix())

C = 299792458.0
# (f1, f2) pairs to form MP from, per constellation.  Both must be present in
# the same epoch for that SV.
PAIRS = (("GPS-L1CA", "GPS-L2W"), ("GAL-E1C", "GAL-E5aQ"))
CN0_BINS = ((0, 38), (38, 44), (44, 50), (50, 99))
ELEV_BINS = ((0, 15), (15, 30), (30, 50), (50, 91))   # degrees
MIN_ARC = 300          # epochs; shorter arcs give unstable RMS
SLIP_M = 2.0           # jump in MP that means a cycle slip / new arc
# An arc breaks on a data gap.  The threshold MUST scale with the sample
# interval: hardcoding 5 s meant that at 30 s RINEX every epoch looked like a
# gap and every arc was length 1, which surfaced as "no arcs long enough"
# rather than as a wrong number -- a benign failure, but only by luck.
GAP_FACTOR = 2.5       # a gap is > GAP_FACTOR x the median epoch interval


# RINEX 3 observation codes -> the sig_name vocabulary used above, so RINEX
# results are directly comparable with the RTCM/SBF paths.
_RNX_MAP = {
    ("G", "C1C", "L1C"): ("GPS-L1CA", 1575.42e6),
    ("G", "C2W", "L2W"): ("GPS-L2W",  1227.60e6),
    ("E", "C1C", "L1C"): ("GAL-E1C",  1575.42e6),
    ("E", "C5Q", "L5Q"): ("GAL-E5aQ", 1176.45e6),
}


def load_rinex(path):
    """Minimal RINEX 3 OBS reader — only the observables the MP pairs need.

    Written rather than pulled in from georinex because we need four columns
    out of a 37 MB file and not an xarray Dataset.  RINEX 3 data records are
    F14.3 + LLI + SSI = 16 chars per observable, SV id in cols 1-3.
    """
    import datetime as _dt
    obs_types, ep = {}, collections.defaultdict(lambda: collections.defaultdict(dict))
    rx_ecef = None
    with open(path, "r", errors="replace") as fh:
        sysc = None
        for line in fh:
            lab = line[60:].strip()
            if lab == "APPROX POSITION XYZ":
                try: rx_ecef = tuple(float(line[i*14:(i+1)*14]) for i in range(3))
                except ValueError: pass
            if lab == "SYS / # / OBS TYPES":
                if line[0] != " ":
                    sysc = line[0]
                    obs_types[sysc] = []
                obs_types[sysc] += line[7:60].split()
            elif lab == "END OF HEADER":
                break
        t0 = None
        for line in fh:
            if line.startswith(">"):
                f = line[1:].split()
                if len(f) < 6:
                    t = None; continue
                y, mo, d, h, mi = (int(x) for x in f[:5])
                sec = float(f[5])
                dtv = _dt.datetime(y, mo, d, h, mi) + _dt.timedelta(seconds=sec)
                # GPS TOW, so ephemeris matching works directly.  RINEX epochs
                # are UTC; GPS time leads UTC by the leap-second count.
                gps = dtv + _dt.timedelta(seconds=LEAP_S)
                days = (gps - _dt.datetime(1980, 1, 6)).days
                t = (days % 7) * 86400 + gps.hour * 3600 + gps.minute * 60 + gps.second + (gps.microsecond / 1e6)
                continue
            if t is None or len(line) < 4:
                continue
            sv = line[:3].strip()
            sysc = sv[0]
            types = obs_types.get(sysc)
            if not types:
                continue
            vals = {}
            for i, ty in enumerate(types):
                fld = line[3 + i * 16: 3 + i * 16 + 14]
                if fld.strip():
                    try: vals[ty] = float(fld)
                    except ValueError: pass
            for (sc, ccode, lcode), (name, freq) in _RNX_MAP.items():
                if sc != sysc:
                    continue
                if ccode in vals and lcode in vals:
                    cn = vals.get("S" + ccode[1:], None)
                    ep[t][sv][name] = (vals[ccode], vals[lcode] * C / freq, freq, cn)
    return ep, rx_ecef


# ── True elevation, via broadcast ephemeris ─────────────────────────────────
# C/N0 is only a PROXY for elevation, and a biased one: a lower-gain antenna
# reads lower C/N0 at every elevation, so its ">=50 dBHz" bucket holds
# satellites a better antenna would have put in 44-50.  Comparing two different
# antennas by C/N0 bucket therefore mixes the buckets in a way that can flatten
# or fake an elevation dependence.  Binning by geometric elevation removes that
# entirely, which is why this exists.
LEAP_S = 18          # GPS-UTC as of 2026
OMEGA_E = 7.2921151467e-5
GM = {"G": 3.986005e14, "E": 3.986004418e14, "C": 3.986004418e14}


def _kepler_ecef(e, tk, gm):
    """IS-GPS-200 Table 20-IV. Same model as PePPAR-Fix broadcast_eph."""
    import math
    a = e["sqrt_a"] ** 2
    n = math.sqrt(gm / a ** 3) + e["delta_n"]
    Mk = e["M0"] + n * tk
    Ek = Mk
    for _ in range(12):
        Ek = Mk + e["e"] * math.sin(Ek)
    den = 1.0 - e["e"] * math.cos(Ek)
    vk = math.atan2(math.sqrt(1.0 - e["e"] ** 2) * math.sin(Ek) / den,
                    (math.cos(Ek) - e["e"]) / den)
    phik = vk + e["omega"]
    s2, c2 = math.sin(2 * phik), math.cos(2 * phik)
    uk = phik + e["Cus"] * s2 + e["Cuc"] * c2
    rk = a * den + e["Crs"] * s2 + e["Crc"] * c2
    ik = e["i0"] + e["Cis"] * s2 + e["Cic"] * c2 + e["i_dot"] * tk
    xo, yo = rk * math.cos(uk), rk * math.sin(uk)
    Ok = e["omega0"] + (e["omega_dot"] - OMEGA_E) * tk - OMEGA_E * e["toe"]
    return (xo * math.cos(Ok) - yo * math.cos(ik) * math.sin(Ok),
            xo * math.sin(Ok) + yo * math.cos(ik) * math.cos(Ok),
            yo * math.sin(ik))


# RINEX 3 NAV: the 7 broadcast-orbit lines after the SV/epoch line, in order.
_NAV_FIELDS = [
    ("iode", "Crs", "delta_n", "M0"),
    ("Cuc", "e", "Cus", "sqrt_a"),
    ("toe", "Cic", "omega0", "Cis"),
    ("i0", "Crc", "omega", "omega_dot"),
    ("i_dot", "l2code", "week", "l2pflag"),
]


def load_nav(path):
    """RINEX 3 NAV -> {prn: [eph dicts sorted by toe]}. GPS/GAL/BDS only."""
    import collections as _c
    nav = _c.defaultdict(list)
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if line[60:].strip() == "END OF HEADER":
                break
        cur = None
        for line in fh:
            if line[:1] in ("G", "E", "C") and line[3:4] == " ":
                prn = line[:3]
                cur = {"prn": prn, "sys": prn[0]}
                nav[prn].append(cur)
                cur["_n"] = 0
                continue
            if cur is None or cur["_n"] >= len(_NAV_FIELDS):
                continue
            vals = []
            for k in range(4):
                f = line[4 + k * 19: 4 + k * 19 + 19].replace("D", "E").replace("d", "E")
                try: vals.append(float(f))
                except ValueError: vals.append(0.0)
            for name, v in zip(_NAV_FIELDS[cur["_n"]], vals):
                cur[name] = v
            cur["_n"] += 1
    for prn in nav:
        nav[prn] = [e for e in nav[prn] if e.get("_n", 0) >= 5]
        nav[prn].sort(key=lambda e: e.get("toe", 0))
    return nav


def elevations(nav, rx_ecef, prns, tow):
    """{prn: elevation_deg} for one epoch."""
    import math
    out = {}
    rn = math.sqrt(sum(c * c for c in rx_ecef))
    up = [c / rn for c in rx_ecef]
    for prn in prns:
        cands = nav.get(prn)
        if not cands:
            continue
        e = min(cands, key=lambda x: abs(((tow - x["toe"] + 302400) % 604800) - 302400))
        tk = ((tow - e["toe"] + 302400) % 604800) - 302400
        if abs(tk) > 7200:
            continue
        try:
            sp = _kepler_ecef(e, tk, GM[e["sys"]])
        except Exception:
            continue
        d = [sp[i] - rx_ecef[i] for i in range(3)]
        dn = math.sqrt(sum(x * x for x in d))
        if dn <= 0:
            continue
        out[prn] = math.degrees(math.asin(sum(d[i] * up[i] for i in range(3)) / dn))
    return out


def load_epochs(path, limit_bytes):
    """-> {tow_s: {sv: {sig: (pr_m, cp_m, freq_hz, cno)}}}"""
    if path.endswith((".rnx", ".obs", ".24o", ".25o", ".26o")):
        ep, rx = load_rinex(path)
        return ep, "RINEX 3", rx
    data = open(path, "rb").read(limit_bytes)
    ep = collections.defaultdict(lambda: collections.defaultdict(dict))
    if data[:1] == b"\xd3":
        from pyrtcm import RTCMReader
        from peppar_fix.rtcm_msm_obs import decode_msm_obs
        fmt = "RTCM3 MSM"
        for _raw, msg in RTCMReader(io.BytesIO(data), quitonerror=0):
            if msg is None or str(msg.identity) not in ("1077", "1087", "1097", "1127"):
                continue
            r = decode_msm_obs(msg)
            if not r:
                continue
            _, tow, cells = r
            _stash(ep, tow, cells)
    elif data[:2] == b"$@":
        from pysbf2 import SBFReader
        from peppar_fix.sbf_obs import decode_meas_epoch
        fmt = "Septentrio SBF"
        for _raw, msg in SBFReader(io.BytesIO(data), quitonerror=0):
            if msg is None:
                continue
            r = decode_meas_epoch(msg)
            if r is None:
                continue
            tow, cells = r
            _stash(ep, tow, cells)
    else:
        sys.exit(f"{path}: not RTCM3 (0xD3) or SBF ($@) — first bytes {data[:4]!r}")
    return ep, fmt, None


def _stash(ep, tow_ms, cells):
    for c in cells:
        pr, cp, f = c.get("pr_m"), c.get("cp_cyc"), c.get("freq_hz")
        if not pr or not cp or not f:
            continue
        ep[tow_ms / 1000.0][c["sv"]][c["sig_name"]] = (pr, cp * C / f, f, c.get("cno"))


def epoch_interval(ep):
    ts = sorted(ep)
    if len(ts) < 3:
        return 1.0
    d = np.diff(np.array(ts))
    d = d[d > 0]
    return float(np.median(d)) if len(d) else 1.0


def metrics(ep, first_only=False, min_arc=MIN_ARC, interval=1.0, elev=None):
    gap = max(1.5, GAP_FACTOR * interval)
    mp = collections.defaultdict(list)
    gf = collections.defaultdict(list)
    for tow in sorted(ep):
        for sv, s in ep[tow].items():
            for s1, s2 in PAIRS:
                if s1 in s and s2 in s:
                    P1, L1, f1, c1 = s[s1]
                    P2, L2, f2, c2 = s[s2]
                    a = (f1 / f2) ** 2
                    k = 2.0 / (a - 1.0)
                    el = elev.get(tow, {}).get(sv) if elev else None
                    mp[(sv, s1)].append((tow, P1 - (1 + k) * L1 + k * L2, c1, el))
                    if not first_only:
                        mp[(sv, s2)].append((tow, P2 - (2 * a / (a - 1)) * L1
                                             + (2 * a / (a - 1) - 1) * L2, c2, el))
                    gf[(sv, s1)].append((tow, L1 - L2))

    bysig = collections.defaultdict(list)
    bins = collections.defaultdict(list)
    ebins = collections.defaultdict(list)
    for (sv, sig), pts in mp.items():
        t = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        cn = np.array([(p[2] or 0) for p in pts], dtype=float)
        ev = np.array([(p[3] if p[3] is not None else np.nan) for p in pts], dtype=float)
        brk = np.where((np.diff(t) > gap) | (np.abs(np.diff(y)) > SLIP_M))[0]
        for seg in np.split(np.arange(len(y)), brk + 1):
            if len(seg) < min_arc:
                continue
            r = y[seg] - y[seg].mean()
            bysig[sig].append(np.std(r))
            for lo, hi in CN0_BINS:
                m = (cn[seg] >= lo) & (cn[seg] < hi)
                if m.sum() > 60:
                    bins[(lo, hi)].append(np.std(r[m]))
            for lo, hi in ELEV_BINS:
                m = (ev[seg] >= lo) & (ev[seg] < hi)
                if m.sum() > 60:
                    ebins[(lo, hi)].append(np.std(r[m]))

    # Phase noise from the 1 s scatter of the geometry-free combination only
    # works when consecutive epochs are 1 s apart -- over 30 s the ionosphere
    # has genuinely moved and the difference is no longer dominated by phase
    # noise.  Report n/a rather than a number that means something else.
    scatter = []
    if interval > 1.5:
        return bysig, bins, ebins, float("nan"), 0
    for _key, pts in gf.items():
        t = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        d, dt = np.diff(y), np.diff(t)
        ok = (dt == 1.0) & (np.abs(d) < 0.05)
        if ok.sum() > 500:
            scatter.append(np.std(d[ok]))
    # two signals differenced once: divide by 2 for a per-signal figure
    phase_mm = (np.median(scatter) / 2 * 1000) if scatter else float("nan")
    return bysig, bins, ebins, phase_mm, len(scatter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--label", default=None)
    ap.add_argument("--nav", default=None,
                    help="RINEX NAV file — enables binning by TRUE ELEVATION "
                         "instead of C/N0, which is the only way to compare two "
                         "different antennas fairly (see the module docstring)")
    ap.add_argument("--rx-ecef", default=None,
                    help="X,Y,Z metres; needed with --nav when the input has no "
                         "APPROX POSITION XYZ header (RTCM/SBF logs)")
    ap.add_argument("--signals", choices=("all", "l1"), default="all",
                    help="'l1' restricts to the first signal of each pair "
                         "(GPS-L1CA / GAL-E1C). The L2/E5 signals carry "
                         "markedly worse code multipath, so this choice moves "
                         "the answer -- state which you used.")
    ap.add_argument("--min-arc", type=int, default=None,
                    help="minimum epochs per arc (default: 5 min worth at the "
                         "file's own sample rate, so 300 at 1 Hz and 10 at 30 s "
                         "-- raise it for sparse data)")
    ap.add_argument("--decimate", type=float, default=None,
                    help="keep only epochs on this many seconds, e.g. 30 to "
                         "compare 1 Hz data against 30 s RINEX on equal terms")
    ap.add_argument("--max-mb", type=float, default=12.0,
                    help="bytes to read; keeps a long log to a sane runtime")
    a = ap.parse_args()
    label = a.label or os.path.basename(a.path)

    ep, fmt, rx_ecef = load_epochs(a.path, int(a.max_mb * 1e6))
    if a.rx_ecef:
        rx_ecef = tuple(float(x) for x in a.rx_ecef.replace(",", " ").split())
    if a.decimate:
        ep = {t: v for t, v in ep.items() if abs(t % a.decimate) < 1e-6}
    iv = epoch_interval(ep)
    min_arc = a.min_arc if a.min_arc else max(10, int(round(300 / max(iv, 1e-9))))
    elev = None
    if a.nav:
        if not rx_ecef:
            sys.exit("--nav needs a receiver position: the file has no "
                     "APPROX POSITION XYZ header, so pass --rx-ecef X,Y,Z")
        nav = load_nav(a.nav)
        elev = {t: elevations(nav, rx_ecef, ep[t].keys(), t) for t in ep}
        n_el = sum(len(v) for v in elev.values())
        if not n_el:
            sys.exit("--nav produced no elevations — wrong day, or TOW mismatch")
    bysig, bins, ebins, phase_mm, n_ph = metrics(
        ep, first_only=(a.signals == "l1"), min_arc=min_arc, interval=iv, elev=elev)
    allv = [v for vs in bysig.values() for v in vs]

    print(f"== {label}   ({fmt}, {len(ep)} epochs @ {iv:g}s, signals={a.signals}, min_arc={min_arc})")
    if not allv:
        print("   no dual-frequency arcs long enough — need both signals of a pair")
        return
    for sig in sorted(bysig):
        v = np.array(bysig[sig])
        print(f"   {sig:10s} {len(v):3d} arcs   MP RMS {v.mean():.3f} m")
    print(f"   {'ALL':10s} {len(allv):3d} arcs   MP RMS {np.mean(allv):.3f} m   (unweighted — see mask note)")
    print("   by C/N0:  " + "   ".join(
        f"{lo}-{hi}: {np.mean(bins[(lo,hi)]):.3f}" for lo, hi in CN0_BINS if bins.get((lo, hi))))
    if elev is not None:
        print("   by ELEV:  " + "   ".join(
            f"{lo}-{hi}deg: {np.mean(ebins[(lo,hi)]):.3f}" for lo, hi in ELEV_BINS
            if ebins.get((lo, hi))))
    if n_ph:
        print(f"   phase noise {phase_mm:.2f} mm = {phase_mm*1e-3/C*1e12:.2f} ps   ({n_ph} arcs)")
    else:
        print("   phase noise n/a — needs 1 Hz epochs (the 1 s geometry-free "
              "estimator is invalid at this rate)")


if __name__ == "__main__":
    main()
