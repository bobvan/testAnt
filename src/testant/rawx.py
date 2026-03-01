"""
rawx.py — Parse UBX-RXM-RAWX messages into RawxMeas dataclasses.

UBX-RXM-RAWX provides raw GNSS measurements: pseudorange, carrier phase,
and Doppler for every tracked signal.  These are the inputs for computing
code-minus-carrier (CMC) multipath metrics.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class RawxMeas:
    """One raw GNSS measurement from a single UBX-RXM-RAWX message."""
    rcv_tow: float        # receiver time of week, seconds
    gnss_id: str          # "GPS", "GAL", "BDS", "GLO", …
    sv_id: int            # satellite identifier
    signal_id: str        # "GPS-L1CA", "GPS-L5Q", "GAL-E1C", …
    pseudorange: float    # prMes, metres
    carrier_phase: float  # cpMes, cycles
    doppler: float        # doMes, Hz
    cno: float            # carrier-to-noise density, dBHz
    locktime_ms: int      # carrier phase lock time counter, ms
    pr_valid: bool        # trkStat bit 0 — pseudorange valid
    cp_valid: bool        # trkStat bit 1 — carrier phase valid
    half_cyc: bool        # trkStat bit 2 — half-cycle resolved (1 = good; 0 = unknown)


# ── GNSS ID mapping ──────────────────────────────────────────────────── #
_GNSS_ID_MAP = {
    0: "GPS", 1: "SBAS", 2: "GAL", 3: "BDS",
    4: "IMES", 5: "QZSS", 6: "GLO",
}

# ── Signal name mapping ──────────────────────────────────────────────── #
# Maps (UBX gnssId int, UBX sigId int) → human-readable signal name.
# UBX sigId differs from NMEA signalID — this is the UBX protocol table.
# Source: u-blox ZED-F9T Interface Description, UBX-RXM-RAWX sigId field.
_RAWX_SIGNAL_NAME: dict[tuple[int, int], str] = {
    (0, 0): "GPS-L1CA",
    (0, 3): "GPS-L2CL",
    (0, 4): "GPS-L2CM",
    (0, 6): "GPS-L5I",
    (0, 7): "GPS-L5Q",
    (1, 0): "SBAS-L1CA",
    (2, 0): "GAL-E1C",
    (2, 1): "GAL-E1B",
    (2, 3): "GAL-E5aI",
    (2, 4): "GAL-E5aQ",
    (2, 5): "GAL-E5bI",
    (2, 6): "GAL-E5bQ",
    (3, 0): "BDS-B1I",
    (3, 1): "BDS-B1I-D2",
    (3, 2): "BDS-B2I",
    (3, 3): "BDS-B2I-D2",
    (3, 5): "BDS-B1C",
    (3, 6): "BDS-B1C-Q",
    (3, 7): "BDS-B2aI",
    (5, 0): "QZSS-L1CA",
    (5, 4): "QZSS-L2CM",
    (5, 5): "QZSS-L2CL",
    (6, 0): "GLO-L1OF",
    (6, 2): "GLO-L2OF",
}


def snapshot_from_rawx(
    msg: Any,
    label: str,
    antenna_mount: str,
    timestamp: datetime,
) -> list[RawxMeas]:
    """
    Parse one UBX-RXM-RAWX message into a list of RawxMeas.

    Skips measurements where prValid (trkStat bit 0) is not set.
    Uses the same _NN zero-padded attribute naming as NAV-SAT in pyubx2.
    """
    rcv_tow = float(getattr(msg, "rcvTow", 0.0))
    num     = int(getattr(msg, "numMeas", 0))
    result: list[RawxMeas] = []

    # pyubx2 uses 1-based indices for RXM-RAWX repeated group fields
    # (e.g. prMes_01 … prMes_NN) and pre-parses trkStat bits into individual
    # boolean fields: prValid_NN, cpValid_NN, halfCyc_NN.
    for i in range(1, num + 1):
        pr_valid = bool(getattr(msg, f"prValid_{i:02d}", 0))
        if not pr_valid:
            continue

        gnss_id_raw = int(getattr(msg, f"gnssId_{i:02d}", 0))
        sig_id_raw  = int(getattr(msg, f"sigId_{i:02d}", 0))
        sv_id       = int(getattr(msg, f"svId_{i:02d}", 0))
        pr          = float(getattr(msg, f"prMes_{i:02d}", float("nan")))
        cp          = float(getattr(msg, f"cpMes_{i:02d}", float("nan")))
        do          = float(getattr(msg, f"doMes_{i:02d}", float("nan")))
        cno         = float(getattr(msg, f"cno_{i:02d}", 0.0))
        locktime    = int(getattr(msg, f"locktime_{i:02d}", 0))
        cp_valid    = bool(getattr(msg, f"cpValid_{i:02d}", 0))
        half_cyc    = bool(getattr(msg, f"halfCyc_{i:02d}", 0))

        gnss_name  = _GNSS_ID_MAP.get(gnss_id_raw, f"GNSS{gnss_id_raw}")
        signal_name = _RAWX_SIGNAL_NAME.get(
            (gnss_id_raw, sig_id_raw),
            f"{gnss_name}-sig{sig_id_raw}",
        )

        result.append(RawxMeas(
            rcv_tow=rcv_tow,
            gnss_id=gnss_name,
            sv_id=sv_id,
            signal_id=signal_name,
            pseudorange=pr,
            carrier_phase=cp,
            doppler=do,
            cno=cno,
            locktime_ms=locktime,
            pr_valid=pr_valid,
            cp_valid=cp_valid,
            half_cyc=half_cyc,
        ))

    return result
