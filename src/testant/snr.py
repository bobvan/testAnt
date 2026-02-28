"""
snr.py — Extract per-satellite SNR and satellite count from parsed messages.

Handles:
  • NMEA GSV  sentences  (signal strength in dBHz per satellite)
  • UBX-NAV-SAT messages (richer: C/N0, elevation, azimuth, health flags)

Both sources are normalised into the same SatSnapshot dataclass so the
rest of the code doesn't care which message type arrived.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class SatInfo:
    """Signal data for one satellite at one epoch."""
    gnss_id: str          # e.g. "GPS", "GLO", "GAL", "BDS"
    sv_id: int            # satellite PRN / slot number
    cno: float            # carrier-to-noise density, dBHz
    elev: float = float("nan")   # elevation angle, degrees
    azim: float = float("nan")   # azimuth angle, degrees
    used: bool = False    # receiver is using this SV in the solution


@dataclass
class SatSnapshot:
    """All satellites seen at one epoch from one receiver."""
    timestamp: datetime
    receiver_label: str
    satellites: list[SatInfo] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.satellites)

    @property
    def used_count(self) -> int:
        return sum(1 for s in self.satellites if s.used)

    @property
    def mean_cno(self) -> float:
        cnos = [s.cno for s in self.satellites if s.cno > 0]
        return sum(cnos) / len(cnos) if cnos else float("nan")


# ------------------------------------------------------------------ #
# Parsers                                                             #
# ------------------------------------------------------------------ #

_GNSS_ID_MAP = {0: "GPS", 1: "SBAS", 2: "GAL", 3: "BDS", 5: "QZSS", 6: "GLO"}


def snapshot_from_navsat(msg: Any, label: str, timestamp: datetime) -> SatSnapshot:
    """Build a SatSnapshot from a UBX-NAV-SAT message."""
    snap = SatSnapshot(timestamp=timestamp, receiver_label=label)
    num = getattr(msg, "numSvs", 0)
    for i in range(num):
        gnss_id_raw = getattr(msg, f"gnssId_{i:02d}", None)
        sv_id       = getattr(msg, f"svId_{i:02d}", 0)
        cno         = getattr(msg, f"cno_{i:02d}", 0)
        elev        = getattr(msg, f"elev_{i:02d}", float("nan"))
        azim        = getattr(msg, f"azim_{i:02d}", float("nan"))
        flags       = getattr(msg, f"flags_{i:02d}", 0)
        used        = bool(flags & 0x08)  # bit 3 = svUsed
        gnss_name   = _GNSS_ID_MAP.get(gnss_id_raw, f"GNSS{gnss_id_raw}")
        snap.satellites.append(SatInfo(gnss_name, sv_id, float(cno), float(elev), float(azim), used))
    return snap


def snapshot_from_gsv(sentences: list[Any], label: str, timestamp: datetime) -> SatSnapshot:
    """
    Build a SatSnapshot from a group of NMEA GSV sentences for one talker.

    sentences should be all GSV messages for one talker ID (e.g. $GPGSV)
    collected within the same epoch.
    """
    snap = SatSnapshot(timestamp=timestamp, receiver_label=label)
    for msg in sentences:
        for slot in range(1, 5):
            prn  = getattr(msg, f"sv_prn_num_{slot}", None)
            snr  = getattr(msg, f"snr_{slot}", None)
            elev = getattr(msg, f"elevation_deg_{slot}", float("nan"))
            azim = getattr(msg, f"azimuth_{slot}", float("nan"))
            if prn is None:
                continue
            cno = float(snr) if snr not in (None, "") else 0.0
            snap.satellites.append(SatInfo("?", int(prn), cno, float(elev or 0), float(azim or 0)))
    return snap
