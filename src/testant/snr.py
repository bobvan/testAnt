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
    """Signal data for one satellite signal at one epoch."""
    gnss_id: str          # e.g. "GPS", "GLO", "GAL", "BDS"
    sv_id: int            # satellite PRN / slot number
    cno: float            # carrier-to-noise density, dBHz
    elev: float = float("nan")   # elevation angle, degrees
    azim: float = float("nan")   # azimuth angle, degrees
    used: bool = False    # receiver is using this SV in the solution
    signal_id: str = ""   # e.g. "GPS-L1CA", "GAL-E1C", "BDS-B1I", "BDS-B2aI"


@dataclass
class SatSnapshot:
    """All satellites seen at one epoch from one receiver."""
    timestamp: datetime
    receiver_label: str
    antenna_mount: str = ""
    mount_site: str = ""
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


def snapshot_from_navsat(msg: Any, label: str, timestamp: datetime,
                         antenna_mount: str = "", mount_site: str = "") -> SatSnapshot:
    """Build a SatSnapshot from a UBX-NAV-SAT message."""
    snap = SatSnapshot(timestamp=timestamp, receiver_label=label,
                       antenna_mount=antenna_mount, mount_site=mount_site)
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


_TALKER_GNSS = {"GP": "GPS", "GL": "GLO", "GA": "GAL", "GB": "BDS", "GQ": "QZSS"}

# Maps (NMEA talker, signalID string) → human-readable signal name.
# Source: u-blox ZED-F9T NMEA protocol specification.
_GSV_SIGNAL_NAME: dict[tuple[str, str], str] = {
    ("GP", "1"): "GPS-L1CA",
    ("GP", "6"): "GPS-L2CL",
    ("GP", "7"): "GPS-L5I",
    ("GP", "8"): "GPS-L5Q",
    ("GA", "1"): "GAL-E1B",
    ("GA", "7"): "GAL-E1C",
    ("GA", "2"): "GAL-E5aI",
    ("GA", "3"): "GAL-E5aQ",
    ("GA", "4"): "GAL-E6C",
    ("GB", "1"): "BDS-B1I",
    ("GB", "2"): "BDS-B1I-D2",
    ("GB", "3"): "BDS-B2I",
    ("GB", "4"): "BDS-B2I-D2",
    ("GB", "5"): "BDS-B2aI",
    ("GB", "6"): "BDS-B2aQ",
    ("GL", "1"): "GLO-L1OF",
    ("GL", "3"): "GLO-L2OF",
    ("GQ", "1"): "QZSS-L1CA",
}


def snapshot_from_gsv(sentences: list[Any], label: str, timestamp: datetime) -> SatSnapshot:
    """
    Build a SatSnapshot from a group of NMEA GSV sentences for one talker.

    sentences should be all GSV messages for one talker ID (e.g. $GPGSV)
    collected within the same epoch.  Uses pynmeagps attribute naming:
    svid_NN, elv_NN, az_NN, cno_NN (slots 01-04).
    """
    snap = SatSnapshot(timestamp=timestamp, receiver_label=label)
    for msg in sentences:
        talker    = getattr(msg, "_talker", "GN")
        gnss_id   = _TALKER_GNSS.get(talker, talker)
        sid_raw   = str(getattr(msg, "signalID", ""))
        if sid_raw in ("0", ""):   # signalID=0 = NMEA "all signals" aggregate; skip
            continue
        signal_id = _GSV_SIGNAL_NAME.get((talker, sid_raw), f"{gnss_id}-sig{sid_raw}")
        for slot in ("01", "02", "03", "04"):
            sv_id = getattr(msg, f"svid_{slot}", None)
            cno   = getattr(msg, f"cno_{slot}",  None)
            elev  = getattr(msg, f"elv_{slot}",  float("nan"))
            azim  = getattr(msg, f"az_{slot}",   float("nan"))
            if sv_id is None:
                continue
            snap.satellites.append(SatInfo(
                gnss_id, int(sv_id),
                float(cno) if cno not in (None, "") else 0.0,
                float(elev or 0), float(azim or 0),
                signal_id=signal_id,
            ))
    return snap


class GsvAccumulator:
    """
    Accumulate NMEA GSV sentences across all talkers and emit one combined
    SatSnapshot per navigation epoch.

    The F9T outputs a 1 Hz NMEA burst structured roughly as:
        GGA  ← epoch anchor (start of burst)
        GSA, GSV×N (all constellations)
        RMC, VTG, GLL, ZDA
        GGA  ← next epoch starts here

    Strategy: buffer all GSV sentences; when the next GGA arrives, emit
    the previous epoch's accumulated data as a single combined snapshot.
    Feed ALL messages (not just GSV) so the accumulator can see the GGA.
    """

    _GGA_IDS = {"GNGGA", "GPGGA", "GAGGA", "GBGGA"}

    def __init__(self, label: str, antenna_mount: str = "", mount_site: str = ""):
        self.label = label
        self.antenna_mount = antenna_mount
        self.mount_site = mount_site
        self._buffer: list[Any] = []   # GSV sentences for the current epoch

    def feed(self, msg: Any) -> SatSnapshot | None:
        """
        Feed one parsed message.  Returns a combined SatSnapshot at each
        epoch boundary (GGA), otherwise returns None.
        """
        identity = getattr(msg, "identity", "")

        if identity in self._GGA_IDS:
            # Epoch boundary — emit whatever GSV data we buffered
            if self._buffer:
                return self._emit()
            return None

        if identity.endswith("GSV"):
            self._buffer.append(msg)

        return None

    def flush(self) -> SatSnapshot | None:
        """Emit whatever has been accumulated (call at end of stream)."""
        return self._emit() if self._buffer else None

    def _emit(self) -> SatSnapshot:
        from datetime import datetime, timezone
        ts   = datetime.now(tz=timezone.utc)
        snap = SatSnapshot(timestamp=ts, receiver_label=self.label,
                           antenna_mount=self.antenna_mount, mount_site=self.mount_site)
        for msg in self._buffer:
            partial = snapshot_from_gsv([msg], self.label, ts)
            snap.satellites.extend(partial.satellites)
        self._buffer.clear()

        # The F9T may repeat the same (gnss_id, sv_id, signal_id) across
        # multiple GSV sentences within one epoch.  Deduplicate on the full
        # triple, keeping the highest C/N0 for each unique signal observation.
        # Different signals for the same satellite are intentionally kept.
        best: dict[tuple, SatInfo] = {}
        for s in snap.satellites:
            key = (s.gnss_id, s.sv_id, s.signal_id)
            if key not in best or s.cno > best[key].cno:
                best[key] = s
        snap.satellites = list(best.values())

        return snap
