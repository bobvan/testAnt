"""
logger.py — Append SatSnapshot rows to a CSV file.

Each satellite observation becomes one row, making it easy to load
the file later with pandas for analysis and plotting.
"""

import csv
import io
from datetime import datetime
from pathlib import Path

from .snr import SatSnapshot

FIELDS = [
    "timestamp", "receiver", "gnss_id", "sv_id",
    "cno_dBHz", "elev_deg", "azim_deg", "used",
]


class SnapshotLogger:
    def __init__(self, path: Path):
        self.path = path
        self._file: io.TextIOWrapper | None = None
        self._writer: csv.DictWriter | None = None

    def __enter__(self):
        new_file = not self.path.exists()
        self._file = open(self.path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        if new_file:
            self._writer.writeheader()
        return self

    def __exit__(self, *_):
        if self._file:
            self._file.close()

    def write(self, snap: SatSnapshot) -> None:
        ts = snap.timestamp.isoformat()
        for sat in snap.satellites:
            self._writer.writerow({
                "timestamp":  ts,
                "receiver":   snap.receiver_label,
                "gnss_id":    sat.gnss_id,
                "sv_id":      sat.sv_id,
                "cno_dBHz":   f"{sat.cno:.1f}",
                "elev_deg":   f"{sat.elev:.1f}",
                "azim_deg":   f"{sat.azim:.1f}",
                "used":       int(sat.used),
            })
        self._file.flush()
