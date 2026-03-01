"""
rawx_logger.py — Append RawxMeas rows to a CSV file.

One row per measurement per epoch.  Context-manager pattern matches
SnapshotLogger so the calling code is symmetric.
"""

import csv
import io
import threading
from datetime import datetime
from pathlib import Path

from .rawx import RawxMeas

FIELDS = [
    "timestamp", "receiver", "antenna_mount",
    "gnss_id", "signal_id", "sv_id",
    "pseudorange_m", "carrier_phase_cy", "doppler_hz", "cno_dBHz",
    "lock_duration_ms", "pr_valid", "cp_valid", "half_cyc",
]


class RawxLogger:
    def __init__(self, path: Path):
        self.path = path
        self._file: io.TextIOWrapper | None = None
        self._writer: csv.DictWriter | None = None
        self._lock = threading.Lock()

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

    def write(self, timestamp: datetime, receiver: str,
              antenna_mount: str, meas_list: list[RawxMeas]) -> None:
        ts = timestamp.isoformat()
        with self._lock:
            for m in meas_list:
                self._writer.writerow({
                    "timestamp":       ts,
                    "receiver":        receiver,
                    "antenna_mount":   antenna_mount,
                    "gnss_id":         m.gnss_id,
                    "signal_id":       m.signal_id,
                    "sv_id":           m.sv_id,
                    "pseudorange_m":   f"{m.pseudorange:.4f}",
                    "carrier_phase_cy": f"{m.carrier_phase:.6f}",
                    "doppler_hz":      f"{m.doppler:.4f}",
                    "cno_dBHz":        f"{m.cno:.1f}",
                    "lock_duration_ms": m.lock_duration_ms,
                    "pr_valid":        int(m.pr_valid),
                    "cp_valid":        int(m.cp_valid),
                    "half_cyc":        int(m.half_cyc),
                })
            self._file.flush()
