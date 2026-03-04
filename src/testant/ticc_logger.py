"""
ticc_logger.py — Append TICC edge timestamps to a CSV file.

One row per edge (one per channel per second).  Written by a dedicated
reader thread; the lock ensures no interleaving with other threads.

Columns:
  host_timestamp  UTC wall-clock captured immediately after the serial line
                  was received.  Used to join with TIM-TP by UTC second,
                  replacing fragile GPS-offset arithmetic.  Floor to the
                  nearest second to get the epoch key.
  ref_sec         Integer seconds since TICC boot (arbitrary epoch).
  ref_ps          Picoseconds 0..999_999_999_999.  Stored as integer to
                  preserve ps resolution without float64 precision loss.
                  11-digit firmware → 10 ps resolution (last digit = 0).
                  12-digit firmware →  1 ps resolution.
  channel         'chA' or 'chB'
"""

from __future__ import annotations

import csv
import io
import threading
from datetime import datetime, timezone
from pathlib import Path


FIELDS = ["host_timestamp", "ref_sec", "ref_ps", "channel"]


class TiccLogger:
    def __init__(self, path: Path):
        self.path = path
        self._file: io.TextIOWrapper | None = None
        self._writer: csv.DictWriter | None = None
        self._lock = threading.Lock()

    def __enter__(self) -> "TiccLogger":
        new_file = not self.path.exists()
        self._file = open(self.path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        if new_file:
            self._writer.writeheader()
        return self

    def __exit__(self, *_) -> None:
        if self._file:
            self._file.close()

    def write(self, channel: str, ref_sec: int, ref_ps: int,
              host_timestamp: datetime | None = None) -> None:
        if host_timestamp is None:
            host_timestamp = datetime.now(tz=timezone.utc)
        with self._lock:
            self._writer.writerow({
                "host_timestamp": host_timestamp.isoformat(),
                "ref_sec":        ref_sec,
                "ref_ps":         ref_ps,
                "channel":        channel,
            })
            self._file.flush()
