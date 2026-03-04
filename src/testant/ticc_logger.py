"""
ticc_logger.py — Append TICC edge timestamps to a CSV file.

One row per edge (one per channel per second).  Written by a dedicated
reader thread; the lock ensures no interleaving with other threads.

Columns:
  host_timestamp  UTC wall-clock captured immediately after the serial line
                  was received.  Used to join with TIM-TP by UTC second,
                  replacing fragile GPS-offset arithmetic.  Floor to the
                  nearest second to get the epoch key (same technique as
                  f9tResearch/ticcTpAnalysis.ipynb).
  timestamp_s     TICC reference clock value (seconds since TICC boot,
                  arbitrary epoch, 1 ps resolution).
  channel         'chA' or 'chB'
"""

from __future__ import annotations

import csv
import io
import threading
from datetime import datetime, timezone
from pathlib import Path


FIELDS = ["host_timestamp", "timestamp_s", "channel"]


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

    def write(self, channel: str, timestamp_s: float,
              host_timestamp: datetime | None = None) -> None:
        if host_timestamp is None:
            host_timestamp = datetime.now(tz=timezone.utc)
        with self._lock:
            self._writer.writerow({
                "host_timestamp": host_timestamp.isoformat(),
                "timestamp_s":    f"{timestamp_s:.12f}",
                "channel":        channel,
            })
            self._file.flush()
