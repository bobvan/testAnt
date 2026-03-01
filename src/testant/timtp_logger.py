"""
timtp_logger.py — Append UBX-TIM-TP fields to a CSV file.

One row per timepulse per receiver (1 Hz).  The key field is qErr (ps),
which is the predicted quantization error of the *following* PPS edge and
is used to correct TICC timestamps in post-processing.
"""

from __future__ import annotations

import csv
import io
import threading
from datetime import datetime
from pathlib import Path


FIELDS = ["timestamp", "receiver", "qerr_ps", "tow_ms", "week"]


class TimtpLogger:
    def __init__(self, path: Path):
        self.path = path
        self._file: io.TextIOWrapper | None = None
        self._writer: csv.DictWriter | None = None
        self._lock = threading.Lock()

    def __enter__(self) -> "TimtpLogger":
        new_file = not self.path.exists()
        self._file = open(self.path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        if new_file:
            self._writer.writeheader()
        return self

    def __exit__(self, *_) -> None:
        if self._file:
            self._file.close()

    def write(self, timestamp: datetime, receiver: str,
              qerr_ps: int, tow_ms: int, week: int) -> None:
        with self._lock:
            self._writer.writerow({
                "timestamp": timestamp.isoformat(),
                "receiver":  receiver,
                "qerr_ps":   qerr_ps,
                "tow_ms":    tow_ms,
                "week":      week,
            })
            self._file.flush()
