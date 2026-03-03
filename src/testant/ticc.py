"""
ticc.py — Serial reader for the TAPR TICC time interval counter.

The TICC outputs one line per PPS edge:
    <seconds_since_boot>  chA|chB
e.g.
    402.342588195696 chA
    402.342588174417 chB

Timestamps have 12 decimal places (1 ps resolution).
Lines starting with '#' are comments (boot-time header); they are skipped.

Robustness notes:
  - After a power cycle the TICC enters a silent setup mode for ~5-10 s
    before starting timestamp output (like a BIOS waiting for config input).
    The port is open but produces nothing during this window — this is
    normal.  Do not treat it as an error or retry; just wait.  Subsequent
    opens (TICC already in timestamp mode) produce output immediately.
  - reset_input_buffer() on open discards stale OS-buffered lines that
    would otherwise appear as a discontinuity of several seconds.
  - Every line is matched against _LINE_RE before parsing; partial lines,
    corrupt fragments, and buffer artefacts are silently dropped.
"""

from __future__ import annotations

import re

import serial

# Exactly: digits DOT 12-digits whitespace ch followed by A or B, nothing else.
_LINE_RE = re.compile(r"^(\d+\.\d{12})\s+(ch[AB])$")


class Ticc:
    """
    Context manager that opens the TICC serial port and yields
    (channel, timestamp_s) tuples as edges arrive.

    channel     : 'chA' or 'chB'
    timestamp_s : float, seconds since TICC boot (arbitrary epoch)
    """

    def __init__(self, port: str, baud: int = 115200):
        self.port = port
        self.baud = baud
        self._ser: serial.Serial | None = None

    def __enter__(self) -> "Ticc":
        self._ser = serial.Serial(self.port, self.baud, timeout=2.0)
        self._ser.reset_input_buffer()   # discard stale buffered lines
        return self

    def __exit__(self, *_) -> None:
        if self._ser:
            self._ser.close()

    def __iter__(self):
        """Yield (channel, timestamp_s) for each valid edge line."""
        for raw in self._ser:
            line = raw.decode(errors="replace").strip()
            m = _LINE_RE.match(line)
            if not m:
                continue
            yield m.group(2), float(m.group(1))
