"""
receiver.py — Serial connection and message stream for a u-blox F9T.

Wraps pyubx2's UBXReader to yield parsed UBX and NMEA messages from
the receiver's serial port.  Each call to read() blocks until one
complete message is available.
"""

import serial
from pyubx2 import UBXReader, ERR_IGNORE


class Receiver:
    """Open a serial port to one F9T and iterate over parsed messages."""

    def __init__(self, port: str, baud: int = 115200, label: str = ""):
        self.port = port
        self.baud = baud
        self.label = label
        self._ser: serial.Serial | None = None
        self._reader: UBXReader | None = None

    # ------------------------------------------------------------------ #
    # Context-manager interface                                            #
    # ------------------------------------------------------------------ #

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def open(self):
        self._ser = serial.Serial(self.port, self.baud, timeout=1)
        # protfilter=7  →  accept UBX + NMEA + RTCM3
        self._reader = UBXReader(self._ser, protfilter=7, errhandler=ERR_IGNORE)

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()

    def __iter__(self):
        """Yield (raw_bytes, parsed_message) tuples indefinitely."""
        if self._reader is None:
            raise RuntimeError("Receiver not open — use as context manager or call open()")
        for raw, parsed in self._reader:
            if parsed is not None:
                yield raw, parsed

    def send(self, msg) -> None:
        """Write a UBXMessage (e.g. a CFG poll) to the receiver."""
        if self._ser is None or not self._ser.is_open:
            raise RuntimeError("Receiver not open")
        self._ser.write(msg.serialize())
