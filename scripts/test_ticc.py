#!/usr/bin/env python3
"""
test_ticc.py — Verify TICC connectivity and display live edge timestamps.

Prints paired chA/chB lines and the A−B interval for each second.
Stop with Ctrl-C.

Usage:
    python scripts/test_ticc.py [--port /dev/ticc] [--count 20]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from testant.ticc import Ticc


def main():
    ap = argparse.ArgumentParser(description="Test TICC serial connection")
    ap.add_argument("--port",  default="/dev/ticc",
                    help="Serial port (default: /dev/ticc)")
    ap.add_argument("--count", type=int, default=20,
                    help="Number of seconds to display (default: 20)")
    args = ap.parse_args()

    port = args.port
    if not Path(port).exists():
        by_id = Path("/dev/serial/by-id/"
                     "usb-Arduino__www.arduino.cc__0042_"
                     "95037323535351803130-if00")
        if by_id.exists():
            print(f"  {port} not found; using {by_id}")
            port = str(by_id)
        else:
            print(f"ERROR: {port} not found and by-id fallback missing.")
            sys.exit(1)

    print(f"Opening {port} …  (Ctrl-C to stop)\n")
    print(f"{'Second':>8s}  {'chA (s)':>18s}  {'chB (s)':>18s}  {'A−B (ns)':>10s}")
    print("-" * 64)

    pending: dict[int, dict] = {}   # integer_second → {chA: ts, chB: ts}
    n_printed = 0

    try:
        with Ticc(port) as ticc:
            for ch, ts in ticc:
                sec = int(ts)
                pending.setdefault(sec, {})[ch] = ts

                row = pending[sec]
                if "chA" in row and "chB" in row:
                    diff_ns = (row["chA"] - row["chB"]) * 1e9
                    print(f"{sec:>8d}  {row['chA']:>18.12f}  {row['chB']:>18.12f}"
                          f"  {diff_ns:>+10.3f} ns")
                    del pending[sec]
                    n_printed += 1
                    if n_printed >= args.count:
                        break

                # Evict stale half-pairs (edge arrived without partner)
                stale = [s for s in pending if s < sec - 2]
                for s in stale:
                    print(f"{s:>8d}  (only {list(pending[s])[0]} arrived)")
                    del pending[s]

    except KeyboardInterrupt:
        pass

    print(f"\n{n_printed} complete pairs displayed.")


if __name__ == "__main__":
    main()
