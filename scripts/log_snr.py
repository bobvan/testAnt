#!/usr/bin/env python3
"""
log_snr.py — Log per-satellite C/N0 from both F9T receivers to CSV.

Usage:
    python scripts/log_snr.py [--config config/local.toml] [--out data/snr.csv]

Reads NMEA GSV sentences (already output by the F9T by default) from both
receivers concurrently and writes one combined snapshot per epoch to CSV.
Also handles UBX-NAV-SAT if the receiver is configured to output it.

Stop with Ctrl-C.
"""

import argparse
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from testant.receiver import Receiver
from testant.snr import GsvAccumulator, snapshot_from_navsat
from testant.logger import SnapshotLogger


def load_config(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def reader_thread(cfg: dict, label: str, logger: SnapshotLogger, stop: threading.Event):
    port  = cfg["port"]
    baud  = cfg["baud"]
    label = cfg.get("label", label)
    acc   = GsvAccumulator(label)

    while not stop.is_set():
        try:
            with Receiver(port=port, baud=baud, label=label) as rx:
                for _raw, msg in rx:
                    if stop.is_set():
                        break
                    identity = getattr(msg, "identity", "")

                    if identity == "NAV-SAT":
                        ts   = datetime.now(tz=timezone.utc)
                        snap = snapshot_from_navsat(msg, label=label, timestamp=ts)
                    else:
                        snap = acc.feed(msg)

                    if snap is not None:
                        logger.write(snap)
                        ts = snap.timestamp
                        print(
                            f"[{ts.strftime('%H:%M:%S')}] {label:6s} "
                            f"sats={snap.count:2d} used={snap.used_count:2d} "
                            f"mean_C/N0={snap.mean_cno:.1f} dBHz",
                            flush=True,
                        )
        except Exception as exc:
            if stop.is_set():
                break
            print(f"[{label}] serial error: {exc} — reconnecting in 3s", flush=True)
            import time
            time.sleep(3)


def main():
    ap = argparse.ArgumentParser(description="Log GNSS C/N0 from two F9T receivers")
    ap.add_argument("--config", default="config/local.toml",
                    help="TOML config file (default: config/local.toml)")
    ap.add_argument("--out", default="data/snr.csv",
                    help="Output CSV path (default: data/snr.csv)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}")
        print("Copy config/receivers.toml → config/local.toml and set your port paths.")
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg  = load_config(cfg_path)
    stop = threading.Event()

    with SnapshotLogger(out_path) as logger:
        threads = []
        for key, rcv_cfg in cfg["receiver"].items():
            t = threading.Thread(
                target=reader_thread,
                args=(rcv_cfg, key, logger, stop),
                daemon=True,
            )
            threads.append(t)
            t.start()

        print(f"Logging to {out_path} — press Ctrl-C to stop.")
        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            print("\nStopping…")
            stop.set()


if __name__ == "__main__":
    main()
