#!/usr/bin/env python3
"""
log_snr.py — Log per-satellite C/N0 from both F9T receivers to CSV.

Usage:
    python scripts/log_snr.py [--receivers config/receivers.toml] \
                               [--run config/run.toml] \
                               [--out data/snr.csv]

Loads stable hardware config from receivers.toml and per-run antenna
assignments from run.toml, then logs NMEA GSV (or UBX-NAV-SAT) from
each receiver to a combined CSV.

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


def load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def build_thread_configs(receivers: dict, run: dict) -> list[dict]:
    """
    Join receivers.toml hardware entries with run.toml antenna assignments.

    Returns one dict per antenna entry, containing everything reader_thread needs:
        port, baud, receiver (TOP/BOT), antenna_mount, mount_site
    """
    hw = receivers.get("receiver", {})
    result = []
    for ant_key, ant in run.get("antenna", {}).items():
        rx_name = ant["receiver"]
        if rx_name not in hw:
            print(f"ERROR: run.toml antenna.{ant_key} references receiver '{rx_name}' "
                  f"which is not in receivers.toml")
            sys.exit(1)
        result.append({
            "receiver":     rx_name,
            "port":         hw[rx_name]["port"],
            "baud":         hw[rx_name]["baud"],
            "antenna_mount": ant.get("label", ant_key),
            "mount_site":   ant.get("mount_site", ""),
        })
    return result


def reader_thread(cfg: dict, logger: SnapshotLogger, stop: threading.Event):
    receiver     = cfg["receiver"]
    port         = cfg["port"]
    baud         = cfg["baud"]
    antenna_mount = cfg["antenna_mount"]
    mount_site   = cfg["mount_site"]
    acc          = GsvAccumulator(receiver, antenna_mount=antenna_mount, mount_site=mount_site)

    while not stop.is_set():
        try:
            with Receiver(port=port, baud=baud, label=receiver) as rx:
                for _raw, msg in rx:
                    if stop.is_set():
                        break
                    identity = getattr(msg, "identity", "")

                    if identity == "NAV-SAT":
                        ts   = datetime.now(tz=timezone.utc)
                        snap = snapshot_from_navsat(msg, label=receiver, timestamp=ts,
                                                    antenna_mount=antenna_mount,
                                                    mount_site=mount_site)
                    else:
                        snap = acc.feed(msg)

                    if snap is not None:
                        logger.write(snap)
                        ts  = snap.timestamp
                        tag = f"{receiver}/{antenna_mount}"
                        print(
                            f"[{ts.strftime('%H:%M:%S')}] {tag:24s} "
                            f"sats={snap.count:2d} used={snap.used_count:2d} "
                            f"mean_C/N0={snap.mean_cno:.1f} dBHz",
                            flush=True,
                        )
        except Exception as exc:
            if stop.is_set():
                break
            print(f"[{receiver}] serial error: {exc} — reconnecting in 3s", flush=True)
            import time
            time.sleep(3)


def main():
    ap = argparse.ArgumentParser(description="Log GNSS C/N0 from two F9T receivers")
    ap.add_argument("--receivers", default="config/receivers.toml",
                    help="Receiver hardware config (default: config/receivers.toml)")
    ap.add_argument("--run", default="config/run.toml",
                    help="Per-run antenna assignment config (default: config/run.toml)")
    ap.add_argument("--out", default="data/snr.csv",
                    help="Output CSV path (default: data/snr.csv)")
    args = ap.parse_args()

    receivers_path = Path(args.receivers)
    run_path       = Path(args.run)

    for p in (receivers_path, run_path):
        if not p.exists():
            print(f"Config not found: {p}")
            if p == run_path:
                print("Copy config/run.toml, edit antenna labels/sites/receivers, then re-run.")
            sys.exit(1)

    receivers = load_toml(receivers_path)
    run       = load_toml(run_path)
    thread_cfgs = build_thread_configs(receivers, run)

    desc = run.get("run", {}).get("description", "")
    if desc:
        print(f"Run: {desc}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stop = threading.Event()
    with SnapshotLogger(out_path) as logger:
        threads = []
        for cfg in thread_cfgs:
            t = threading.Thread(
                target=reader_thread,
                args=(cfg, logger, stop),
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
