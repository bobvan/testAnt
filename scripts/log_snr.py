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
from testant.rawx import snapshot_from_rawx
from testant.rawx_logger import RawxLogger
from testant.timtp_logger import TimtpLogger
from testant.ticc import Ticc
from testant.ticc_logger import TiccLogger


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


def reader_thread(cfg: dict, logger: SnapshotLogger, rawx_logger: RawxLogger,
                  timtp_logger: TimtpLogger, stop: threading.Event):
    receiver      = cfg["receiver"]
    port          = cfg["port"]
    baud          = cfg["baud"]
    antenna_mount = cfg["antenna_mount"]
    mount_site    = cfg["mount_site"]
    acc         = GsvAccumulator(receiver, antenna_mount=antenna_mount, mount_site=mount_site)
    navsat_used: dict[tuple[str, int], bool] = {}  # (gnss_id, sv_id) -> used

    while not stop.is_set():
        try:
            with Receiver(port=port, baud=baud, label=receiver) as rx:
                for _raw, msg in rx:
                    if stop.is_set():
                        break
                    identity = getattr(msg, "identity", "")

                    if identity == "NAV-SAT":
                        # Extract used flags; annotate GSV snapshots rather than
                        # logging NAV-SAT directly (NAV-SAT lacks signal_id).
                        ts      = datetime.now(tz=timezone.utc)
                        nav_snap = snapshot_from_navsat(msg, label=receiver, timestamp=ts,
                                                        antenna_mount=antenna_mount,
                                                        mount_site=mount_site)
                        navsat_used = {(s.gnss_id, s.sv_id): s.used
                                       for s in nav_snap.satellites}
                        snap = None
                    elif identity == "RXM-RAWX":
                        ts   = datetime.now(tz=timezone.utc)
                        meas = snapshot_from_rawx(msg, label=receiver,
                                                  antenna_mount=antenna_mount,
                                                  timestamp=ts)
                        rawx_logger.write(ts, receiver, antenna_mount, meas)
                        snap = None
                    elif identity == "TIM-TP":
                        ts = datetime.now(tz=timezone.utc)
                        timtp_logger.write(
                            timestamp = ts,
                            receiver  = receiver,
                            qerr_ps   = int(getattr(msg, "qErr",  0)),
                            tow_ms    = int(getattr(msg, "towMS", 0)),
                            week      = int(getattr(msg, "week",  0)),
                        )
                        snap = None
                    else:
                        snap = acc.feed(msg)
                        if snap is not None and navsat_used:
                            # Annotate with used flags from the latest NAV-SAT message
                            for sat in snap.satellites:
                                sat.used = navsat_used.get((sat.gnss_id, sat.sv_id), False)

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


def ticc_thread(cfg: dict, logger: TiccLogger, stop: threading.Event):
    port = cfg["port"]
    baud = cfg["baud"]
    while not stop.is_set():
        try:
            with Ticc(port=port, baud=baud) as ticc:
                for ch, ts in ticc:
                    host_ts = datetime.now(tz=timezone.utc)
                    if stop.is_set():
                        break
                    logger.write(ch, ts, host_ts)
        except Exception as exc:
            if stop.is_set():
                break
            print(f"[TICC] error: {exc} — reconnecting in 3s", flush=True)
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

    out_path   = Path(args.out)
    stem       = str(out_path.with_suffix(""))
    rawx_path  = Path(stem + "_rawx.csv")
    timtp_path = Path(stem + "_timtp.csv")
    ticc_path  = Path(stem + "_ticc.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ticc_cfg = receivers.get("ticc")

    stop = threading.Event()
    with (SnapshotLogger(out_path)  as logger,
          RawxLogger(rawx_path)     as rawx_logger,
          TimtpLogger(timtp_path)   as timtp_logger,
          TiccLogger(ticc_path)     as ticc_logger):
        threads = []
        for cfg in thread_cfgs:
            t = threading.Thread(
                target=reader_thread,
                args=(cfg, logger, rawx_logger, timtp_logger, stop),
                daemon=True,
            )
            threads.append(t)
            t.start()

        if ticc_cfg:
            t = threading.Thread(
                target=ticc_thread,
                args=(ticc_cfg, ticc_logger, stop),
                daemon=True,
            )
            threads.append(t)
            t.start()
        else:
            print("(No [ticc] in receivers.toml — TICC logging skipped)")

        print(f"Logging C/N0  → {out_path}")
        print(f"Logging RAWX  → {rawx_path}")
        print(f"Logging TIM-TP→ {timtp_path}")
        if ticc_cfg:
            print(f"Logging TICC  → {ticc_path}")
        print("Press Ctrl-C to stop.")
        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            print("\nStopping…")
            stop.set()


if __name__ == "__main__":
    main()
