#!/usr/bin/env python3
"""
configure_receivers.py — Factory reset and configure both ZED-F9T receivers.

Hardware:
  REF — ZED-F9T-20B (FWVER=TIM 2.25): GPS, Galileo, BeiDou, SBAS, QZSS, NavIC  (no GLONASS)
  DUT — ZED-F9T     (FWVER=TIM 2.20): GPS, GLONASS, Galileo, BeiDou, SBAS, QZSS, NavIC

Both receivers are configured to the same constellation set — the smallest common
subset — so that neither has a constellation advantage over the other:

    ON : GPS (L1C/A), Galileo (E1), BeiDou (B1I)
    OFF: GLONASS, SBAS, QZSS (Japanese), NavIC (Indian)

GLONASS is explicitly disabled on both.  On REF (no GLONASS hardware) the
disable message is expected to NAK and that is treated as benign — the receiver
has no GLONASS to enable anyway.  On DUT, whose ROM defaults have GLONASS on,
the disable is required.

NOTE: L2 signal keys (GPS_L2C, GAL_E5B, BDS_B2, GLO_L2) are firmware-locked
and NAK'd on all known TIM firmware variants.

Steps performed on each receiver:
  1. Factory reset  — CFG-CFG (clear + save ROM defaults to flash) + CFG-RST cold start
  2. Reconnect      — wait for USB device to reappear
  3. Configure      — CFG-VALSET (layers=RAM+BBR+Flash) for signal enables

Usage:
    python scripts/configure_receivers.py [--config config/local.toml]
"""

import argparse
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import tomllib
except ImportError:
    import tomli as tomllib

import serial
from pyubx2 import UBXMessage, UBXReader, SET, POLL

# ── timing ─────────────────────────────────────────────────────────── #
ACK_TIMEOUT = 5.0    # seconds to wait for ACK-ACK after a config command
RESET_WAIT  = 10.0   # seconds to wait for device to restart after CFG-RST

# ── CFG-CFG mask helpers ────────────────────────────────────────────── #
# X4 fields in pyubx2 require bytes (little-endian).
# 0xFFFF covers all standard config sections (ioPort, msgConf, navConf, …).
_MASK_ALL  = struct.pack("<I", 0x0000_FFFF)
_MASK_NONE = struct.pack("<I", 0x0000_0000)

# ── constellation / signal configuration ───────────────────────────── #
# layers=7 → RAM(1) | BBR(2) | Flash(4) — survives power cycling.
#
# Target: GPS + GAL + BDS on both receivers (smallest common subset).
# GLONASS is off even on DUT, which supports it, so that constellation
# composition is identical and neither receiver has a satellite-count advantage.
#
# NOTE: L2 signal keys (GPS_L2C, GAL_E5B, BDS_B2, GLO_L2) are NAK'd on all
# known TIM firmware variants — firmware-locked, L1-only signals are effective.
LAYERS = 7

SIGNAL_CONFIG = [
    # GPS ── enable system + L1C/A
    ("CFG_SIGNAL_GPS_ENA",       1),
    ("CFG_SIGNAL_GPS_L1CA_ENA",  1),
    # Galileo ── enable system + E1
    ("CFG_SIGNAL_GAL_ENA",       1),
    ("CFG_SIGNAL_GAL_E1_ENA",    1),
    # BeiDou ── enable system + B1I
    ("CFG_SIGNAL_BDS_ENA",       1),
    ("CFG_SIGNAL_BDS_B1_ENA",    1),
    # SBAS ── off (augmentation layer, not a science constellation)
    ("CFG_SIGNAL_SBAS_ENA",      0),
    # QZSS ── off (Japanese regional)
    ("CFG_SIGNAL_QZSS_ENA",      0),
    # NavIC / IRNSS ── off (Indian regional)
    ("CFG_SIGNAL_NAVIC_ENA",     0),
]

# Sent as a separate CFG-VALSET to ensure GLONASS is off on both receivers.
# DUT (ZED-F9T) has GLONASS enabled in its ROM defaults, so this is required
# there.  REF (ZED-F9T-20B) lacks GLONASS hardware and will NAK these keys —
# that NAK is benign and is logged but does not abort configuration.
GLO_DISABLE_CONFIG = [
    ("CFG_SIGNAL_GLO_ENA",    0),
    ("CFG_SIGNAL_GLO_L1_ENA", 0),
]


# ── serial helpers ──────────────────────────────────────────────────── #

def open_port(port: str, baud: int) -> tuple[serial.Serial, UBXReader]:
    ser = serial.Serial(port, baud, timeout=1)
    ser.reset_input_buffer()
    rdr = UBXReader(ser, protfilter=7, quitonerror=0)
    return ser, rdr


def wait_ack(rdr: UBXReader, timeout: float = ACK_TIMEOUT) -> bool:
    """Read messages until ACK-ACK / ACK-NAK or timeout."""
    deadline = time.monotonic() + timeout
    for _, msg in rdr:
        if time.monotonic() > deadline:
            return False
        ident = getattr(msg, "identity", "")
        if ident == "ACK-ACK":
            return True
        if ident == "ACK-NAK":
            print("    (NAK)")
            return False
    return False


def send_cfg(ser: serial.Serial, rdr: UBXReader, msg: UBXMessage, label: str) -> bool:
    ser.write(msg.serialize())
    ok = wait_ack(rdr)
    print(f"    {label}: {'OK' if ok else 'FAIL'}")
    return ok


# ── factory reset ───────────────────────────────────────────────────── #

def factory_reset(ser: serial.Serial) -> None:
    """
    Clear all configuration from RAM, BBR, and Flash, then trigger a cold-start
    software reset.  The caller must close the port immediately after this returns
    and wait RESET_WAIT seconds before reconnecting.
    """
    # 1. Clear all config sections and save the cleared state to flash.
    #    After restart the receiver will use its ROM defaults.
    cfg_clear = UBXMessage("CFG", "CFG-CFG", SET,
        clearMask=_MASK_ALL,    # wipe all config sections from RAM/BBR
        saveMask=_MASK_ALL,     # write cleared (= ROM default) state to flash
        loadMask=_MASK_NONE,    # don't load from flash; start from ROM
        devBBR=1, devFlash=1, devEEPROM=1, reserved1=0, devSpiFlash=1,
    )
    ser.write(cfg_clear.serialize())
    time.sleep(0.5)

    # 2. Cold start: clear all navigation data (almanac, ephemeris, …)
    #    and trigger a controlled software reset.
    cfg_rst = UBXMessage("CFG", "CFG-RST", SET,
        navBbrMask=0xFFFF,   # clear all BBR nav data (true cold start)
        resetMode=0x01,      # controlled software reset
        reserved0=0,
    )
    ser.write(cfg_rst.serialize())


# ── constellation configuration ─────────────────────────────────────── #

def configure_signals(ser: serial.Serial, rdr: UBXReader) -> bool:
    """Apply SIGNAL_CONFIG then GLO_DISABLE_CONFIG via CFG-VALSET."""
    msg = UBXMessage.config_set(
        layers=LAYERS,
        transaction=0,
        cfgData=SIGNAL_CONFIG,
    )
    ok = send_cfg(ser, rdr, msg, "CFG-VALSET signal config")

    # Disable GLONASS explicitly.  A NAK here is expected on REF (no GLO
    # hardware) and is treated as benign — GLO was already off.
    glo_msg = UBXMessage.config_set(
        layers=LAYERS,
        transaction=0,
        cfgData=GLO_DISABLE_CONFIG,
    )
    glo_ok = send_cfg(ser, rdr, glo_msg, "CFG-VALSET GLO disable")
    if not glo_ok:
        print("    (GLO disable NAK'd — no GLONASS hardware on this receiver, GLO already off)")

    return ok


# ── per-receiver orchestration ──────────────────────────────────────── #

def configure_one(label: str, port: str, baud: int) -> None:
    bar = "─" * 52
    print(f"\n{bar}")
    print(f"  {label}  ·  {port}")
    print(bar)

    # ── Step 1: factory reset ──────────────────────────────────────── #
    print("\n[1] Factory reset")
    try:
        ser, _ = open_port(port, baud)
        factory_reset(ser)
        ser.close()
        print(f"    Reset sent — waiting {RESET_WAIT:.0f}s for restart …")
    except serial.SerialException as exc:
        print(f"    ERROR: {exc}")
        return

    time.sleep(RESET_WAIT)

    # ── Step 2: reconnect ──────────────────────────────────────────── #
    print("\n[2] Reconnect")
    ser = None
    for attempt in range(6):
        try:
            ser, rdr = open_port(port, baud)
            print(f"    Connected (attempt {attempt + 1})")
            break
        except serial.SerialException as exc:
            print(f"    Attempt {attempt + 1}/6 failed — retrying in 3 s …")
            time.sleep(3)

    if ser is None:
        print("    ERROR: could not reconnect after reset")
        return

    # ── Step 3: configure signals ──────────────────────────────────── #
    print("\n[3] Configure constellations")
    for key, val in SIGNAL_CONFIG + GLO_DISABLE_CONFIG:
        state = "ON " if val else "OFF"
        print(f"    {state}  {key}")

    ok = configure_signals(ser, rdr)

    ser.close()
    print(f"\n  {'Done.' if ok else 'FAILED — check receiver.'}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Factory reset and configure ZED-F9T receivers for antenna testing"
    )
    ap.add_argument("--config", default="config/local.toml",
                    help="TOML config file (default: config/local.toml)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}")
        sys.exit(1)

    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)

    for key, rcv in cfg["receiver"].items():
        configure_one(
            label=rcv.get("label", key),
            port=rcv["port"],
            baud=rcv["baud"],
        )

    print("\nAll receivers configured.")


if __name__ == "__main__":
    main()
