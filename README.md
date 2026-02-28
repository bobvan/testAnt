# testAnt — GNSS Antenna Test Toolkit

Test and compare GNSS antennas using a pair of **u-blox F9T** timing receivers
on a Raspberry Pi.  The primary goal is evaluating multipath rejection for
sub-nanosecond timing applications.

## Hardware setup

| Item | Detail |
|------|--------|
| Receivers | 2× u-blox F9T, connected via USB-serial |
| Host | Raspberry Pi (any model with 2 USB ports) |
| Role A | Reference antenna (stable, known-good location) |
| Role B | Antenna under test |

## Project structure

```
config/          TOML configuration (copy receivers.toml → local.toml)
data/            Logged CSV / UBX files (gitignored)
scripts/         Entry-point scripts
src/testant/     Library code
tests/           Unit tests
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure your serial ports
cp config/receivers.toml config/local.toml
$EDITOR config/local.toml   # set /dev/ttyUSB0, /dev/ttyUSB1, etc.

# 3. Log C/N0 (SNR) from both receivers
python scripts/log_snr.py

# Output: data/snr.csv with per-satellite C/N0, elevation, azimuth
```

## Roadmap

- [x] SNR / satellite count logging (UBX-NAV-SAT)
- [ ] Raw carrier-phase logging (UBX-RXM-RAWX)
- [ ] Multipath metric derivation (MP1, MP2 from dual-frequency carrier phase)
- [ ] Live plot: C/N0 skyplot comparison A vs B
- [ ] Automated test report generation
