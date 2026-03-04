# testAnt — GNSS Antenna Test Toolkit

Test and compare GNSS antennas using a pair of **u-blox ZED-F9T** timing receivers
and a **TICC** time-interval counter on a Raspberry Pi.  The primary goal is
evaluating multipath rejection for sub-nanosecond timing applications.

## Hardware setup

| Item | Detail |
|------|--------|
| Receivers | 2× u-blox ZED-F9T timing receivers, connected via USB-serial |
| Counter | TICC time-interval counter (chA = TOP, chB = BOT) |
| Host | Raspberry Pi |
| Role A (TOP) | Reference antenna (stable, known-good location) |
| Role B (BOT) | Antenna under test |

## Project structure

```
config/          TOML configuration (receivers.toml, run.toml)
data/            Logged CSV files (gitignored)
scripts/         Entry-point and analysis scripts
src/testant/     Library code
tests/           Unit tests
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure receivers and antennas
#    Edit config/receivers.toml  (serial ports, baud rates)
#    Edit config/run.toml        (which antenna is on which receiver)
python scripts/configure_receivers.py   # factory-reset + enable TIM-TP / RAWX

# 3. Log everything (C/N0, RAWX, TIM-TP, TICC)
python scripts/log_snr.py

# 4. Analyze
python scripts/analyze_snr.py  data/snr_<stem>.csv  --out data/<stem>
python scripts/analyze_rawx.py data/rawx_<stem>.csv --out data/<stem>
python scripts/analyze_pps.py  --ticc data/<stem>_ticc.csv \
    --timtp data/<stem>_timtp.csv --out data/<stem>_pps
```

## Roadmap

### Signal quality
- [x] SNR / satellite count logging (UBX-NAV-SAT and NMEA GSV)
- [x] Raw carrier-phase logging (UBX-RXM-RAWX)
- [x] Code-minus-carrier (CMC) multipath metric per signal
- [x] CMC vs elevation and azimuth plots (CMC skyplot, elevation scatter)
- [x] Carrier-phase lock duration and cycle-slip analysis
- [ ] Live plot: C/N0 skyplot comparison A vs B
- [ ] Multipath metric on a per-frequency basis (MP1, MP2 observable)

### PPS timing
- [x] TICC logging with picosecond resolution (int64 ref_sec + ref_ps)
- [x] UBX-TIM-TP (qErr) logging from both F9T receivers
- [x] PPS ADEV and TDEV analysis (raw and qErr-corrected, per-channel and differential)
- [x] UTC wall-clock join (host timestamp on TIM-TP receipt → unambiguous epoch alignment)
- [x] qErr debug export for gnuplot inspection (epoch-aligned corrected vs raw)
- [ ] Automated test report generation (SNR + CMC + PPS in one PDF)

### Infrastructure
- [x] TOML-based run configuration (receivers.toml + run.toml)
- [x] Receiver configuration script (factory reset, enable TIM-TP / RAWX output)
- [x] Pre-commit hook: blocks coordinates, credentials, and accidental data-file commits
