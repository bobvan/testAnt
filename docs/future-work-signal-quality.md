# Future work — from antenna testing to signal-quality measurement

**Status: groundwork, not a plan of record.** Written 2026-08-26 to capture
thinking that would otherwise be lost, at a point when the work is being paused
rather than continued. Nothing here is scheduled.

**On the filing.** This sits in `testAnt` because that is where the closest
prior art and the only working rig live, and it is where the next person to
think about this would look. But the scope described below **outgrows the
repository's name**: it is no longer about antennas, and a repo called
`testAnt` is the wrong long-term home for a framework about the net signal
quality of an entire equipment chain at a mount point. Treat this document as
belonging to a repository that does not exist yet. A name should say
*signal quality*, not *antenna*.

---

## 1. What already exists, and what each generation could not answer

Three generations of tooling, each fixing the previous one's blind spot.

### `bobvan/f9tResearch` — `antMountEval.py`

Single receiver, **sequential labelled runs** (`runData.py` carries
`baseline1..5` and friends). Per-epoch features derived from the receiver's own
navigation output — no raw observations required:

    numTracked  meanCn0  fracAbove70  elevWeightedCoverage  meanAbsPrRes
    highElevCn0Std  dualBandFrac  pdop  tdop  noSlipFrac

**Strength:** cheap. Any receiver that emits NAV-SAT/NAV-SIG can produce these,
no raw-observation logging needed, no second receiver.

**Blind spot:** runs are *sequential*, so every comparison is confounded by
time — a different sky, a different ionosphere, a different satellite geometry.
Fine for gross effects (is this mount catastrophic?), useless for small ones.

Note `elevWeightedCoverage` and `meanAbsPrRes` — elevation weighting and a
code-quality proxy were both there from the start. The instincts were right;
the measurement was not yet precise enough to use them.

### `bobvan/testAnt` — this repo

**Simultaneous** dual-receiver A/B: two matched ZED-F9Ts, one host, one TICC,
reference antenna on TOP and antenna-under-test on BOT. Adds **raw**
observations (RXM-RAWX → code-minus-carrier) and **PPS timing** (TICC → ADEV /
TDEV), and `report_card.py` scores `cmc_std` *and* `adev_1s` on one page.

**Strength:** simultaneity kills the time confound. Both antennas see the same
sky, the same ionosphere, the same satellites in the same places, so everything
that is not the antenna is common-mode and cancels in the difference. This is
the only rig here that can resolve small differences.

`report_card.py` already takes `--antenna`, `--mount` and `--receiver` as
**separate fields**, which is the discipline the whole framework needs.

**Blind spot:** the metric is *relative by construction*. `cmc_std` removes only
the per-arc mean, so it retains the arc's ionospheric trend. That is harmless
in an A/B difference and **wrong** if quoted as an absolute figure or compared
against an absolute threshold. Also: it needs two receivers and a TICC, so it
cannot grade a station that already exists in the field.

### The third generation — external absolute grading (2026-08, onocoy)

A station contributed to onocoy is graded continuously by a third party with
**published thresholds**, on `RMS Code`, `RMS Phase`, cycle-slip ratio and sky
visibility. Full write-up in PePPAR-Fix
`docs/antenna-quality-metrics-and-timing.md`.

**Strength:** absolute, continuous, free, and computed by someone with no stake
in the result. It grades the *whole chain at a mount point*, which is exactly
the unit of comparison scenario 1 (below) needs.

**Blind spot:** it grades a chain, so it attributes nothing. And the
computation is undocumented — the elevation masking below is *inferred from the
numbers*, not published.

## 2. The two questions, and why they need different instruments

### Scenario 1 — absolute comparison across equipment chains

*A user reads two or more reports covering different antenna / mount / receiver
combinations and wants a quantitative comparison, understanding the confounds.*

The thing compared is the **net signal quality of the whole chain at its mount
point**. Nothing is isolated, and that is the point: this is the number that
answers "is this installation any good?"

Requirements a metric must meet to be usable this way:

- **No free constants.** Anything removed per-arc (a mean, a polynomial) makes
  the result relative and unquotable.
- **Cancel what varies with site and time but is not equipment.** Ionosphere
  above all — which means the dual-frequency MP combination, not mean-removed
  CMC.
- **Publish its mask.** A code-multipath number without its elevation cutoff is
  meaningless. Measured on one 2.4 h dataset from the same antenna:

      all arcs, unweighted    0.79 m
      C/N0 44-50 dBHz         0.31 m
      C/N0 >= 50 dBHz         0.19 m
      onocoy's published RMS  0.23 m

  A factor of four separates the same antenna from itself. Every one of those
  numbers is correct; only the masked ones are comparable to each other.
- **Normalise for the residual confounds** the user cannot remove: latitude
  (sky coverage genuinely differs — the polar hole is real), duration,
  constellation and band mix, season.

### Scenario 2 — relative attribution of a component

*A user isolates variables to find out how much a particular antenna, mount, or
receiver contributes to the total.*

**Simultaneity is the instrument.** With two chains observing the same sky at
the same instant, ionosphere, orbits, satellite clocks and troposphere are
common-mode and cancel. That is what buys the precision to see a small
difference, and it is why `testAnt` is built the way it is.

Two design rules that follow:

- **A metric with a free constant is fine here** — the constant cancels in the
  difference. `cmc_std` is legitimate for this and only this.
- **Swap to separate component from site.** Two chains a metre apart do *not*
  share a multipath environment; that is the whole point of comparing them.
  Running A-at-site-1 / B-at-site-2 and then **swapping** gives four
  configurations, enough to separate the *component* term from the *site* term.
  A single pairing cannot, and will silently attribute the site to the
  component.

## 3. The design idea that ties the two together

Onocoy bakes its four measurements into **one score** weighted for RTK. That is
right for them and wrong for a general instrument, because the same underlying
quantities matter in different proportions to different users:

| Quantity | RTK | Carrier-phase timing |
|---|---|---|
| Phase RMS | important | **dominant** — it *is* the noise on the observable the clock rides on |
| Cycle-slip rate | important | **dominant** — a slip is a phase discontinuity |
| Code RMS | **important** — drives ambiguity fixing and float convergence | weak — enters only via coarse sync and WL/MW fixing |
| Sky visibility | important | moderate — geometry and observation count |

So: **measure the components, do not bake a score.** Report the raw quantities
with their masks, and offer *profiles* — an RTK weighting, a timing weighting —
as a presentation layer over the same measurements. A single number is a
conclusion, and conclusions belong to the user.

This also resolves a tension that looked like a contradiction: our station fails
onocoy's code metric while passing phase. Under an RTK profile that is a real
deficiency; under a timing profile it is close to irrelevant. Both readings are
correct, and only a component-wise report can say so.

## 4. Broadening past the F9T

The original scope was one receiver family because that is what was on the
bench. The natural scope is **any receiver that can log raw observations**, and
the ingest problem is now largely solved across three formats by existing code:

| Format | Decoder that already works | Where |
|---|---|---|
| u-blox UBX RXM-RAWX | `analyze_rawx.py` | this repo |
| Septentrio SBF MeasEpoch | `peppar_fix.sbf_obs.decode_meas_epoch` | PePPAR-Fix |
| RTCM3 MSM | `peppar_fix.rtcm_msm_obs` | PePPAR-Fix |
| RINEX | (not yet — the obvious fourth) | — |

All three emit the same per-signal cell shape (`sv`, `sig_name`, `freq_hz`,
`pr_m`, `cp_cyc`, `cno`, `lock`), so a common metric layer above them is a small
piece of work rather than a rewrite. **RINEX ingest is the missing fourth** and
would unlock every public archive as a comparison corpus — which is the cheapest
possible route to a large scenario-1 dataset.

## 4b. The scale needs a top end, and it is already reachable

A quality scale is only interpretable against known-good anchors. Those exist,
they are free, and one has already been measured.

**National metrology institutes publish their IGS observations in real time.**
Using credentials already held, `igs-ip.net` carries PTB Braunschweig
(`PTBB00DEU0`), ORB Brussels (`BRUX00BEL0`), INRIM Torino (`IENG00ITA0`), RISE
Borås (`SPT000SWE0`) and NAOJ Mizusawa (`MIZU00JPN0`) — a *panel*, enough to
characterise the spread at the top rather than trusting one station.

**The panel has now been measured** (2026-08-26), all on a matched ~15 min
window with the same mask (`--signals l1`, i.e. GPS-L1CA + GAL-E1C only), using
`scripts/obs_quality.py` in this repo:

| Station | Site | MP all | MP 44-50 | MP >=50 | phase |
|---|---|---|---|---|---|
| **IENG** / INRIM | Torino IT, 45.0 N | **0.164 m** | **0.090 m** | 0.075 m | 0.54 mm |
| **SPT0** / RISE | Boras SE, 57.7 N | 0.256 m | 0.140 m | 0.081 m | 0.51 mm |
| **PTBB** / PTB | Braunschweig DE, 52.3 N | 0.344 m | 0.242 m | n/a | **0.40 mm** |
| a lab rooftop, non-choke-ring | Wheaton IL, 41.8 N | 0.443 m | 0.248 m | 0.171 m | 0.55 mm |

**The single most important result here: there is a 2.7x spread among the
national-lab stations themselves.**  An earlier pass used PTBB alone as *the*
anchor and concluded the rooftop station was close to national-lab quality.
PTBB turns out to be the *worst* of the three labs on code multipath, only 1.3x
better than the rooftop, while INRIM is 2.7x better.

That is the case for a panel, and it is not a hypothetical: **anchoring a scale
on one station gives a materially wrong answer.**  One anchor sets a point.

Two further findings, both of which are really about masks:

- **Latitude does not order the results** (IENG 45.0 N best, SPT0 57.7 N, PTBB
  52.3 N, rooftop 41.8 N worst).  The differences are siting and installation
  quality, not geography -- which is encouraging, because it means the metric is
  measuring the thing we want it to.
- **Window length is a third mask.**  The metric is duration-sensitive,
  materially so for phase: PTBB reads 0.40 mm over 15 min and 0.62 mm over an
  hour.  An earlier unmatched comparison (1 h vs 2.4 h) reported a 1.6x phase
  gap that is really 1.38x on matched windows.  So a figure needs its **signal
  set**, its **C/N0 or elevation cut**, *and* its **window length** before it
  means anything.  Three masks, each of which moves the answer.

**The asymmetry is the useful result.**  On code the rooftop is last by 2.7x; on
phase it is last by only 1.38x, and **all four stations sit comfortably inside
onocoy's phase threshold**.  A rooftop with a non-choke-ring antenna costs a lot
of code quality and comparatively little phase quality -- which is exactly the
split that decides whether an antenna upgrade is an RTK purchase or a timing
one.

Caveat: single ~15 min captures.  Ordering is indicative, not settled.

The anchors also calibrate the external scale: INRIM and RISE sit *inside*
onocoy's top-tier code threshold once a high-elevation mask is applied, which
shows those thresholds track what a genuine geodetic installation achieves
rather than being arbitrary — and that is what makes a score you *fail*
informative instead of dismissible.

Two cautions:

- **This is squarely scenario 1** and attributes nothing. PTBB differs in
  receiver, antenna, mount, site, installation quality *and* latitude at once.
- **A lab's published IGS stream is not necessarily the chain its UTC(k) rides
  on.** It may be decimated, filtered, or a different antenna entirely. Anchors
  are anchors, not ground truth.

**NIST and USNO are not on that caster** (nor NPL, OPMT, TWTF, KRISS, METAS).
They are IGS stations, so their data exists as post-processed RINEX from CDDIS —
which is the same **RINEX ingest** gap noted in §4. That single piece of work
would unlock both the US labs and the entire public archive as a comparison
corpus, and is probably the highest value-per-effort item in this document.

## 4c. `scripts/obs_quality.py` — the absolute metric, implemented

The scenario-1 metric from §2 now exists in this repo. It auto-detects RTCM3 MSM
or Septentrio SBF, computes the TEQC dual-frequency MP combination (ionosphere
cancelled exactly, only a per-arc constant removed, so the number is absolute)
and phase noise from the 1 s scatter of the geometry-free combination.

    ./scripts/obs_quality.py --label IENG --signals l1 IENG00ITA0.rtcm3

It prints the C/N0 breakdown **by default rather than on request**, because the
same antenna reads 0.443 m unweighted and 0.171 m at C/N0 >= 50 -- and
`--signals` exists because L2W/E5aQ carry markedly worse code multipath than
L1CA/E1C, so including them moves the answer. Both are the "publish your mask"
rule made mechanical.

`cmc_std` in `analyze_rawx.py` stays exactly as it is. It is the **scenario-2**
metric -- correct for a simultaneous A/B where the ionosphere is common-mode --
and the two are now clearly separated rather than one being mistaken for the
other.

**Known wart:** the decoders live in `bobvan/PePPAR-Fix`
(`scripts/peppar_fix/{rtcm_msm_obs,sbf_obs}.py`) and are *referenced*, not
vendored, because two copies of a bit-unpacker drift silently and hand you
plausible wrong numbers. The script locates a sibling checkout or takes
`PEPPAR_FIX_SCRIPTS=`. That is a real coupling and the honest fix is for the
shared ingest layer to become its own installable package when this work gets
its own repo -- which is another argument for §0's "this outgrows the name".

## 4d. RINEX ingest done — and NIST/USNO need no credentials at all

The "missing fourth" ingest is implemented, and it turned out to matter more
than expected: **the US national labs publish raw observations anonymously.**

    https://igs.bkg.bund.de/root_ftp/IGS/obs/<year>/<doy>/NIST00USA_R_<...>_01D_30S_MO.crx.gz

Daily 30 s multi-constellation RINEX, Hatanaka-compressed, **no login**. CDDIS
carries the same files behind an Earthdata account; BKG does not. Verified
present for one day: `NIST00USA`, `USN800USA` (USNO), `PTBB00DEU`, `BRUX00BEL`,
`IENG00ITA`, `SPT000SWE`, `WAB200CHE` (METAS) — **seven national labs, no
credentials**. That is a better panel than the authenticated caster gives, and
it reaches back through the archive rather than only forward from now.

(Credit where due: the credential-free BKG path was already written up in the
blog's `can-i-build-my-own-link-to-utc-nist` — this work rediscovered it the
long way round.)

### The cross-check that validates the whole chain

PTBB was measured **two completely independent ways**:

| path | result |
|---|---|
| our NTRIP relay → RTCM3 MSM, 1 Hz, 15 min | 0.344 m |
| anonymous BKG RINEX, 30 s, full day | **0.348 m** |

**Agreement to 1.2 %** — different transport, different format, different
sample rate, different window, different decoder. That validates the RINEX
parser, the RTCM path, and the metric's insensitivity to rate and window for
*code* multipath specifically.

### The panel, extended

| Station | MP all | 44–50 | ≥50 | source |
|---|---|---|---|---|
| IENG / INRIM | **0.164 m** | 0.090 | 0.075 | 15 min @1 Hz |
| USN8 / USNO | 0.255 m | 0.145 | 0.127 | full day @30 s |
| SPT0 / RISE | 0.256 m | 0.140 | 0.081 | 15 min @1 Hz |
| NIST / NIST | 0.313 m | 0.213 | 0.180 | full day @30 s |
| PTBB / PTB | 0.348 m | 0.254 | 0.195 | full day @30 s |
| a lab rooftop | 0.443 m | 0.248 | 0.171 | 15 min @1 Hz |

**The counterintuitive result: ranked at C/N0 ≥ 50 — the high-elevation view —
the rooftop places 4th of 6, ahead of NIST and PTBB.** It is last only on the
all-arcs figure.

That is a genuinely useful diagnosis rather than a consolation prize. It says
the rooftop's *zenith* view is competitive with national metrology institutes,
and essentially all of its deficit is at **low elevation** — horizon clutter,
which is what a rooftop in a built environment has and a properly sited geodetic
pillar does not. It also demonstrates the §2 point about masks more sharply than
any argument could: **the same six stations rank differently depending on which
cut you publish**, and both rankings are true.

Caveats: single days and single short windows; NIST/USN8/PTBB are full-day 30 s
while IENG/SPT0/rooftop are 15 min at 1 Hz. PTBB appearing in both at 1.2 %
agreement is the reason those two sets can be compared at all.

## 5. First steps, if this is ever resumed

1. **Settle the naming discipline first.** `CHOKE1` / `UFO1` name *mount points*
   in the lab database but are habitually used to mean the *antennas* on them.
   Any swap experiment makes every prior statement ambiguous. Decide whether
   names follow the mount or the hardware, and give the other one its own
   identifier. `report_card.py`'s separate `--antenna` / `--mount` /
   `--receiver` fields are the model.
2. **Promote the metric layer to absolute.** Add the dual-frequency MP
   combination alongside `cmc_std`, and make every reported figure carry its
   elevation mask. Keep `cmc_std` — it is the right metric for A/B — but stop it
   being mistakable for an absolute number.
3. ~~Add RINEX ingest~~ **done** — see §4d. Next: pull a month rather than a
   day, and add the remaining labs (BRUX, WAB2).
4. **Split the report card into components + profiles**, per §3.
5. **Only then** consider the swap experiment: it is the most expensive
   measurement here and the least useful without 1–4 in place.

## 6. Related reading

- PePPAR-Fix `docs/antenna-quality-metrics-and-timing.md` — reconciles the three
  code-minus-carrier metrics numerically, and argues which of them predict
  timing.
- PePPAR-Fix `docs/antenna-calibration-plan.md` — absolute antenna delay, and
  why the antenna-to-antenna *difference* is the term a cross-host timing budget
  actually needs.
- PePPAR-Fix `docs/two-site-sync-budget.md` — where the picosecond numbers that
  make phase RMS interesting come from.
