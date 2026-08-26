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

PTBB was measured 2026-08-26 with the same metric and the same signal pairs as a
lab rooftop station: **0.363 m MP all-arcs and 0.62 mm phase noise**, against
0.789 m and 1.00 mm for the rooftop chain — and **0.099 m at C/N0 ≥50**, which
is *inside* onocoy's top-tier code threshold of 0.140 m. Full numbers in
PePPAR-Fix `docs/antenna-quality-metrics-and-timing.md`.

That is the anchor doing real work: it shows an external network's thresholds
are calibrated to what a genuine geodetic installation achieves, rather than
being arbitrary — which is what makes a score you *fail* informative instead of
dismissible.

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
3. **Add RINEX ingest** — the public-archive corpus, and the only route to the
   US labs (NIST, USNO). Highest value per unit effort here; see §4b.
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
