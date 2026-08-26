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

## 0c. What "the test rooftop" means here — it is not a generic rooftop

Every rooftop number in this document comes from **one specific installation**,
and the comparisons are only meaningful if that is stated:

- a **calibrated, survey-grade antenna** — `SFESPK6618H NONE`, with an NGS
  absolute calibration, not a consumer patch;
- mounted on the roof of a **single-storey suburban house** in Wheaton,
  Illinois (41.84 N), not a purpose-built geodetic pillar;
- with **typical suburban clutter** in the near field — an adjacent **second
  storey** of the same house, and **trees**;
- feeding a **Septentrio mosaic-T** inside a SparkPNT SXT-D.

So it is a *good antenna in an ordinary place*, which is exactly the interesting
case: the deficits measured here are attributable to **siting**, not to cheap
hardware. A generic "rooftop" with a $30 patch antenna would tell you nothing —
you could not separate the antenna from the sky it sees. That distinction is the
whole point of §2's two scenarios, and it is why this particular installation is
a useful specimen rather than an anecdote.

The national labs it is compared against sit on engineered monuments with clear
horizons and choke-ring antennas. **The gap is the price of a suburban roof**,
which is a number worth having, because most people who need good GNSS have a
suburban roof and not a monument.

## 0b. Why bother? — the case, from evidence rather than assertion

Anyone finding this file is entitled to ask why signal-quality measurement
deserves a repository rather than a scratch script. Everything below was
*demonstrated* in the course of the work recorded here, not argued for.

**1. A number without its mask is meaningless — and nobody publishes masks.**
The same antenna, same data, same day measures **0.19 m or 0.80 m** depending on
the C/N0 cut; **0.44 m or 0.69 m** depending on the window length; and different
again depending on whether L2/E5 are included. A factor of four from itself.
Every vendor quality figure, every datasheet multipath number, and every
forum comparison you will ever read omits at least one of those three.

**2. Short windows are biased, not merely noisy.** A 15 min window samples
whatever sky happens to be overhead. INRIM measured 0.164 m over 15 min and
0.278 m over a full day — not a noisy estimate of the same thing, a *different
quantity*. Two conclusions drawn earlier in this very work were wrong for
exactly this reason and had to be retracted. An instrument that makes this
mistake hard to make is worth having.

**3. One reference is not a scale.** Anchoring on PTB alone said a rooftop was
near national-lab quality. Against a seven-lab band it is 2.4x the median. A
single anchor sets a point; a panel sets a scale — and the panel turns out to be
*tight* (1.33x across seven labs, 3-9 % month-to-month within each), which is
what makes an outlier interpretable.

**4. It separates a bad site from ordinary equipment — and those cost different
money.** §4e: the intuitive fix for a rooftop is a higher elevation mask. The
measured ratio against the lab band is **flat**, so the deficit is not
elevation-selective and a mask would discard geometry for no benefit. A
competent engineer following sound reasoning would have made the wrong change.

**5. Metrics that look identical differ threefold, and the difference is
methodological.** `cmc_std` here, the TEQC MP combination, and an external
network's "RMS Code" are all code-minus-carrier and disagree by 3x on the same
antenna, entirely because of what each removes. Without something that pins the
definitions down, cross-comparison is noise dressed as measurement.

**6. Third-party quality scores are weighted for someone else's problem.** An
RTK network's score is dominated by code multipath; a carrier-phase clock cares
about phase noise and cycle slips and barely about code. The same station can
fail an external grade and be entirely fit for purpose. Only a component-wise
report can say so — hence §3's "measure components, do not bake a score".

**7. Proxies lie, and only the real quantity settles it.** Binned by C/N0 the
test rooftop's deficit looked flat with elevation, which says "a mask cannot
help". Binned by *geometric elevation* it is 1.42x overhead and 2.42x at 30-50
degrees, which says the opposite — and locates a reflector at mid elevation that
turned out to match the site's actual geometry. Same data, same metric,
different x-axis, opposite engineering decision.

**8. The reference data is free, and better than expected.** Seven national
metrology institutes publish raw observations with **no credentials at all**,
archived daily, going back years. The comparison corpus does not need to be
built or bought.

**9. It cross-validates.** PTB measured through two entirely independent paths —
our own NTRIP relay at 1 Hz in RTCM3, and anonymous BKG RINEX at 30 s — agrees
to **1.2 %**. When a measurement can be reproduced through unrelated transport,
format, rate and decoder, it has stopped being a script's output and become a
number.

The short version: **without this, you cannot tell whether your installation is
good, whether a vendor's figure means anything, or whether the obvious fix is
the right one.** With it, all three are ordinary measurements.

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

**The panel has now been measured over a month** (2026-08-26): 7 national labs
x 11 days sampled across 30 days, full-day 30 s RINEX pulled anonymously from
BKG (see §4d), all through `scripts/obs_quality.py --signals l1`.

| Lab | 0-38 | 38-44 | 44-50 | >=50 | **all** | month range |
|---|---|---|---|---|---|---|
| USNO | 0.386 | 0.272 | 0.146 | 0.124 | **0.260** | 0.254-0.269 |
| METAS | 0.274 | n/a | n/a | n/a | **0.274** | 0.260-0.285 |
| INRIM | 0.403 | 0.243 | 0.110 | 0.078 | **0.278** | 0.270-0.290 |
| ORB | 0.384 | 0.239 | 0.119 | 0.084 | **0.293** | 0.286-0.307 |
| RISE | 0.432 | 0.286 | 0.143 | 0.095 | **0.311** | 0.305-0.319 |
| NIST | 0.431 | 0.352 | 0.213 | 0.180 | **0.320** | 0.313-0.323 |
| PTB | 0.464 | 0.338 | 0.255 | 0.191 | **0.347** | 0.337-0.359 |
| *lab median* | *0.403* | *0.279* | *0.144* | *0.110* | ***0.293*** | |
| **the test rooftop** (see below) | **0.804** | **0.442** | **0.298** | **0.191** | **0.689** | 13 h @1 Hz |

**Two things this overturns, both from the same cause.** Earlier passes used
15 min windows and reported (a) a 2.7x spread among the labs and (b) that the
rooftop beat NIST and PTB at high elevation. Over full days the spread is
**1.33x**, and the rooftop is last or tied-last on every cut.

**Short windows are biased, not merely noisy.** A 15 min window samples whatever
slice of sky happens to be overhead; a full day samples all of it. INRIM
measured 0.164 m over 15 min and 0.278 m over a full day — the short figure was
not a noisy estimate of the long one, it was a different quantity. Meanwhile
full-day figures are highly reproducible: **within-lab month-to-month scatter is
only 3-9 %**. So the metric is stable *given a proper window*, and window length
belongs with the other masks as something that must be declared.

The practical consequence: the labs form a **tight calibration band**
(0.26-0.35 m) rather than a spread. That is far more useful than a single
anchor, and it is what makes a rooftop's 0.689 m unambiguously interpretable.

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

## 4e. Can this set an intelligent elevation mask? Sometimes — and here, no

The obvious use for a lab-band comparison: *you are not a national lab, your
siting is compromised, so mask the elevations where your data is bad.* The
analysis can answer that — and for the rooftop measured here the answer is
**no, a mask is the wrong lever**, which is a more useful result than a yes.

### The method

1. Measure your MP-vs-elevation curve.
2. Measure the same curve for a clean-horizon reference. The lab band is free,
   credential-free, and reproducible to 3-9 % month-to-month.
3. **Take the ratio, not the difference.** The absolute curve rises toward the
   horizon *for everyone* — that is geometry and atmosphere, not your site. What
   is yours is the part where your curve rises **faster** than a clean site's.
4. If the ratio grows toward the horizon, the deficit is elevation-selective:
   horizon clutter. A mask or a down-weight is justified, and the ratio says
   where to put it.
5. If the ratio is flat, the deficit is *not* elevation-selective and a mask
   buys nothing — look at the antenna or the near field instead.

### What the numbers say once binned by TRUE elevation

`obs_quality.py --nav` now computes geometric elevation from broadcast
ephemeris and the receiver position, so the buckets are angles rather than
signal strength. **That reverses the earlier answer**, and the reversal is the
lesson.

| MP RMS (m) | 0-15 deg | 15-30 deg | 30-50 deg | 50-91 deg |
|---|---|---|---|---|
| USNO | 0.370 | 0.237 | 0.152 | 0.118 |
| INRIM | 0.388 | 0.244 | 0.119 | 0.075 |
| NIST | 0.422 | 0.291 | 0.209 | 0.181 |
| PTB | 0.437 | 0.310 | 0.254 | 0.217 |
| *lab median* | *0.405* | *0.267* | *0.180* | *0.149* |
| **test rooftop** | **0.798** | **0.602** | **0.436** | **0.213** |
| **ratio** | **1.97x** | **2.25x** | **2.42x** | **1.42x** |

Binned by **C/N0** the ratio looked flat (2.00 / 1.58 / 2.06 / 1.74) and the
conclusion was "the deficit is not elevation-selective, so a mask cannot help".
Binned by **elevation** it is clearly not flat: **1.42x overhead, rising to
2.42x**. The C/N0 proxy had smeared the very structure being looked for —
precisely the confound flagged when it was used, now confirmed rather than
suspected.

**And the shape is not the one anyone would have guessed.** The worst band is
not the horizon: it is **30-50 degrees** (2.42x), with 0-15 degrees at 1.97x and
the zenith nearly clean at 1.42x. A pure horizon-clutter site would degrade
monotonically toward the horizon. This does not.

A mid-elevation peak is what a **reflector at moderate elevation angle** looks
like — and §0c says exactly what is there: an adjacent **second storey** and
**trees**, which from a single-storey roof subtend precisely those angles. The
measurement recovered the site's geometry without being told it.

### The mask answer, sharpened

- **A low-elevation cutoff is the wrong instrument here.** The damage is worst
  at 30-50 degrees, which cannot be masked without discarding most of the
  constellation. Masking the horizon would remove the *second*-worst band and
  leave the worst untouched.
- **A measured weighting is the right one.** `w(el) = 1/sigma^2(el)` built from
  this very table down-weights 30-50 degrees appropriately while keeping its
  geometry. This is the case where the usual advice ("just raise the mask") and
  the generic `sin^2(el)` model are *both* wrong, and only a site-specific
  measurement finds it.
- **Azimuth is the obvious next cut.** A second storey lies in one direction;
  elevation-only binning averages it around the compass and so understates the
  worst sectors. `analyze_rawx.py` already produces a CMC skyplot — the two
  should be joined.

### For timing, prefer a weighting to a mask anyway

Even when the ratio *does* climb — the case where a mask is justified — a hard
cutoff is the crude form of the right answer. Low-elevation satellites are
disproportionately valuable to a **clock** estimate because they decorrelate
clock from height; masking them cleans each remaining observation while
worsening the geometry that separates the parameter you care about. A measured
**w(el) = 1/sigma^2(el)** from your own site keeps their geometric contribution
and de-weights their noise correctly, and beats the generic `sin^2(el)` model
every processing package ships with.

The validation loop then closes inside this repo: derive the weighting from
signal quality, apply it, and check the result against `adev_1s` — which
`report_card.py` already puts on the same page as `cmc_std`.

### The confound that must be fixed before trusting any of this

**C/N0 is a proxy for elevation, and a biased one.** A lower-gain antenna reads
lower C/N0 *at every elevation*, so its ">=50 dBHz" bucket contains satellites
that a lab receiver would have placed in 44-50. That systematically mixes the
buckets and could by itself flatten a ratio that is really elevation-selective.

The flat ratio above is therefore **suggestive, not established**. Confirming it
needs binning by **true elevation**, which is entirely available: the RINEX
header carries `APPROX POSITION XYZ`, broadcast ephemeris sits in the same
anonymous BKG archive, and PePPAR-Fix `scripts/broadcast_eph.py` already
computes satellite positions. That is the single highest-value next step here —
it turns every claim in this section from *suggestive* to *measured*.

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
