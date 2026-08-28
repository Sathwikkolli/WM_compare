# Informed detection: does an aligned original recover watermarks that blind detection loses?

**Status:** planned, not yet run. Predictions registered below.
**Code:** `informed/` (not yet written)
**Predecessors:** `2026-08-05_align-method-bakeoff`, `2026-08-10_aware-detection-ab`,
`2026-08-17_attack-damage-control`, `2026-08-18_frame-align-null`

---

## The question

We hold the unwatermarked original. Blind detection ignores it. Subtracting an
aligned original cancels *host interference* — the dominant noise term for a
blind detector — which should leave the watermark exposed rather than buried.

> **When blind detection fails on audio that is still usable, does informed
> detection recover the watermark?**

The qualifier "still usable" is the whole design. See below.

---

## Why the obvious test cases are the wrong ones

`2026-08-10_aware-detection-ab` found two conditions where AWARE fails:
`highpass_0.2` (0/20) and `quantize_8lvl` (1/20). Those look like the natural
test cases for this experiment. **They are not.**

`2026-08-17_attack-damage-control` established why: at those settings the audio
sits at PESQ **1.32** and **1.04** against a clean 4.64. It is destroyed. Its
own conclusion was that the A/B's failures are *"destroyed audio, not a fragile
watermark"*, and that AWARE **outlives usability** — audio crosses PESQ 2.0 at
~850 Hz high-pass while AWARE only loses the mark at 2500 Hz.

Recovering a watermark from audio nobody would ever use demonstrates nothing.
An attacker who destroys the audio has already lost, because they wanted to use
it.

**So the experiment must first find attacks that break detection while the audio
remains usable.** That is Phase A.

---

## The attacker's constraint, and how we measure it

The real adversary problem is constrained:

> Remove the watermark, **subject to** the audio staying good enough to use.

For each attack, sweeping strength traces a monotone trade-off: stronger attack,
worse quality, weaker detection. The quantity that matters is the crossing point.

For each attack we find **s\*** — the strongest setting that keeps quality above
a usability floor — and ask whether detection survives at s\*.

| result at s* | interpretation |
|---|---|
| watermark still detected | **secure** against this attack: the attacker must wreck the audio to win |
| watermark gone | **a real vulnerability**: usable audio, no detection |

That is a *security margin*, expressed in the attacker's own currency.

### Quality goes on the x-axis, not attack strength

Attack strengths are not comparable across attacks. "+4 dB music SNR",
"reverb 0.3" and "8 quantisation levels" share no units. **Quality does.**

So the primary artefact is a **detection-vs-quality** curve, one per attack,
traced by sweeping strength, all on one chart. A vertical line at the usability
floor then separates vulnerabilities (curves already collapsed to its right)
from attacks where the watermark outlives usability.

---

## Quality metric: DNSMOS primary, PESQ secondary

**This choice decides whether the screen works, so it is recorded here.**

PESQ measures *fidelity to the original*. An attacker does not need fidelity —
they need the result to **sound acceptable**. The two come apart badly:

- Speech mixed with a music bed has poor PESQ (it differs from the clean speech)
  but sounds completely normal. It is a podcast. It is fully usable.
- Anything that re-synthesises audio — neural codecs, vocoders, voice conversion
  — scores badly on PESQ while sounding fine.

Screening on PESQ would file the music attack under "destroyed audio" alongside
`highpass_0.2` and discard the most promising vulnerability we have.

Therefore: **DNSMOS (no-reference) is the primary metric.** It is already a
column in the Emilia manifest, so it costs nothing. PESQ is reported alongside
for the attacks where fidelity genuinely is the point (noise, filtering,
quantisation), and the disagreement between the two is itself recorded.

**Usability floor: DNSMOS ≥ 3.0** (primary), with results also reported at
**≥ 3.5** (strict). Detection threshold stays at **0.50** per
`results/THRESHOLD_DECISION.md`.

---

## The lead candidate is already in the repo

`real_attacks_summary.csv` (from `real_attacks_experiment.py`) swept a music bed
against watermarked speech:

| music SNR | confidence | bit accuracy | detected @ 0.50 |
|---|---|---|---|
| +10 dB | 0.897 | 1.000 | yes |
| +6 dB | 0.613 | 0.925 | yes |
| +5 dB | 0.506 | 0.925 | marginal |
| **+4 dB** | **0.402** | **0.925** | **no** |
| +3 dB | 0.292 | 0.925 | no |

At +4 dB detection fails **while bit accuracy is still 92.5%** — the watermark is
present and readable, the detector simply cannot clear threshold through the
interference. And the audio is an ordinary speech-over-music mix.

This is the target case: realistic, quality intact, detection failing, watermark
surviving. It is also the client-relevant one — the Metapyxl chain carries a
music bed, recorded as an untested limitation in `align_bench/README.md`.

**What is missing:** that sweep recorded **no quality numbers at all**. We know
where the watermark dies; we do not know where the audio stops being usable, so
the crossing point cannot be located yet. Fixing that is the first task.

---

## Design

### Phase A — attack boundary screening

For each of 50 Emilia clips, each attack, each strength:

1. Embed AWARE → `wm`; keep `org` and the known message bits
2. Apply attack at strength *s* → `wm_attacked`
3. Measure **DNSMOS** and **PESQ**
4. Run blind detection (unchanged cascade) → confidence, bit accuracy

Output: detection-vs-quality curves, and for each attack the value of **s\***
and whether detection survives there.

**Attacks to screen** (superset; the screen decides what survives into Phase B):

| family | why it is a candidate |
|---|---|
| music bed (SNR sweep) | the known case above |
| background noise — babble, factory, machinegun | `cascade/noises/` already present; additive, quality-preserving at moderate SNR |
| reverb | `reverb_test_emilia.py`; realistic room/recording effect |
| re-recording | `rerecord/`; the acoustic path, entirely realistic |
| denoising / enhancement | a denoiser may strip the watermark *as noise* while **raising** perceived quality |
| neural codec round-trip | re-synthesis; high perceptual quality, waveform rebuilt |
| mastering chain | `fsss/exp_v12_metapxyl_compare.py`; the client's real processing |
| platform re-encode | upload chains re-compress by default |

### Phase B — informed vs blind

Only on attacks that Phase A shows failing **inside** the usable region.

Per clip and selected attack setting:

1. **Blind arm** — cascade detect on `wm_attacked` (this is the baseline)
2. **Informed arm**
   a. Align `wm_attacked` to `org` with **`gcc_phat`** — see below
   b. Shift to the recovered offset
   c. Gain-match: `a = <wm_attacked, org> / <org, org>`
   d. Residual: `w_observed = wm_attacked − a·org`
   e. Reference pattern: `w_clean = wm − org` (exact; we made the file)
   f. Statistic: `ρ = corr(w_clean, w_observed)`
3. Compare arms at **matched false-positive rates**, not raw accuracy

**`gcc_phat`, not `aof`.** Subtraction needs sample-exact alignment. `aof`
quantises to a ~7.7 ms grid (~123 samples); at that error the host does not
cancel, it smears and doubles. `2026-08-18_frame-align-null` §5 measured
`gcc_phat` as sample-exact whenever it succeeds (94.7% at 500 ms, 100% at 2 s).
That finding is the reason this experiment is possible at all. Rows where
`gcc_phat` does not report a trusted alignment are excluded and **counted**.

**Why a correlation statistic and not the existing detector.** AudioSeal's and
AWARE's detectors are networks trained on watermarked *speech*. A residual is not
speech; feeding one in is out-of-distribution and its output would not be
interpretable. Correlating against the known clean watermark is a matched filter
— the optimal detector for a known signal in noise — and needs no network.

### The null — mandatory

Run **unwatermarked** audio through the identical pipeline: attack, align,
gain-match, subtract, correlate. There is no watermark, so every accept is a
false alarm.

Without it, ρ has no meaning: 0.3 might be excellent or might be what pure noise
produces. `2026-08-18_frame-align-null` §3 measured the cost of skipping this
step — a matched-data-only calibration shipped a 30–41% false-accept rate.
**Both arms are reported as TPR at a null-calibrated 1% FPR.**

---

## Validity: what may and may not be claimed

`2026-08-05_align-method-bakeoff` registered a constraint so its result would not
be circular: the original may be used only to estimate generic inverse parameters
— offset, speed, gain, EQ — never to inject signal.

**This experiment deliberately goes further:** full subtraction uses the original
waveform in the detection statistic and does not pass that litmus test. That is
the treatment, not an oversight. It is legitimate here because FakeXpose
verify-mode *holds the original by construction* — that is the premise of the
feature.

The consequence must be stated wherever this is reported:

- **Valid:** informed vs blind on the same files, same clips, same attacks.
- **Not valid:** comparing these numbers against AudioMarkBench, VoxWatermark,
  SoK-2025 or any published blind benchmark. Different detector class.

Informed detection and host-interference cancellation are **classical, not
novel** (Cox et al.; *Selective Host-Interference Cancellation*, IEICE;
*Informed Detection Revisited*). The contribution claimed here is the
**measurement on modern neural audio watermarks**, where every published
benchmark evaluates blind — not the idea. A proper literature check is an open
item before any novelty claim.

---

## Registered predictions

Written before the run, per `align_bench/README.md` §3.

1. **Phase A finds at least one attack that breaks detection at DNSMOS ≥ 3.0.**
   If not, AWARE outlives usability everywhere and informed detection has nothing
   to fix — a legitimate result about the watermark, and the experiment stops.
2. **The music bed is that attack**, and informed detection shifts its breaking
   point by roughly **5 dB** — from ~4.9 dB SNR to near **0 dB**. At +4 dB SNR
   speech carries ~2.5× the power of the music, so removing it should drop total
   interference by ~5.5 dB.
3. **Additive attacks respond; subtractive ones do not.** Informed detection
   un-buries a masked watermark but cannot resurrect a deleted one. Music, noise
   and quantisation (which *floods* the band, 251% energy) should improve;
   high-pass (which *empties* it, 1.9% left) should not, at any quality level.
4. **PESQ and DNSMOS disagree on the music attack** — PESQ ranks it as destroyed,
   DNSMOS as usable. If they agree, the metric argument above is wrong and the
   screen can be simplified.
5. **Codec and re-synthesis attacks do not respond to informed detection**,
   because subtraction assumes additive damage and these rebuild the waveform.

---

## Known boundaries

- **Time-stretch defeats the method entirely.** The bake-off found *no* aligner
  handles time-stretched audio. No alignment, no subtraction — informed detection
  is unavailable, not merely worse. Included in the screen, excluded from Phase B.
- **Gain matching may be insufficient.** If an attack applies EQ, one scale
  factor will not cancel the host and leftover speech will swamp the residual.
  Fallback is a short estimated FIR filter — still within "generic inverse
  parameters". Try the simple version first and record whether it was enough.
- **Emilia is speech only**, and DNSMOS is itself a model, not a listening test.

---

## Budget

50 clips × ~8 attacks × ~6 strengths × 2 arms (watermarked / null) ≈ 4,800
conditions, plus per-clip embedding. Slurm array of 50, one task per clip.
Phase A and Phase B share the generation step, so Phase B adds alignment and
correlation but no new audio.

---

## Phasing

- **Phase A** — screening. Produces the curves and *chooses* the Phase B attack
  set by evidence rather than assumption. Stop and read before continuing.
- **Phase B** — informed vs blind on the selected attacks, with the null.
- **Phase C** — if Phase B works, extend to AudioSeal and Timbre.

**First concrete task:** re-run the music sweep in `real_attacks_experiment.py`
recording DNSMOS and PESQ alongside confidence. Small, cheap, and it decides
whether the lead candidate really sits in the danger zone.

---

## Conclusions

*(empty until the run completes — do not fill this in from expectation)*
