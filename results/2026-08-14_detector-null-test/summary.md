# Summary — what do the detectors score on audio with no watermark?

**Verdict: keep the 0.5 detection threshold. It is correct for speech, with
0.23–0.35 of headroom, and 0 false positives in 300 clean clips.**

## The headline

| Model | n (Emilia speech) | median | p95 | **max** | headroom below 0.5 |
|---|---|---|---|---|---|
| AudioSeal | 300 | 0.0000 | 0.0405 | **0.1497** | **+0.3503** |
| AWARE | 300 | 0.0734 | 0.1635 | **0.2723** | **+0.2277** |

Zero false positives in 300 clean speech clips. By the rule of three that
supports a false-positive rate **below 1% at 95% confidence** for speech.

Add the curated set (6 public-domain voices, 2 mp3s, 1 non-speech) and no speech
file anywhere in the experiment exceeded **0.2723**.

## What this settles

The threshold in production had been set three times by three people and never
measured: `BIT_ACC_PRESENT = 0.8`, then `0.001` in config, then `0.5` from
upstream defaults. Two competing claims were on the table:

* clean audio scores 0.049–0.175 (from `cascade_out/cascade_controls.csv`, n=1)
* clean speech "often scores ~0.4–0.55" (observation recorded on the
  `for-prod-watermark` branch)

**The first is right.** Over 306 clean files, speech never came close to 0.4.
The second claim is not reproduced by this pipeline and should not drive the
threshold. Moving to 0.65/0.55 would buy nothing on this evidence and would cost
detection sensitivity — the solo-model benchmark has AudioSeal at 0.628 under
`quantize_8bit` and 0.335 under `noise_10db`, both of which a 0.65 threshold
would discard.

## What it does not settle

**One file failed: a pure 440 Hz sine tone scored AWARE = 0.9679.**

Not a threshold problem — 0.9679 clears any candidate threshold, including the
conservative 0.65. And not an energy problem: digital silence scored 0.0036 and
white noise 0.0053 on the same detector. It is specific to *tonal* structure,
and notably 440 Hz sits below AWARE's 1000–4000 Hz embedding band, so the
detector is responding to spectral leakage pushed through an untrained
random-projection network.

AudioSeal scored **0.0000** on the same file. The two detectors disagree
completely, and the current OR rule discards that disagreement.

This is logged as a separate defect, deliberately **not** addressed by this
experiment. Its practical reach is unmapped: pure sines are not customer audio,
but hold music, test tones, DTMF and single-instrument recordings lie in the
same direction and nobody has measured where the boundary is.

## Confidence and limits

* **Deterministic.** Every file in the curated set was scored twice; all 12
  matched exactly (`stable=True`). The non-determinism affecting Timbre's
  confidence does not touch AudioSeal or AWARE.
* **Timbre is absent by design.** It ships a decoder, not a detector; its adapter
  returns bit-accuracy against whatever message the process embedded last. It
  cannot answer "is a watermark present" and was excluded.
* **Speech-dominated.** 300 Emilia clips (5–30 s) plus 6 telephone-band voices.
  Music is represented by one 1.3 s file. Non-speech content is under-tested.
* **Detection only.** No embedding was performed, so this says nothing about
  whether real watermarks are still *found* at 0.5. That is the paired positive
  control and it remains outstanding.
* **Two mp3s hit libmpg123 decode errors** and fell back to `audioread`. They
  scored fine, but that decode path deserves a check in production.
