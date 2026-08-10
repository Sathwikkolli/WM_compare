# Email draft — alignment bake-off results

**To:** [prof]
**Subject:** Alignment method benchmark — results and recommendation

---

Hi Professor [NAME],

Quick update on the alignment work for non-blind watermark detection.

**Context.** Since we hold the unwatermarked original, we can align it to a
distorted copy before running the detector, rather than detecting blind as every
published benchmark does. Alignment is the gate: misalign by a few samples and
the comparison is worthless. I benchmarked six candidate alignment methods to
pick one.

**Setup.** 30 Emilia clips (9–28 s) x 76 attack configurations x 6 methods =
13,680 runs on Great Lakes. Attacks are the Metapyxl mastering-chain proxy plus
the VoxWatermark grid, with crop/splice/insert added by hand — VoxWatermark has
no cropping attack, which is the one case that matters most for us. Ground truth
is logged at generation time, so every offset is known exactly rather than
inferred.

**Recommendation: GCC-PHAT** — classical phase-transform cross-correlation,
about 15 lines of scipy, no new dependency. It is sample-exact on cropping
(median error 0.000 ms), never reports a false shift on unmodified audio, has
the only well-behaved confidence score of the six, and runs 20–30x faster than
the fingerprinting libraries.

**Two findings worth flagging:**

1. **No method aligns time-stretched audio.** Every non-DTW method scored
   exactly 0% across all five time-stretch strengths. If time-scale modification
   is in our threat model, this needs a separate resample-factor search — none
   of the off-the-shelf tools address it.

2. **Two libraries report confidence scores that are actively misleading.** One
   is constant (carries no information); another is inverted at the top end —
   its highest-confidence predictions are its least accurate. Fallback logic
   keyed on those scores would fire backwards. Worth knowing before anyone
   builds on them.

I have also locked AWARE to its `20bps` configuration, which handles cropping
through per-second repeated embedding and sliding-window majority voting. It
costs some audio quality (a 1.5 dB louder watermark) — happy to walk through
that trade-off.

**Attached:** full analysis, the accuracy curves, a method x attack heatmap, and
the raw metrics table. Everything is in the repo under
`results/2026-08-05_align-method-bakeoff/`.

Two open items I would value your steer on: whether time-stretch robustness
matters enough to build the resample search, and whether we should extend the
benchmark to music content — our clips are speech-only, but the Metapyxl chain
includes a music bed.

Best,
Sathwik

---

## Attachments

| File | Why |
|---|---|
| `ANALYSIS.md` | Full write-up with per-family breakdown |
| `figures/error_cdf.png` | The single best figure — accuracy at any tolerance |
| `figures/heatmap.png` | Method x attack at a glance |
| `figures/reliability.png` | Shows the confidence problem visually |
| `data/metrics.csv` | Raw numbers |

Optional: `figures/scatter.png` if you want to show *why* methods fail —
the off-diagonal clusters are the DTW false-shift problem.
