"""
fsss/exp_v16_critical_decimation.py -- exp1: does CRITICALLY-SAMPLED subband
embedding beat the overcomplete band-steer host, and does salient gating help?

THE QUESTION
------------
fsss/band_steer.py carries an arbitrary band onto AWARE's window with an affine
map, and its docstring already concedes the cost:

    "host has more samples than degrees of freedom (74667 vs 64000 ...), so most
     of its spectrum is interpolation, not signal, and anything written there
     dies on the way back."

fsss/band_critical.py removes that waste. Strip k of a uniform M-band split is
exactly sr/2M Hz wide, so decimating by M lands it critically sampled: as many
samples as degrees of freedom, not one more. Every optimizer variable then
points somewhere that survives the trip home.

Does that help? Two readings, and the experiment is built to separate them:
  * it helps  -- the optimizer was wasting effort on directions that get deleted
  * it does not -- the perceptual clamp already pinned those directions, and all
                   criticality buys is speed

CONFIGS -- a 2x2 plus a reference (each = one column of every table)
--------------------------------------------------------------------
                        ALL cells            salient-gated
  CRITICAL (exp1)       exp1a                exp1b        <- band_critical
  OVERCOMPLETE (exp2)   exp2a                exp2b        <- band_steer
  reference             stock = plain AWARE as shipped, no strip, no chain

  exp1a vs exp2a   the SAMPLING question at matched strip and matched gating
  exp1a vs exp1b   the GATING question under critical sampling
  exp2a vs exp2b   the same gating question under overcomplete sampling
  stock            so all four are readable against something familiar

Both salient arms build their masks with the SAME code
(fsss/salient_mapped.py), differing only in the time scale their resampling
implies -- 1/M for exp1, up/down for exp2. Anything else differing between them
would be a confound rather than a result.

exp2a/exp2b may be impossible for a given strip: BandPlan rejects a plan whose
shifted band leaves (0, sr/2), which happens when the native window is 500-4000
and the strip is narrow. Reported SKIP with the reason rather than crashing.
Use --only to trim the config set when a run needs to be cheap.

DETECTOR-BAND FIX (critical -- every earlier fsss experiment that missed this
reported artificially weak numbers)
-----------------------------------------------------------------------------
The AWARE detector zeros everything outside ITS OWN embedding_bands before
reading. The correct band differs per config and is NOT the same everywhere:

  stock  -> whatever load() shipped. Unchanged.
  exp1a/b-> (0, sr/2). After decimation the strip fills the whole relabelled
            spectrum, so AWARE's 1000-4000 window would mask most of it away.
            Safe to open fully: the perceptual bounds pin empty bins by
            themselves (delta = coeff * 10**(-tol/20) is 0 when coeff is 0).
  exp2a  -> the NATIVE window, unchanged. band_steer maps the strip ONTO that
            window on purpose, so widening it here would be wrong.

The band is set on the detector immediately before every detect call, because
one detector object is shared across configs.

WHY 10-SECOND CLIPS
-------------------
Decimating by M divides the frame count by M. At hop 256 a 10 s clip gives 626
frames stock but only 63 after decimation, and the BRH readout head averages
over frames -- that averaging is where crop/stretch robustness comes from.
Shorter clips make the exp1 arms look bad for a reason that has nothing to do
with criticality. 10 s is the floor; --dur can raise it. --hop lets you buy
frames back at the cost of overlap (see the "read:" note at the end).

ATTACKS
-------
Same set as fsss/exp_v12_metapxyl_compare.py so the numbers are comparable:
  METAPXYL-proxy mastering chain: dynamic_compression, echo, mp3, quantization,
  lowpass, gaussian_noise.
  Key VoxWatermark additions: time_stretch and time_jitter (desync -- the family
  that most threatens a fixed decimation phase), highpass, encodec (neural-codec
  ceiling), background_noise, opus.
Each swept across its full VOX_GRID strengths. A missing dependency SKIPs.

The clean table is printed BEFORE the attack sweep starts, so a broken chain
shows up in the log within minutes instead of after the full grid.

READ THE TABLES ACROSS TOLERANCES, NOT DOWN ONE
-----------------------------------------------
Robustness tracks loudness. At a single --tol the arms are NOT comparable:
measured at tol 6 on 3 Emilia clips, mean PESQ was stock 4.23, exp1b 3.23,
exp1a 2.32. exp1a beat stock on six attack families there, but it was ~1.9 PESQ
louder, so that says almost nothing about criticality.

Hence --tols. The run prints a PESQ ladder at the end; find cells with similar
PESQ in DIFFERENT rows and compare those. Comparing down a column at fixed tol
measures which config happens to be louder. exp_v12's header makes the same
point about the staircase.

Run on a GPU node in wmcompare:
    conda activate wmcompare
    python -m fsss.exp_v16_critical_decimation --clean-only --tols 6,9,12
    python -m fsss.exp_v16_critical_decimation --tols 6,9,12        # + attacks
    python -m fsss.exp_v16_critical_decimation --only stock,exp1a,exp2a
    python -m fsss.exp_v16_critical_decimation --strip 4 --anchor-rate 0.8
"""

import os
import sys
import csv
import traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(BASE, "cascade"))

from aware.utils.models import load
from aware.service import embed_watermark, detect_watermark

import vox_attacks
from fsss.chain_embed_critical import CriticalBandAWAREEmbedder
from fsss.exp_a_repeatability import load_16k, pick_clips, EMILIA_CSV, WORK_SR

WM_BITS = 20
DET_CONF = 0.5

METAPXYL_PROXY = ["dynamic_compression", "echo", "mp3", "quantization",
                  "lowpass", "gaussian_noise"]
VOX_IMPORTANT = ["time_stretch", "time_jitter", "highpass", "encodec",
                 "background_noise", "opus"]
DEFAULT_ATTACKS = METAPXYL_PROXY + VOX_IMPORTANT

OUT_DIR = os.path.join(BASE, "fsss_out")
CSV_NAME = "exp_v16_critical_decimation.csv"


# --------------------------------------------------------------------------- #
#  tiny CLI helpers (same shape as exp_v9 / exp_v12)
# --------------------------------------------------------------------------- #
def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def get_list(argv, flag, default, cast):
    if flag in argv:
        return [cast(x) for x in argv[argv.index(flag) + 1].split(",") if x.strip()]
    return default


def bit_acc(bits, pattern):
    p = np.asarray(pattern).astype(int).ravel()
    b = np.asarray(bits).astype(int).ravel()
    n = min(len(b), len(p))
    return float(np.mean(b[:n] == p[:n])) if n else float("nan")


def wsr_db(wm, audio):
    """Watermark-to-signal ratio: 20*log10(rms(wm - x) / rms(x)), in dB.

    THE number that decides whether tolerance_db means what it says. The clamp
    lets each coefficient move to between 0.50x and 1.50x itself at tol 6, so a
    watermark that respects it end to end should land somewhere well below 0 dB
    -- stock typically lands far below.

    It is reported per config because the clamp is applied to the HOST, and for
    the strip configs the host is a peak-normalised band in isolation, ~18 dB
    quieter than the mix (see verify()'s budget_gap_db, and chain_embed.py's
    "No perceptual re-budgeting" note). If the strip arms show a WSR ~18 dB
    higher than stock at the SAME tol, the PESQ gap is that scale mismatch and
    not a property of critical sampling -- fix the budget, do not sweep tol to
    paper over it.

    Lengths are trimmed to the shorter of the two. AWARE's own round trip is
    not length-preserving -- torch.istft with no explicit `length` returns
    (n_frames-1)*hop, so stock comes back 32 samples short of its input at
    n_fft=1024/hop=256 -- while the strip arms reconstruct to len(lo) exactly.
    Do not assume either direction.
    """
    x = np.asarray(audio, dtype=float)
    y = np.asarray(wm, dtype=float)
    n = min(len(x), len(y))
    x, d = x[:n], y[:n] - x[:n]
    return 20.0 * np.log10((np.sqrt(np.mean(d ** 2)) + 1e-20)
                           / (np.sqrt(np.mean(x ** 2)) + 1e-20))


def set_hop(e, hop):
    """Change the STFT hop on an already-built embedder.

    Setting e.hop_length alone is NOT enough and would silently desynchronise
    the pipeline: AWAREEmbedder bakes the hop into the STFT/ISTFT objects it
    builds in __init__, so the preprocess pipeline would keep the old hop while
    our _stft_mag/_istft and the mask builder used the new one.

    Rebuilds both pipelines. `window`/`win_length` are not kept as attributes by
    AWAREEmbedder, so we reassert the card defaults (hann, win_length =
    frame_length) -- correct for both shipped configs, but it is an assumption
    rather than something read back from the object.
    """
    from aware.utils.audio import (ISTFT, STFT, STFTAssembler, STFTDecomposer,
                                   WaveformNormalizer)
    e.hop_length = int(hop)
    e.audio_preprocess_pipeline = [
        WaveformNormalizer(),
        STFT(e.frame_length, e.hop_length, "hann", e.frame_length),
        STFTDecomposer()]
    e.audio_postprocess_pipeline = [
        STFTAssembler(),
        ISTFT(e.frame_length, e.hop_length, "hann", e.frame_length),
        WaveformNormalizer()]
    return e


# --------------------------------------------------------------------------- #
#  configs
# --------------------------------------------------------------------------- #
class Config:
    """One row of every table.

    embedder    : the object whose .embed() we call
    det_band    : embedding_bands to force on the SHARED detector before reading
    prep        : audio -> what the detector should actually be handed. Identity
                  for stock; the analysis half of the chain for the others.
    """

    def __init__(self, name, embedder, det_band, prep, note=""):
        self.name = name
        self.embedder = embedder
        self.det_band = det_band
        self.prep = prep
        self.note = note

    def embed(self, audio, sr, bits):
        return embed_watermark(audio, sample_rate=sr, watermark_bits=bits,
                               model=self.embedder)

    def detect(self, audio, sr, detector):
        detector.embedding_bands = self.det_band      # the detector-band fix
        return detect_watermark(self.prep(audio), sr, detector)


def build_configs(base, detector, args, tol):
    """Construct every config at tolerance `tol`. Anything that cannot be built
    is reported and skipped rather than taking the whole run down."""
    strip, nbands = args["strip"], args["nbands"]
    nyq = WORK_SR / 2.0
    stock_band = tuple(getattr(base, "embedding_bands", (1000, 4000)))
    cfgs = []

    # ---- stock ---------------------------------------------------------- #
    # `base` is shared and reused across tolerances, so set it here rather than
    # once at load(): the exp1 arms take a copy of base.__dict__ inside
    # from_embedder and then override tolerance_db themselves.
    base.tolerance_db = float(tol)
    cfgs.append(Config("stock", base, stock_band, lambda a: a,
                       note=f"as shipped, band {stock_band}"))

    # ---- exp1a / exp1b --------------------------------------------------- #
    for name, salient in (("exp1a", False), ("exp1b", True)):
        try:
            e = CriticalBandAWAREEmbedder.from_embedder(
                base, band_index=strip, num_bands=nbands, sampling_rate=WORK_SR,
                guard_hz=args["guard"] or None, numtaps=args["numtaps"] or None,
                tolerance_db=tol, salient=salient,
                anchor=args["anchor"], anchor_rate=args["anchor_rate"],
                region_ms=args["region_ms"])
            if args["hop"]:
                set_hop(e, args["hop"])
            cfgs.append(Config(name, e, (0, nyq), e.to_detector_input,
                               note=repr(e.plan)))
        except Exception as exc:
            print(f"  SKIP {name}: {type(exc).__name__}: {exc}")

    # ---- exp2a / exp2b (the overcomplete arm) ---------------------------- #
    # Same strip, carried by band_steer's affine map instead of decimation.
    # exp1a vs exp2a is the sampling question; exp2a vs exp2b mirrors exp1a vs
    # exp1b so the gating result can be checked under both representations.
    W = WORK_SR / (2.0 * nbands)
    target = (strip * W, (strip + 1) * W)
    for name, salient in (("exp2a", False), ("exp2b", True)):
        try:
            if salient:
                from fsss.chain_embed_salient import SalientBandSteerAWAREEmbedder
                e2 = SalientBandSteerAWAREEmbedder.from_embedder(
                    base, key=b"unused", target_band=target,
                    sampling_rate=WORK_SR, tolerance_db=tol,
                    anchor=args["anchor"], anchor_rate=args["anchor_rate"],
                    region_ms=args["region_ms"])
            else:
                from fsss.chain_embed import BandSteerAWAREEmbedder
                e2 = BandSteerAWAREEmbedder.from_embedder(
                    base, key=b"unused", target_band=target,
                    sampling_rate=WORK_SR, tolerance_db=tol)
            if args["hop"]:
                set_hop(e2, args["hop"])
            # band_steer maps the strip ONTO the native window on purpose, so
            # the detector keeps its NATIVE band here -- the opposite of exp1,
            # which opens the band fully because decimation spreads the strip
            # across the whole relabelled spectrum.
            cfgs.append(Config(name, e2, tuple(e2.native_band),
                               e2.to_detector_input, note=repr(e2.plan)))
        except Exception as exc:
            print(f"  SKIP {name}: {type(exc).__name__}: {exc}")
            print("       (BandPlan rejects a shifted band that leaves "
                  "(0, sr/2) -- happens for narrow strips when the native "
                  "window is 500-4000. Try a wider strip or a smaller guard.)")

    if args["only"]:
        keep = set(args["only"])
        cfgs = [c for c in cfgs if c.name in keep]
    return cfgs


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
def main(argv):
    args = {
        "seed": get_arg(argv, "--seed", 0, int),
        "dur": get_arg(argv, "--dur", 10.0, float),
        "nclips": get_arg(argv, "--nclips", 3, int),
        "strip": get_arg(argv, "--strip", 2, int),
        "nbands": get_arg(argv, "--nbands", 10, int),
        # Sweeping tolerance is not optional for a fair comparison: robustness
        # tracks loudness, so configs must be compared at MATCHED PESQ rather
        # than matched tol. 6 is AWARE's shipped default; lower is louder.
        "tols": get_list(argv, "--tols", [6.0], float),
        "hop": get_arg(argv, "--hop", 0, int),
        # Raise these if verify() reports a low decimation_snr_db on real speech:
        # more taps sharpen the filter, more guard keeps its roll-off inside the
        # slot. Both cost usable bandwidth or compute, neither costs correctness.
        "numtaps": get_arg(argv, "--numtaps", 0, int),
        "guard": get_arg(argv, "--guard", 0.0, float),
        # AWARE embeds by optimizing per file, so this is the single biggest
        # lever on wall clock. Drop it to shake out plumbing, restore it for
        # any number you intend to report.
        "iters": get_arg(argv, "--iters", 0, int),
        "anchor": get_arg(argv, "--anchor", "librosa_flux", str),
        "anchor_rate": get_arg(argv, "--anchor-rate", 1.2, float),
        "region_ms": get_arg(argv, "--region-ms", 250.0, float),
        # All five configs run by default now that exp2 has a salient arm.
        # --only stock,exp1a trims the set when a run needs to be cheap.
        "only": get_list(argv, "--only", [], str),
        "clean_only": "--clean-only" in argv,
        "no_pesq": "--no-pesq" in argv,
        "attacks": get_list(argv, "--attacks", DEFAULT_ATTACKS, str),
        "clip": get_arg(argv, "--clip", "", str),
    }

    # ---- data ------------------------------------------------------------- #
    if args["clip"]:
        clips = [args["clip"]]
    else:
        clips = pick_clips(EMILIA_CSV, args["nclips"])
    if not clips:
        print("no clips found (set --clip or EMILIA_CSV)")
        return
    n_want = int(round(args["dur"] * WORK_SR))
    audios = [load_16k(c)[:n_want] for c in clips]
    bits = np.random.default_rng(args["seed"]).integers(0, 2, size=WM_BITS,
                                                        dtype=np.int32)

    W = WORK_SR / (2.0 * args["nbands"])
    print("=" * 78)
    print("exp_v16 -- critically-sampled subband embedding (exp1)")
    print("=" * 78)
    print(f"clips        : {len(audios)} x {args['dur']:.0f}s")
    print(f"strip        : {args['strip']}/{args['nbands']} = "
          f"{args['strip']*W:.0f}-{(args['strip']+1)*W:.0f} Hz "
          f"({'even/upright' if args['strip'] % 2 == 0 else 'ODD/reversed'})")
    print(f"tolerances   : {args['tols']}  (lower = louder/stronger)")
    print(f"salient      : {args['anchor']} @ {args['anchor_rate']}/s, "
          f"{args['region_ms']:.0f} ms regions (exp1b only)")
    print(f"attacks      : {'(clean only)' if args['clean_only'] else args['attacks']}")
    print()

    # Record what actually ran. AWAREEmbedder picks its device at __init__
    # ("cuda" if available), so a GPU job that silently fell back to CPU looks
    # identical in the results and only shows up as wall clock.
    try:
        import torch
        if torch.cuda.is_available():
            print(f"device       : cuda -- {torch.cuda.get_device_name(0)}")
        else:
            print("device       : CPU  (torch.cuda.is_available() is False)")
    except Exception as exc:
        print(f"device       : unknown ({exc})")
    print()

    embedder, detector = load()
    if args["iters"]:
        print(f"  (num_iterations overridden to {args['iters']} for ALL configs)")

    # Configs are rebuilt per tolerance inside the sweep below, because
    # tolerance_db is fixed at construction. verify() runs once, on the first
    # tolerance only -- it measures the chain, which does not depend on tol.

    # ---- run --------------------------------------------------------------- #
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, CSV_NAME)
    fout = open(csv_path, "w", newline="")
    writer = csv.writer(fout)
    writer.writerow(["clip", "config", "strip", "tol", "attack", "param",
                     "bit_acc", "conf", "detected", "pesq", "wsr_db",
                     "coverage", "cells", "cells_full"])

    # keyed by (tol, config, attack) so each tolerance gets its own table, the
    # way exp_v9 does it
    agg = {}
    pesq_agg = {}                 # (tol, config) -> [pesq, ...]
    wsr_agg = {}                  # (tol, config) -> [wsr_db, ...]

    def record(tol, cfg, attack, acc, conf):
        agg.setdefault((tol, cfg, attack), []).append((acc, int(conf >= DET_CONF)))

    pesq_metric = None
    if not args["no_pesq"]:
        try:
            from aware.metrics.audio import PESQ
            pesq_metric = PESQ()
        except Exception as exc:
            print(f"PESQ unavailable ({exc}); continuing without it\n")

    all_names = []

    for tol in args["tols"]:
        print("\n" + "=" * 78)
        print(f"TOLERANCE {tol}   (lower = louder/stronger)")
        print("=" * 78)
        cfgs = build_configs(embedder, detector, args, tol)
        if args["iters"]:
            for c in cfgs:
                c.embedder.num_iterations = int(args["iters"])
        for c in cfgs:
            if c.name not in all_names:
                all_names.append(c.name)
            print(f"  {c.name:8s} {c.note}")

        # verify() only depends on the chain, not on tolerance, so run it once.
        if tol == args["tols"][0]:
            for c in cfgs:
                if hasattr(c.embedder, "verify"):
                    print(f"\n--- verify {c.name} " + "-" * (56 - len(c.name)))
                    try:
                        c.embedder.verify(audios[0], WORK_SR)
                    except Exception as exc:
                        print(f"  verify FAILED: {type(exc).__name__}: {exc}")
                        traceback.print_exc()

        wm_cache = {}             # (clip_index, config) -> watermarked audio

        # ---- phase 1: clean ------------------------------------------------ #
        print(f"\n--- PHASE 1 clean, tol {tol} ---")
        for ci, (clip, audio) in enumerate(zip(clips, audios)):
            print(f"clip {ci+1}/{len(audios)}: {os.path.basename(clip)}")
            for c in cfgs:
                try:
                    wm = c.embed(audio, WORK_SR, bits)
                except Exception as exc:
                    print(f"  {c.name:8s} EMBED FAILED: {type(exc).__name__}: {exc}")
                    continue
                wm_cache[(ci, c.name)] = wm

                pat, conf = c.detect(wm, WORK_SR, detector)
                acc = bit_acc(bits, pat)
                record(tol, c.name, "clean", acc, conf)

                pq = float("nan")
                if pesq_metric is not None:
                    try:
                        pq = float(pesq_metric(wm, audio, WORK_SR))
                        pesq_agg.setdefault((tol, c.name), []).append(pq)
                    except Exception:
                        pass                              # too little speech content

                wsr = wsr_db(wm, audio)
                wsr_agg.setdefault((tol, c.name), []).append(wsr)

                st = getattr(c.embedder, "last_mask_stats", None) or {}
                writer.writerow([ci, c.name, args["strip"], tol, "clean", "",
                                 round(acc, 4), round(float(conf), 4),
                                 int(conf >= DET_CONF), round(pq, 4),
                                 round(wsr, 2),
                                 round(st.get("coverage", float("nan")), 4),
                                 st.get("cells", ""), st.get("cells_full_stripe", "")])
                print(f"  {c.name:8s} bit_acc {acc:.3f}  conf {float(conf):.3f}  "
                      f"pesq {pq:.2f}  wsr {wsr:+6.1f}dB" +
                      (f"  coverage {st['coverage']*100:.0f}%" if "coverage" in st else ""))
        fout.flush()
        print_table(agg, tol, [c.name for c in cfgs], ["clean"], f"CLEAN, tol {tol}")

        if args["clean_only"]:
            continue

        # ---- phase 2: attacks ---------------------------------------------- #
        print(f"\n--- PHASE 2 attacks, tol {tol} ---")
        for ci, clip in enumerate(clips):
            print(f"clip {ci+1}/{len(clips)}: {os.path.basename(clip)}")
            for c in cfgs:
                wm = wm_cache.get((ci, c.name))
                if wm is None:
                    continue
                for attack in args["attacks"]:
                    for label, param in vox_attacks.VOX_GRID.get(attack, []):
                        try:
                            wa = vox_attacks.apply(attack, param,
                                                   wm.astype("float32"), WORK_SR)
                        except Exception:
                            continue                      # missing dep -> SKIP
                        if wa is None:
                            continue
                        pat, conf = c.detect(wa, WORK_SR, detector)
                        acc = bit_acc(bits, pat)
                        record(tol, c.name, attack, acc, conf)
                        writer.writerow([ci, c.name, args["strip"], tol,
                                         attack, label, round(acc, 4),
                                         round(float(conf), 4),
                                         int(conf >= DET_CONF),
                                         "", "", "", "", ""])
                fout.flush()
                print(f"  {c.name:8s} done")
        print_table(agg, tol, [c.name for c in cfgs],
                    ["clean"] + args["attacks"], f"CLEAN + ATTACKS, tol {tol}")

    fout.close()

    # ---- the PESQ ladder: the whole reason for sweeping ------------------- #
    # Robustness tracks loudness, so a config that is 2 PESQ louder will look
    # more robust for reasons that have nothing to do with its design. Read the
    # attack tables ACROSS tolerances at equal PESQ, not down a single table.
    print("\n" + "#" * 72)
    print("###  PESQ / WSR LADDER -- use this to pick comparable cells")
    print("#" * 72)
    print(f"  {'tol':>5s}" + "".join(f"{n + ' pesq/wsr':>21s}" for n in all_names))
    for tol in args["tols"]:
        line = f"  {tol:>5.1f}"
        for n in all_names:
            p = pesq_agg.get((tol, n), [])
            w = wsr_agg.get((tol, n), [])
            cell = (f"{np.mean(p):.2f} / {np.mean(w):+.1f}dB"
                    if p and w else "-")
            line += cell.rjust(21)
        print(line)
    print("#" * 72)
    print("  Compare cells with SIMILAR PESQ across rows. Comparing down a")
    print("  column at fixed tol measures loudness, not design.")
    print()
    print("  FIRST look at WSR at the SAME tol. tolerance_db is applied to the")
    print("  HOST, and for the strip arms the host is a peak-normalised band in")
    print("  isolation -- verify() measured it 18 dB quieter than the mix. If the")
    print("  strip arms sit ~18 dB above stock in WSR at equal tol, the PESQ gap")
    print("  is THAT scale mismatch, not critical sampling, and the fix is to")
    print("  re-budget the clamp against the original -- not to sweep tol until")
    print("  the numbers happen to line up.")

    print(f"\nwrote {csv_path}")
    print("read:")
    print("  exp1a vs stock at MATCHED PESQ -- the real question. exp1a wins on "
          "robustness at fixed tol, but only because it is far louder there; "
          "find the tol where their PESQ agrees and re-read.")
    print("  exp1a vs exp1b -- the gating question, same rule: match PESQ first.")
    print("  time_stretch / time_jitter -- shift the decimation PHASE. A constant "
          "phase rotation is invisible to a magnitude spectrogram, so exp1 should "
          "track stock rather than collapse.")
    print("  highpass / lowpass -- with everything in one strip these are "
          f"all-or-nothing. A notch inside {args['strip']*W:.0f}-"
          f"{(args['strip']+1)*W:.0f} Hz removes the entire watermark, where "
          "stock only loses a fraction.")
    print(f"  if exp1a trails stock everywhere, try --hop {max(1, 256 // args['nbands'])} "
          "-- decimation cut the frame count by the band count, and the BRH "
          "averages over frames.")


def print_table(agg, tol, names, rows, title):
    Wc = 15
    width = 22 + Wc * len(names)
    print("\n" + "#" * width)
    print(f"###  {title}    cell = detected/total  mean_bit_acc")
    print("#" * width)
    print(f"  {'attack':20s}" + "".join(f"{n:>{Wc}s}" for n in names))
    for attack in rows:
        line = f"  {attack:20s}"
        for n in names:
            r = agg.get((tol, n, attack), [])
            if not r:
                line += "-".rjust(Wc)
                continue
            a = float(np.mean([x[0] for x in r]))
            d = sum(x[1] for x in r)
            line += f"{d:>2d}/{len(r):<2d} {a:4.2f}".rjust(Wc)
        print(line)
    print("#" * width)


if __name__ == "__main__":
    main(sys.argv[1:])
