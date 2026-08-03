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

CONFIGS (each = one row of every table)
---------------------------------------
  stock   plain AWARE as shipped. No strip, no chain. The reference.
  exp1a   critical strip, ALL cells writable.          <- band_critical
  exp1b   critical strip, salient-gated (librosa_flux). <- band_critical + gating
  exp2a   the SAME strip via band_steer's affine map.   <- overcomplete control
          Opt-in (--with-exp2). May be impossible for a given strip: BandPlan
          rejects a plan whose shifted band leaves (0, sr/2), which happens when
          the native window is 500-4000 and the strip is narrow. Reported SKIP
          with the reason rather than crashing the run.

exp1a vs exp2a is the sampling question at matched strip.
exp1a vs exp1b is the gating question at matched sampling.
stock is there so both are readable against something familiar.

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

Run on a GPU node in wmcompare:
    conda activate wmcompare
    python -m fsss.exp_v16_critical_decimation --clean-only        # ~minutes
    python -m fsss.exp_v16_critical_decimation                      # clean + attacks
    python -m fsss.exp_v16_critical_decimation --with-exp2 --nclips 5
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


def build_configs(base, detector, args):
    """Construct every config. Anything that cannot be built is reported and
    skipped rather than taking the whole run down."""
    strip, nbands, tol = args["strip"], args["nbands"], args["tol"]
    nyq = WORK_SR / 2.0
    stock_band = tuple(getattr(base, "embedding_bands", (1000, 4000)))
    cfgs = []

    # ---- stock ---------------------------------------------------------- #
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

    # ---- exp2a (opt-in overcomplete control) ----------------------------- #
    if args["with_exp2"]:
        try:
            from fsss.chain_embed import BandSteerAWAREEmbedder
            W = WORK_SR / (2.0 * nbands)
            target = (strip * W, (strip + 1) * W)
            e2 = BandSteerAWAREEmbedder.from_embedder(
                base, key=b"unused", target_band=target,
                sampling_rate=WORK_SR, tolerance_db=tol)
            # band_steer maps the strip ONTO the native window on purpose, so the
            # detector keeps its native band here -- unlike exp1.
            cfgs.append(Config("exp2a", e2, tuple(e2.native_band),
                               e2.to_detector_input, note=repr(e2.plan)))
        except Exception as exc:
            print(f"  SKIP exp2a: {type(exc).__name__}: {exc}")
            print("       (BandPlan rejects a shifted band that leaves (0, sr/2); "
                  "try --with-exp2 on a wider strip, or a smaller guard.)")
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
        "tol": get_arg(argv, "--tol", 6.0, float),
        "hop": get_arg(argv, "--hop", 0, int),
        # Raise these if verify() reports a low decimation_snr_db on real speech:
        # more taps sharpen the filter, more guard keeps its roll-off inside the
        # slot. Both cost usable bandwidth or compute, neither costs correctness.
        "numtaps": get_arg(argv, "--numtaps", 0, int),
        "guard": get_arg(argv, "--guard", 0.0, float),
        "anchor": get_arg(argv, "--anchor", "librosa_flux", str),
        "anchor_rate": get_arg(argv, "--anchor-rate", 1.2, float),
        "region_ms": get_arg(argv, "--region-ms", 250.0, float),
        "with_exp2": "--with-exp2" in argv,
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
    print(f"tolerance    : {args['tol']}  (lower = louder/stronger)")
    print(f"salient      : {args['anchor']} @ {args['anchor_rate']}/s, "
          f"{args['region_ms']:.0f} ms regions (exp1b only)")
    print(f"attacks      : {'(clean only)' if args['clean_only'] else args['attacks']}")
    print()

    embedder, detector = load()
    print("configs:")
    cfgs = build_configs(embedder, detector, args)
    for c in cfgs:
        print(f"  {c.name:8s} {c.note}")
    print()

    # ---- verify the chain BEFORE spending time on a full run -------------- #
    # A convention mismatch produces a silently wrong watermark, not an
    # exception, so this is not optional.
    for c in cfgs:
        if hasattr(c.embedder, "verify"):
            print(f"--- verify {c.name} " + "-" * (58 - len(c.name)))
            try:
                c.embedder.verify(audios[0], WORK_SR)
            except Exception as exc:
                print(f"  verify FAILED: {type(exc).__name__}: {exc}")
                traceback.print_exc()
            print()

    # ---- run --------------------------------------------------------------- #
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, CSV_NAME)
    fout = open(csv_path, "w", newline="")
    writer = csv.writer(fout)
    writer.writerow(["clip", "config", "strip", "tol", "attack", "param",
                     "bit_acc", "conf", "detected", "pesq",
                     "coverage", "cells", "cells_full"])

    agg = {}                      # (config, attack) -> [(bit_acc, detected), ...]

    def record(cfg, attack, acc, conf):
        agg.setdefault((cfg, attack), []).append((acc, int(conf >= DET_CONF)))

    pesq_metric = None
    if not args["no_pesq"]:
        try:
            from aware.metrics.audio import PESQ
            pesq_metric = PESQ()
        except Exception as exc:
            print(f"PESQ unavailable ({exc}); continuing without it\n")

    wm_cache = {}                 # (clip_index, config) -> watermarked audio

    # ---- phase 1: clean ---------------------------------------------------- #
    print("=" * 78)
    print("PHASE 1 -- clean (no attacks)")
    print("=" * 78)
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
            record(c.name, "clean", acc, conf)

            pq = float("nan")
            if pesq_metric is not None:
                try:
                    pq = float(pesq_metric(wm, audio, WORK_SR))
                except Exception:
                    pass                                  # too little speech content

            st = getattr(c.embedder, "last_mask_stats", None) or {}
            writer.writerow([ci, c.name, args["strip"], args["tol"], "clean", "",
                             round(acc, 4), round(float(conf), 4),
                             int(conf >= DET_CONF), round(pq, 4),
                             round(st.get("coverage", float("nan")), 4),
                             st.get("cells", ""), st.get("cells_full_stripe", "")])
            print(f"  {c.name:8s} bit_acc {acc:.3f}  conf {float(conf):.3f}  "
                  f"pesq {pq:.2f}" +
                  (f"  coverage {st['coverage']*100:.0f}%" if "coverage" in st else ""))
    fout.flush()

    print_table(agg, cfgs, ["clean"], "CLEAN")

    if args["clean_only"]:
        fout.close()
        print(f"\nwrote {csv_path}")
        print("read: clean bit_acc should be ~1.0 for every config. If exp1a is "
              "below stock here, the chain is wrong -- check verify()'s "
              "chain_null and decimation_snr before reading anything into the "
              "attack numbers.")
        return

    # ---- phase 2: attacks --------------------------------------------------- #
    print("=" * 78)
    print("PHASE 2 -- attacks")
    print("=" * 78)
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
                        continue                          # missing dep -> SKIP
                    if wa is None:
                        continue
                    pat, conf = c.detect(wa, WORK_SR, detector)
                    acc = bit_acc(bits, pat)
                    record(c.name, attack, acc, conf)
                    writer.writerow([ci, c.name, args["strip"], args["tol"],
                                     attack, label, round(acc, 4),
                                     round(float(conf), 4), int(conf >= DET_CONF),
                                     "", "", "", ""])
            fout.flush()
            print(f"  {c.name:8s} done")
    fout.close()

    print_table(agg, cfgs, ["clean"] + args["attacks"], "CLEAN + ATTACKS")
    print(f"\nwrote {csv_path}")
    print("read:")
    print("  exp1a vs exp2a -- the sampling question. If they tie, criticality "
          "buys speed but not strength, and band_steer's 'excess dies on the way "
          "back' was already being handled by the perceptual clamp.")
    print("  exp1a vs exp1b -- the gating question. Only fair at matched PESQ; "
          "at fixed --tol the two arms are audibly different, so sweep --tol and "
          "compare across runs at equal PESQ.")
    print("  time_stretch / time_jitter -- the attacks that shift the decimation "
          "PHASE. Constant phase rotation is invisible to a magnitude "
          "spectrogram, so exp1 should mostly survive; if it does not, the "
          "phase sensitivity is real and worth its own experiment.")
    print("  highpass / lowpass -- with everything in one strip these are "
          f"all-or-nothing. A notch inside {args['strip']*W:.0f}-"
          f"{(args['strip']+1)*W:.0f} Hz removes the entire watermark, where "
          "stock only loses a fraction.")
    print(f"  if exp1a trails stock everywhere, try --hop {max(1, 256 // args['nbands'])} "
          "-- decimation cut the frame count by the band count, and the BRH "
          "averages over frames.")


def print_table(agg, cfgs, rows, title):
    names = [c.name for c in cfgs]
    Wc = 15
    width = 22 + Wc * len(names)
    print("\n" + "#" * width)
    print(f"###  {title}    cell = detected/total  mean_bit_acc")
    print("#" * width)
    print(f"  {'attack':20s}" + "".join(f"{n:>{Wc}s}" for n in names))
    for attack in rows:
        line = f"  {attack:20s}"
        for n in names:
            r = agg.get((n, attack), [])
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
