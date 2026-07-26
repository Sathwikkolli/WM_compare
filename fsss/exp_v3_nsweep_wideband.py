"""
fsss/exp_v3_nsweep_wideband.py -- key-driven subband-hopping over a WIDE (500-6000
Hz) range, swept across N = 2,3,4,5,6 sub-bands, on ONE 5-second Emilia clip.

WHAT THIS TESTS
---------------
Stock AWARE embeds the 20-bit mark in a single static stripe (the installed card's
`embedding_bands`, ~1000-4000 Hz). Here we instead:

  1. widen the embedding band to (500, 6000) Hz -- both the writable region AND the
     detector's zero-mask, by setting `band_range` (staircase tiling) *and*
     `embedding_bands` (base AWARE's `_get_embedding_frequency_indices`) together;
  2. split that range into N equal-Hz sub-bands;
  3. cut the clip into FIXED equal-length dwell segments (no content analysis);
  4. let the KEY choose which sub-band each segment writes into, via
     HMAC(key, segment_index) % N -- a pseudorandom frequency-HOP sequence.

DESIGN DECISIONS (grounded in the literature) -- see the thesis notes:
  * Hop order  = key-driven PN hop SEQUENCE, repeats allowed (NOT a permutation).
                 Faithful to FHSS and to Malik/Khokhar/Ansari FSSS ("randomly
                 selects sub-band(s) according to a secret key"). Already what
                 StaircaseAWAREEmbedder does: HMAC(key, seg_index) % N.
  * Segments   = FIXED equal-length dwell periods ("predefined time period", the
                 canonical frequency-hopping-watermarking choice), which also
                 matches AWARE's own 1-s partitioning. AWARE's detector is
                 time-order-agnostic (BRH + majority voting), so it does NOT
                 re-derive segment boundaries -> content-adaptive anchors buy
                 little here; kept only as a future ablation (segment_mode=librosa).

FEASIBILITY NOTE
----------------
AWARE's detector reads the FULL 128-mel spectrogram (0-8 kHz), not a cropped band,
so gradients DO reach 4000-6000 Hz once we stop zeroing it. DeepMark's own
`config_segments.yaml` already uses embedding_bands [1000, 5000], so pushing to
6000 is a modest step. Whether the frozen detector can be STEERED as effectively up
there is exactly what the per-N bit_acc / conf / PESQ below answer.

Detection always uses the STOCK AWARE detector (unchanged). More sub-bands => fewer
writable cells per segment => weaker mark, so we embed at a louder budget
(tolerance_db default -6; lower = louder). Re-run with --tol to trade PESQ<->bit_acc.

Run on a GPU node in wmcompare (AWARE + Emilia both live on Great Lakes):
    conda activate wmcompare
    python -m fsss.exp_v3_nsweep_wideband
    python -m fsss.exp_v3_nsweep_wideband --clip audio/client_original_16k.wav
    python -m fsss.exp_v3_nsweep_wideband --key thesis --dur 5 --tol -6 --seglen 0.5
"""

import os
import sys
import csv
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(BASE, "cascade"))

import librosa
from aware.utils.models import load
from aware.service import embed_watermark, detect_watermark
from aware.metrics.audio import PESQ

from fsss.staircase import StaircaseAWAREEmbedder
from fsss.exp_a_repeatability import load_16k, pick_clips, EMILIA_CSV, WORK_SR

WM_BITS = 20
BAND = (500, 6000)
N_LIST = [2, 3, 4, 5, 6]
OUT_DIR = os.path.join(BASE, "fsss_out")


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def bit_acc(bits, pattern):
    p = np.asarray(pattern).astype(int).ravel()
    b = np.asarray(bits).astype(int).ravel()
    n = min(len(b), len(p))
    return float(np.mean(b[:n] == p[:n])) if n else float("nan")


def band_edges_hz(n_bands):
    """The N+1 equal-Hz edges tiling BAND, for human-readable reporting."""
    return np.linspace(BAND[0], BAND[1], n_bands + 1)


def hop_schedule(st, audio):
    """Reproduce the exact per-segment band sequence the embedder will use.

    Reuses the staircase's OWN _segments/_hmac_band so this matches embedding
    exactly. n_frames is computed with librosa (center=True, same as AWARE's
    torch.stft), so magnitude.shape[1] agrees frame-for-frame.
    """
    S = librosa.stft(audio.astype(np.float32),
                     n_fft=st.frame_length, hop_length=st.hop_length)
    n_frames = S.shape[1]
    segs = st._segments(audio, WORK_SR, n_frames)
    bands = [st._hmac_band(i) for i in range(len(segs))]      # 0-indexed
    return segs, bands


def fmt_hops(bands):
    """1-indexed hop string, matching the '1 -> 3 -> 2' convention."""
    return " -> ".join(str(b + 1) for b in bands)


def main(argv):
    clip = get_arg(argv, "--clip", None)
    key = get_arg(argv, "--key", "thesis")
    seed = get_arg(argv, "--seed", 0, int)
    dur = get_arg(argv, "--dur", 5.0, float)          # trim clip to this many seconds
    tol = get_arg(argv, "--tol", -6.0, float)         # louder than stock (6.0)
    seglen = get_arg(argv, "--seglen", 0.5, float)    # fixed dwell length (s)

    if clip is None:
        clips = pick_clips(EMILIA_CSV, 1)
        if not clips:
            print("no Emilia clip found; pass --clip PATH")
            return
        clip = clips[0]

    audio = load_16k(clip)
    n_keep = int(round(dur * WORK_SR))
    if len(audio) < n_keep:
        print(f"warning: clip is only {len(audio)/WORK_SR:.2f}s, shorter than --dur {dur}")
    audio = audio[:n_keep]                             # take the first `dur` seconds

    bits = np.random.default_rng(seed).integers(0, 2, size=WM_BITS, dtype=np.int32)
    print(f"clip     : {clip}")
    print(f"duration : {len(audio)/WORK_SR:.2f}s   ({len(audio)} samples @ {WORK_SR} Hz)")
    print(f"payload  : {bits.tolist()}")
    print(f"band     : {BAND[0]}-{BAND[1]} Hz   key='{key}'   tol={tol}   seglen={seglen}s")

    embedder, detector = load()
    pesq_metric = PESQ()

    # stock reference (unmodified AWARE, its own card band) + wideband no-hop (n=1)
    # + the N=2..6 hop sweep. n=1 isolates the *hopping* cost from the *wideband* cost.
    configs = [("stock", None)] + [("n1", 1)] + [(f"n{n}", n) for n in N_LIST]

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "exp_v3_nsweep_wideband.csv")
    fout = open(csv_path, "w", newline="")
    writer = csv.writer(fout)
    writer.writerow(["config", "n_bands", "band_lo", "band_hi", "tol_db", "seglen_s",
                     "n_segments", "hop_sequence", "bit_acc", "conf", "detected",
                     "pesq", "embed_s"])

    rows = []
    for name, n_bands in configs:
        print(f"\n=== {name} ===")
        if n_bands is None:                            # plain stock AWARE
            model = embedder
            segs_str, n_segs = "", 0
        else:
            model = StaircaseAWAREEmbedder.from_embedder(
                embedder, key=key, n_bands=n_bands, band_range=BAND,
                segment_mode="fixed", segment_len_s=seglen, tolerance_db=tol)
            model.embedding_bands = BAND               # widen the zero-mask to 500-6000
            edges = band_edges_hz(n_bands)
            print("  sub-bands: " + ", ".join(
                f"{i+1}:[{edges[i]:.0f}-{edges[i+1]:.0f}]" for i in range(n_bands)))
            segs, bands = hop_schedule(model, audio)
            n_segs = len(segs)
            segs_str = fmt_hops(bands)
            print(f"  {n_segs} segments (~{seglen}s each), key-driven hop:")
            print(f"    {segs_str}")

        t0 = time.time()
        wm = embed_watermark(audio, sample_rate=WORK_SR, watermark_bits=bits, model=model)
        embed_s = time.time() - t0

        pattern, conf = detect_watermark(wm, WORK_SR, detector)
        acc = bit_acc(bits, pattern)
        m = min(len(audio), len(wm))
        try:
            pesq = float(pesq_metric(wm[:m], audio[:m], WORK_SR))
        except Exception as e:
            pesq = float("nan")
            print(f"  PESQ failed: {e}")
        det = int(conf >= 0.5)
        print(f"  bit_acc={acc:.3f}  conf={conf:.3f} ({'DET' if det else 'miss'})  "
              f"PESQ={pesq:.3f}  embed={embed_s:.1f}s")

        nb = 0 if n_bands is None else n_bands
        rows.append((name, acc, conf, pesq, det))
        writer.writerow([name, nb, BAND[0], BAND[1], tol, seglen, n_segs, segs_str,
                         round(acc, 4), round(float(conf), 4), det,
                         round(pesq, 4), round(embed_s, 2)])
    fout.close()

    print("\n" + "=" * 60)
    print(f"{'config':8s} {'bit_acc':>8s} {'conf':>8s} {'PESQ':>8s} {'det':>5s}")
    for name, acc, conf, pesq, det in rows:
        print(f"{name:8s} {acc:8.3f} {conf:8.3f} {pesq:8.3f} {det:5d}")
    print("=" * 60)
    print(f"wrote {csv_path}")
    print("read: want clean conf>=0.5 (detectable) AND high bit_acc as N grows; watch "
          "PESQ. If upper N collapse, the detector isn't steerable in 4-6 kHz -> "
          "lower --tol, use fewer bands, or narrow BAND back toward 4-5 kHz.")


if __name__ == "__main__":
    main(sys.argv[1:])
