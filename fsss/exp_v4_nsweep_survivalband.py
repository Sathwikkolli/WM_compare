"""
fsss/exp_v4_nsweep_survivalband.py -- key-driven subband-hopping N-sweep over the
band where the watermark actually SURVIVES BEST, on ONE 5-second Emilia clip.

WHY THIS BAND (1000-4000 Hz by default)
---------------------------------------
v3 (exp_v3_nsweep_wideband.py) pushed the range to (500,6000) and everything above
~4 kHz was undetectable: the FROZEN AWARE detector is only responsive in its
trained band. Confirmed from the official code -- load("AWARE") reads
config_full_length.yaml whose `embedding_bands: [1000, 4000]`. That range is ALSO
the one the AWARE authors chose "to avoid removal by low/high-pass filters"
(arXiv:2510.17512), and the classical robustness literature agrees the low/mid band
survives codecs/filtering while high subbands are fragile (FSSS excludes its top
octave; DWT high subbands lose accuracy). So (1000,4000) is where detection lives
AND where the mark survives attacks -- the evidence-based "survives best" choice.

This script is v3 with the band made a first-class knob:
  * `--band LO HI` (default 1000 4000). Sets BOTH band_range (staircase tiling) and
    embedding_bands (base AWARE zero-mask) so the detector sees exactly this band.
  * reports bins-per-subband, so you can see when N gets too fine for the band.
Design decisions unchanged from v3 (literature-grounded): key-driven PN hop
SEQUENCE (HMAC, repeats allowed, not a permutation) over FIXED equal dwell
segments; detection with the STOCK AWARE detector.

The test: staying inside the survivable band, does bit_acc hold as N grows (unlike
v3, where >4 kHz killed N>=3)? If N=3..6 now survive, the ceiling was the dead high
band, not the hopping itself.

Run on a GPU node in wmcompare (AWARE + Emilia both on Great Lakes):
    conda activate wmcompare
    python -m fsss.exp_v4_nsweep_survivalband
    python -m fsss.exp_v4_nsweep_survivalband --band 1000 4000 --tol -6
    python -m fsss.exp_v4_nsweep_survivalband --band 500 4000   # wider low edge
    python -m fsss.exp_v4_nsweep_survivalband --clip audio/client_original_16k.wav
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
DEFAULT_BAND = (1000, 4000)          # detector-native + filter-robust "survives best"
N_LIST = [2, 3, 4, 5, 6]
OUT_DIR = os.path.join(BASE, "fsss_out")


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def get_band(argv):
    """--band LO HI -> (int, int); falls back to DEFAULT_BAND."""
    if "--band" in argv:
        i = argv.index("--band")
        return (int(argv[i + 1]), int(argv[i + 2]))
    return DEFAULT_BAND


def bit_acc(bits, pattern):
    p = np.asarray(pattern).astype(int).ravel()
    b = np.asarray(bits).astype(int).ravel()
    n = min(len(b), len(p))
    return float(np.mean(b[:n] == p[:n])) if n else float("nan")


def band_edges_hz(band, n_bands):
    return np.linspace(band[0], band[1], n_bands + 1)


def bins_per_subband(band, n_bands, frame_length):
    """FFT-bin count in each equal-Hz sub-band (capacity sanity per segment)."""
    freqs = librosa.fft_frequencies(sr=WORK_SR, n_fft=frame_length)
    edges = band_edges_hz(band, n_bands)
    counts = []
    for b in range(n_bands):
        hi_incl = b == n_bands - 1
        m = (freqs >= edges[b]) & ((freqs <= edges[b + 1]) if hi_incl else (freqs < edges[b + 1]))
        counts.append(int(m.sum()))
    return counts


def hop_schedule(st, audio):
    """Exact per-segment band sequence (reuses the staircase's own logic)."""
    S = librosa.stft(audio.astype(np.float32),
                     n_fft=st.frame_length, hop_length=st.hop_length)
    segs = st._segments(audio, WORK_SR, S.shape[1])
    bands = [st._hmac_band(i) for i in range(len(segs))]     # 0-indexed
    return segs, bands


def fmt_hops(bands):
    return " -> ".join(str(b + 1) for b in bands)            # 1-indexed for readability


def main(argv):
    clip = get_arg(argv, "--clip", None)
    key = get_arg(argv, "--key", "thesis")
    seed = get_arg(argv, "--seed", 0, int)
    dur = get_arg(argv, "--dur", 5.0, float)
    tol = get_arg(argv, "--tol", -6.0, float)
    seglen = get_arg(argv, "--seglen", 0.5, float)
    band = get_band(argv)

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
    audio = audio[:n_keep]

    bits = np.random.default_rng(seed).integers(0, 2, size=WM_BITS, dtype=np.int32)
    print(f"clip     : {clip}")
    print(f"duration : {len(audio)/WORK_SR:.2f}s   ({len(audio)} samples @ {WORK_SR} Hz)")
    print(f"payload  : {bits.tolist()}")
    print(f"band     : {band[0]}-{band[1]} Hz (survives-best)   key='{key}'   "
          f"tol={tol}   seglen={seglen}s")

    embedder, detector = load()
    pesq_metric = PESQ()

    configs = [("stock", None), ("n1", 1)] + [(f"n{n}", n) for n in N_LIST]

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "exp_v4_nsweep_survivalband.csv")
    fout = open(csv_path, "w", newline="")
    writer = csv.writer(fout)
    writer.writerow(["config", "n_bands", "band_lo", "band_hi", "tol_db", "seglen_s",
                     "n_segments", "bins_per_subband", "hop_sequence", "bit_acc",
                     "conf", "detected", "pesq", "embed_s"])

    rows = []
    for name, n_bands in configs:
        print(f"\n=== {name} ===")
        if n_bands is None:                            # plain stock AWARE (its own card band)
            model = embedder
            segs_str, n_segs, bpb = "", 0, ""
        else:
            model = StaircaseAWAREEmbedder.from_embedder(
                embedder, key=key, n_bands=n_bands, band_range=band,
                segment_mode="fixed", segment_len_s=seglen, tolerance_db=tol)
            model.embedding_bands = band               # detector zero-mask = this band
            edges = band_edges_hz(band, n_bands)
            counts = bins_per_subband(band, n_bands, model.frame_length)
            bpb = "/".join(str(c) for c in counts)
            print("  sub-bands: " + ", ".join(
                f"{i+1}:[{edges[i]:.0f}-{edges[i+1]:.0f}|{counts[i]}bins]"
                for i in range(n_bands)))
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
        writer.writerow([name, nb, band[0], band[1], tol, seglen, n_segs, bpb, segs_str,
                         round(acc, 4), round(float(conf), 4), det,
                         round(pesq, 4), round(embed_s, 2)])
    fout.close()

    print("\n" + "=" * 60)
    print(f"{'config':8s} {'bit_acc':>8s} {'conf':>8s} {'PESQ':>8s} {'det':>5s}")
    for name, acc, conf, pesq, det in rows:
        print(f"{name:8s} {acc:8.3f} {conf:8.3f} {pesq:8.3f} {det:5d}")
    print("=" * 60)
    print(f"wrote {csv_path}")
    print(f"read: inside the survivable {band[0]}-{band[1]} Hz band, want conf>=0.5 AND "
          "bit_acc high as N grows. If higher N still fail, it's the per-segment bin "
          "budget (see bins_per_subband) -> fewer bands, lower --tol, or wider --band.")


if __name__ == "__main__":
    main(sys.argv[1:])
