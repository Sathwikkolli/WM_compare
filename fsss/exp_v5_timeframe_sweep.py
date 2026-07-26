"""
fsss/exp_v5_timeframe_sweep.py -- find the best TIME FRAME (dwell length) for
key-driven subband hopping: how long the watermark stays in one sub-band before it
hops. Sweeps the fixed-segment dwell at a fixed N, on ONE 5-second Emilia clip.

WHY SWEEP DWELL, AND WHAT TO EXPECT
-----------------------------------
The frequency dimension (N) was settled in v4 (N=3 sweet spot over 1000-4000 Hz).
This is the TIME dimension: the dwell = segment_len_s controls the hop rate.

Literature reference points (annotated in the sweep):
  * ~0.023 s  -- FHSS-watermark 1024-sample block @44.1 kHz (Anastasijevic & Coja,
                 "Frequency Hopping Method for Audio Watermarking"); ~0.064 s is the
                 same 1024-sample block at AWARE's 16 kHz. FAST hopping.
  * ~0.25-0.33 s -- FSSS (Malik) 3-4 salient points/sec. CONTENT-driven rate.
  * ~1.0 s    -- AWARE's own segment-mode partition. SLOW hopping.

Physics caveat (be honest): at FIXED N the TOTAL writable budget is ~constant
regardless of dwell -- every STFT frame writes exactly one sub-band's bins either
way. So on CLEAN audio, dwell should matter only modestly. Dwell's real payoff is
temporal REDUNDANCY under desync/crop. Hence this sweep measures BOTH clean AND a
center-crop per dwell -- the crop column is where "perfect time frame" actually
shows up (finer hopping -> more time-diversity -> should survive cropping better,
though AWARE's BRH is already crop-tolerant, so watch whether dwell modulates it).

Design otherwise unchanged from v4: key-driven PN hop sequence (HMAC, repeats
allowed) over fixed dwell segments; STOCK AWARE detector; survival band 1000-4000.

Run on a GPU node in wmcompare (AWARE + Emilia on Great Lakes):
    conda activate wmcompare
    python -m fsss.exp_v5_timeframe_sweep
    python -m fsss.exp_v5_timeframe_sweep --nbands 3 --crop 0.6
    python -m fsss.exp_v5_timeframe_sweep --seglens 0.064,0.25,0.5,1.0 --band 500 4000
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
DEFAULT_BAND = (1000, 4000)
DEFAULT_SEGLENS = [0.032, 0.064, 0.125, 0.25, 0.5, 1.0, 2.5]
# paper anchors for the printout (seconds -> label)
ANCHORS = {0.064: "FHSS-block@16k", 0.25: "FSSS 3-4/s", 1.0: "AWARE-seg"}
OUT_DIR = os.path.join(BASE, "fsss_out")


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def get_band(argv):
    if "--band" in argv:
        i = argv.index("--band")
        return (int(argv[i + 1]), int(argv[i + 2]))
    return DEFAULT_BAND


def get_seglens(argv):
    if "--seglens" in argv:
        raw = argv[argv.index("--seglens") + 1]
        return [float(x) for x in raw.split(",") if x.strip()]
    return DEFAULT_SEGLENS


def bit_acc(bits, pattern):
    p = np.asarray(pattern).astype(int).ravel()
    b = np.asarray(bits).astype(int).ravel()
    n = min(len(b), len(p))
    return float(np.mean(b[:n] == p[:n])) if n else float("nan")


def center_crop(x, frac):
    """Keep the central `frac` of the signal (desync/crop robustness probe)."""
    if frac >= 1.0:
        return x
    keep = int(round(len(x) * frac))
    start = (len(x) - keep) // 2
    return x[start:start + keep]


def hop_schedule(st, audio):
    S = librosa.stft(audio.astype(np.float32),
                     n_fft=st.frame_length, hop_length=st.hop_length)
    segs = st._segments(audio, WORK_SR, S.shape[1])
    bands = [st._hmac_band(i) for i in range(len(segs))]
    return segs, bands


def fmt_hops(bands, cap=16):
    s = [str(b + 1) for b in bands]
    if len(s) > cap:
        s = s[:cap] + ["..."]
    return " -> ".join(s)


def main(argv):
    clip = get_arg(argv, "--clip", None)
    key = get_arg(argv, "--key", "thesis")
    seed = get_arg(argv, "--seed", 0, int)
    dur = get_arg(argv, "--dur", 5.0, float)
    tol = get_arg(argv, "--tol", -6.0, float)
    n_bands = get_arg(argv, "--nbands", 3, int)
    crop = get_arg(argv, "--crop", 0.6, float)
    band = get_band(argv)
    seglens = get_seglens(argv)

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
    frame_s = 256 / WORK_SR  # AWARE hop_length/sr; informational
    print(f"clip     : {clip}")
    print(f"duration : {len(audio)/WORK_SR:.2f}s   (1 STFT frame ~= {frame_s*1000:.0f} ms)")
    print(f"payload  : {bits.tolist()}")
    print(f"fixed    : N={n_bands} bands, band {band[0]}-{band[1]} Hz, tol={tol}, "
          f"crop={crop}   key='{key}'")
    print(f"sweeping seglen (s): {seglens}")

    embedder, detector = load()
    pesq_metric = PESQ()

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "exp_v5_timeframe_sweep.csv")
    fout = open(csv_path, "w", newline="")
    writer = csv.writer(fout)
    writer.writerow(["seglen_s", "anchor", "n_segments", "frames_per_seg", "bit_acc",
                     "conf", "detected", "pesq", f"bit_acc_crop{crop}", "conf_crop",
                     "detected_crop", "hop_sequence", "embed_s"])

    rows = []
    for sl in seglens:
        anchor = ANCHORS.get(sl, "")
        frames_per_seg = max(1, int(sl * WORK_SR / 256))
        tag = f"seglen={sl}s" + (f" ({anchor})" if anchor else "")
        print(f"\n=== {tag} ===")
        st = StaircaseAWAREEmbedder.from_embedder(
            embedder, key=key, n_bands=n_bands, band_range=band,
            segment_mode="fixed", segment_len_s=sl, tolerance_db=tol)
        st.embedding_bands = band

        segs, hb = hop_schedule(st, audio)
        n_segs = len(segs)
        print(f"  {n_segs} segments, ~{frames_per_seg} frames/seg, hop: {fmt_hops(hb)}")

        t0 = time.time()
        wm = embed_watermark(audio, sample_rate=WORK_SR, watermark_bits=bits, model=st)
        embed_s = time.time() - t0

        # clean detection
        pat, conf = detect_watermark(wm, WORK_SR, detector)
        acc = bit_acc(bits, pat)
        m = min(len(audio), len(wm))
        try:
            pesq = float(pesq_metric(wm[:m], audio[:m], WORK_SR))
        except Exception as e:
            pesq = float("nan")
            print(f"  PESQ failed: {e}")

        # crop robustness (the dwell-sensitive test)
        wc = center_crop(wm, crop)
        pat_c, conf_c = detect_watermark(wc, WORK_SR, detector)
        acc_c = bit_acc(bits, pat_c)

        det, det_c = int(conf >= 0.5), int(conf_c >= 0.5)
        print(f"  clean : bit_acc={acc:.3f} conf={conf:.3f} ({'DET' if det else 'miss'})"
              f"  PESQ={pesq:.3f}")
        print(f"  crop{crop}: bit_acc={acc_c:.3f} conf={conf_c:.3f} "
              f"({'DET' if det_c else 'miss'})")

        rows.append((sl, anchor, acc, conf, det, pesq, acc_c, conf_c, det_c))
        writer.writerow([sl, anchor, n_segs, frames_per_seg, round(acc, 4),
                         round(float(conf), 4), det, round(pesq, 4), round(acc_c, 4),
                         round(float(conf_c), 4), det_c, fmt_hops(hb, cap=40),
                         round(embed_s, 2)])
    fout.close()

    print("\n" + "=" * 78)
    print(f"{'seglen':>8s} {'anchor':>14s} {'bit_acc':>8s} {'conf':>7s} {'PESQ':>7s} "
          f"{'acc_crop':>9s} {'conf_crop':>10s}")
    for sl, anchor, acc, conf, det, pesq, acc_c, conf_c, det_c in rows:
        print(f"{sl:8.3f} {anchor:>14s} {acc:8.3f} {conf:7.3f} {pesq:7.3f} "
              f"{acc_c:9.3f} {conf_c:10.3f}")
    print("=" * 78)
    print(f"wrote {csv_path}")
    print("read: clean is likely flat (budget ~const vs dwell); the DECIDER is the "
          "crop columns -- pick the dwell that keeps conf_crop/acc_crop highest. That "
          "is the 'perfect time frame'. Then confirm with the full attack sweep.")


if __name__ == "__main__":
    main(sys.argv[1:])
