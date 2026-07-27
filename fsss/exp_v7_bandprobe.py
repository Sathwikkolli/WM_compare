"""
fsss/exp_v7_bandprobe.py -- map the AWARE detector's READABLE frequency range by
embedding the watermark in ONE narrow band at a time and sweeping that band across
the spectrum. Answers: "what actually happens in 500-1000 Hz?"

BACKGROUND (from the official AWARE code + papers)
--------------------------------------------------
The detector's front end is a Mel filter bank with fmin=0, fmax=sr/2=8000
(src/aware/detection/modules/mel.py) -> it SEES all of 0-8 kHz, and Mel is dense
at low freq (linear below 1000 Hz => ~15 of 128 bins under 1 kHz). So 500-1000 Hz
is NOT invisible to the detector. But the detector was TRAINED with the watermark
in embedding_bands=[1000,4000] (config_full_length.yaml), and its readout is 1x1
convs over Mel channels + temporal pooling (BRH) -> the learned weights only decode
the Mel channels that carried bits in training. Energy planted at 500-1000 is seen
but not decoded. Timbre, by contrast, TRAINS a joint encoder+decoder in the
medium-to-low band, so its decoder reads there fine (Detecting Voice Cloning Attacks
via Timbre Watermarking, NDSS 2024). Same physics, different training.

THIS EXPERIMENT
---------------
Embed ONLY in a single width-Hz band (n_bands=1, band_range=embedding_bands=band),
sweep the band across 500..5000 Hz, and read clean conf / bit_acc. The result is
the detector's empirical frequency response: where conf is high, the detector can
read; where it collapses, it cannot. Prediction: high in ~1000-4000 (trained),
low below 1000 and above 4000 (untrained) -- even though the low band is dense in
Mel and robust to codecs. That gap is a TRAINING boundary, not a physics one, and
is the direct motivation for retraining the detector to use 500-1000 (+ 4-6 kHz).

Run on a GPU node in wmcompare (AWARE + Emilia on Great Lakes):
    conda activate wmcompare
    python -m fsss.exp_v7_bandprobe
    python -m fsss.exp_v7_bandprobe --width 500 --start 250 --stop 5000 --step 250
    python -m fsss.exp_v7_bandprobe --attacks mp3,lowpass        # per-band robustness too
"""

import os
import sys
import csv
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
DET_CONF = 0.5
OUT_DIR = os.path.join(BASE, "fsss_out")


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


def bins_in(band, frame_length):
    freqs = librosa.fft_frequencies(sr=WORK_SR, n_fft=frame_length)
    return int(((freqs >= band[0]) & (freqs <= band[1])).sum())


def bar(x, width=34):
    n = int(round(max(0.0, min(1.0, x)) * width))
    return "#" * n + "." * (width - n)


def main(argv):
    clip = get_arg(argv, "--clip", None)
    key = get_arg(argv, "--key", "thesis")
    seed = get_arg(argv, "--seed", 0, int)
    dur = get_arg(argv, "--dur", 5.0, float)
    tol = get_arg(argv, "--tol", -6.0, float)
    width = get_arg(argv, "--width", 500, int)
    start = get_arg(argv, "--start", 500, int)
    stop = get_arg(argv, "--stop", 5000, int)
    step = get_arg(argv, "--step", 500, int)
    attacks = get_list(argv, "--attacks", [], str)

    widths = get_list(argv, "--widths", None, int)
    center = get_arg(argv, "--center", 2500, int)
    if widths:                                    # centered WIDTH sweep (find readable threshold)
        nyq = WORK_SR // 2 - 1
        windows = [(max(50, center - w // 2), min(nyq, center + w // 2)) for w in widths]
        mode_desc = f"centered WIDTH-sweep at {center} Hz, widths {widths} Hz"
    else:                                         # single-width POSITION slide (default)
        windows = [(f, f + width) for f in range(start, stop, step) if f + width <= stop + 1]
        mode_desc = f"single {width} Hz band, POSITION slide {start}-{stop} Hz (step {step})"

    if clip is None:
        clips = pick_clips(EMILIA_CSV, 1)
        if not clips:
            print("no Emilia clip found; pass --clip PATH")
            return
        clip = clips[0]

    audio = load_16k(clip)[:int(round(dur * WORK_SR))]
    bits = np.random.default_rng(seed).integers(0, 2, size=WM_BITS, dtype=np.int32)

    print(f"clip     : {clip}")
    print(f"duration : {len(audio)/WORK_SR:.2f}s   tol={tol}   key='{key}'")
    print(f"probing  : {mode_desc}  ({len(windows)} windows)")
    print("NOTE: detector Mel fmin=0 fmax=8000 (sees all), trained band [1000,4000].")

    embedder, detector = load()
    pesq_metric = PESQ()
    if attacks:
        import vox_attacks

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "exp_v7_bandprobe.csv")
    fout = open(csv_path, "w", newline="")
    writer = csv.writer(fout)
    cols = ["band_lo", "band_hi", "bins", "bit_acc", "conf", "detected", "pesq"]
    for a in attacks:
        cols += [f"{a}_meanacc", f"{a}_det"]
    writer.writerow(cols)

    # stock reference
    wm0 = embed_watermark(audio, sample_rate=WORK_SR, watermark_bits=bits, model=embedder)
    p0, c0 = detect_watermark(wm0, WORK_SR, detector)
    print(f"\nstock (normal AWARE, its card band): conf={c0:.3f} bit_acc={bit_acc(bits,p0):.3f}\n")

    print(f"{'band (Hz)':>13s} {'bins':>4s} {'conf':>6s} {'acc':>5s}  detector-readable?")
    rows = []
    for band in windows:
        st = StaircaseAWAREEmbedder.from_embedder(
            embedder, key=key, n_bands=1, band_range=band,
            segment_mode="fixed", segment_len_s=0.5, tolerance_db=tol)
        st.embedding_bands = band
        nb = bins_in(band, st.frame_length)
        wm = embed_watermark(audio, sample_rate=WORK_SR, watermark_bits=bits, model=st)
        pat, conf = detect_watermark(wm, WORK_SR, detector)
        acc = bit_acc(bits, pat)
        m = min(len(audio), len(wm))
        try:
            pesq = float(pesq_metric(wm[:m], audio[:m], WORK_SR))
        except Exception:
            pesq = float("nan")
        det = int(conf >= DET_CONF)

        att_cells = []
        if attacks:
            for a in attacks:
                accs, dets = [], []
                for label, param in vox_attacks.VOX_GRID.get(a, []):
                    try:
                        wa = vox_attacks.apply(a, param, wm.astype("float32"), WORK_SR)
                    except Exception:
                        continue
                    if wa is None:
                        continue
                    pp, cc = detect_watermark(wa, WORK_SR, detector)
                    accs.append(bit_acc(bits, pp)); dets.append(int(cc >= DET_CONF))
                att_cells += [round(float(np.mean(accs)), 3) if accs else "",
                              f"{sum(dets)}/{len(dets)}" if dets else ""]

        flag = "YES" if det else ("weak" if conf >= 0.2 else "NO")
        print(f"{band[0]:5d}-{band[1]:<5d}  {nb:4d} {conf:6.3f} {acc:5.2f}  "
              f"{bar(conf)} {flag}")
        writer.writerow([band[0], band[1], nb, round(acc, 4), round(float(conf), 4),
                         det, round(pesq, 4)] + att_cells)
        rows.append((band, conf, acc, det))
    fout.close()

    # readable-band summary
    readable = [b for b, c, a, d in rows if d]
    print("\n" + "=" * 60)
    if readable:
        lo = min(b[0] for b in readable); hi = max(b[1] for b in readable)
        print(f"detector-READABLE span (conf>=0.5 single-band): {lo}-{hi} Hz")
    else:
        print("no single band read on its own at this tol")
    print(f"wrote {csv_path}")
    print("read: high conf = detector was TRAINED to read there; the low-freq (500-1000) "
          "and high-freq (>4000) collapse is a TRAINING boundary, not physics -- Timbre "
          "reads low freq because its decoder was trained for it. Motivates retraining.")


if __name__ == "__main__":
    main(sys.argv[1:])
