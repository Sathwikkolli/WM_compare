"""
fsss/viz_nsweep.py -- watermark difference spectrogram for N = 2,3,4,5,6 bands
over the (500, 6000) Hz range.

Embeds the same payload with the key-driven fixed-segment staircase at each band
count, then stacks the difference |wm| - |orig| for each -- so you can see how the
band-hopping gets finer as N grows, over the wider 500-6000 range.

Note: AWARE normally zeroes everything outside (500,4000) before the detector. To
actually put (and keep) the watermark in 500-6000, we widen `embedding_bands` to
match `band_range`. Whether the frozen detector can still be STEERED up in
4000-6000 is exactly what the printed conf/PESQ per N tells us.

Run on a GPU node in wmcompare:
    conda activate wmcompare
    python -m fsss.viz_nsweep
    python -m fsss.viz_nsweep --clip audio/x.wav --tol -6 --seg fixed
"""

import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

N_FFT, HOP = 1024, 256
WM_BITS = 20
BAND = (500, 6000)
N_LIST = [2, 3, 4, 5, 6]
OUT = os.path.join(BASE, "fsss_out", "viz")


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def main(argv):
    clip = get_arg(argv, "--clip", None)
    key = get_arg(argv, "--key", "thesis")
    tol = get_arg(argv, "--tol", -6, float)
    seg = get_arg(argv, "--seg", "fixed")
    if clip is None:
        clips = pick_clips(EMILIA_CSV, 1)
        if not clips:
            print("no clip found; pass --clip PATH")
            return
        clip = clips[0]

    audio = load_16k(clip)
    bits = np.random.default_rng(0).integers(0, 2, size=WM_BITS, dtype=np.int32)
    embedder, detector = load()
    pesq_metric = PESQ()

    results = []                                  # (N, diff, conf, pesq, dur)
    for N in N_LIST:
        st = StaircaseAWAREEmbedder.from_embedder(
            embedder, key=key, n_bands=N, band_range=BAND,
            segment_mode=seg, tolerance_db=tol)
        st.embedding_bands = BAND                 # keep 500-6000 (don't zero above 4k)
        wm = embed_watermark(audio, sample_rate=WORK_SR, watermark_bits=bits, model=st)
        n = min(len(audio), len(wm))
        Sa = np.abs(librosa.stft(audio[:n].astype(np.float32), n_fft=N_FFT, hop_length=HOP))
        Sw = np.abs(librosa.stft(wm[:n].astype(np.float32), n_fft=N_FFT, hop_length=HOP))
        _, conf = detect_watermark(wm, WORK_SR, detector)
        try:
            pesq = pesq_metric(wm, audio, WORK_SR)
        except Exception:
            pesq = float("nan")
        results.append((N, Sw - Sa, float(conf), pesq, n / WORK_SR))
        print(f"N={N}  conf={conf:.3f}  PESQ={pesq:.3f}")

    # shared color scale across panels for fair comparison
    vlim = max(np.percentile(np.abs(d), 99.5) for _, d, _, _, _ in results) + 1e-9

    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(len(results) + 1, 1, figsize=(11, 2.1 * (len(results) + 1)),
                           sharex=True)
    n0 = min(len(audio), int(results[0][4] * WORK_SR))
    Sa0 = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio[:n0].astype(np.float32), n_fft=N_FFT, hop_length=HOP)),
        ref=np.max)
    ax[0].imshow(Sa0, origin="lower", aspect="auto",
                 extent=[0, results[0][4], 0, WORK_SR / 2], cmap="magma")
    ax[0].set_title("original spectrogram")
    ax[0].set_ylabel("Hz")

    for i, (N, diff, conf, pesq, dur) in enumerate(results, start=1):
        ax[i].imshow(diff, origin="lower", aspect="auto", extent=[0, dur, 0, WORK_SR / 2],
                     cmap="seismic", vmin=-vlim, vmax=vlim)
        ax[i].set_title(f"diff = watermark  |  N={N}  |  conf={conf:.2f}  PESQ={pesq:.2f}")
        ax[i].set_ylabel("Hz")

    for a_ in ax:
        a_.set_ylim(0, 6500)
        a_.axhline(BAND[0], color="cyan", lw=0.6, ls=":")
        a_.axhline(BAND[1], color="cyan", lw=0.6, ls=":")
    ax[-1].set_xlabel("time (s)")

    path = os.path.join(OUT, "watermark_nsweep.png")
    fig.suptitle(f"watermark vs N (bands)  |  {os.path.basename(clip)}  |  "
                 f"range {BAND[0]}-{BAND[1]} Hz  tol={tol}  |  cyan = band edges", y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
