"""
fsss/viz_watermark.py -- show WHERE the watermark lives in the audio.

Watermarks a clip with the key-driven staircase embedder, then plots four stacked
panels (0-5 kHz) sharing a time axis:

  1. original spectrogram           -- the host audio
  2. watermarked spectrogram        -- host + hidden mark (looks ~identical)
  3. difference |wm| - |orig|       -- THE WATERMARK: exactly the energy the
                                       embedder added/removed. Reveals the
                                       band-hopping staircase.
  4. key-driven mask M              -- where writing was ALLOWED (ground truth),
                                       for comparison with panel 3.

The difference panel is the point: for N=2 you should see energy alternating
between the lower and upper half of the (500,4000) band across time segments --
the key hopping made visible.

Run on a GPU node in wmcompare:
    conda activate wmcompare
    python -m fsss.viz_watermark                       # first Emilia clip, N=2 loud
    python -m fsss.viz_watermark --clip audio/x.wav --n-bands 2 --tol -6 --seg fixed
    python -m fsss.viz_watermark --seg librosa         # anchor-driven segments
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
import librosa.display
from aware.utils.models import load
from aware.service import embed_watermark, detect_watermark

from fsss.staircase import StaircaseAWAREEmbedder
from fsss.exp_a_repeatability import load_16k, pick_clips, EMILIA_CSV, WORK_SR

N_FFT, HOP = 1024, 256          # match AWARE's STFT so the staircase aligns
WM_BITS = 20
OUT = os.path.join(BASE, "fsss_out", "viz")


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def main(argv):
    clip = get_arg(argv, "--clip", None)
    key = get_arg(argv, "--key", "thesis")
    n_bands = get_arg(argv, "--n-bands", 2, int)
    tol = get_arg(argv, "--tol", -6, float)     # loud by default so the mark is visible
    seg = get_arg(argv, "--seg", "fixed")       # fixed | librosa
    if clip is None:
        clips = pick_clips(EMILIA_CSV, 1)
        if not clips:
            print("no clip found; pass --clip PATH")
            return
        clip = clips[0]

    audio = load_16k(clip)
    bits = np.random.default_rng(0).integers(0, 2, size=WM_BITS, dtype=np.int32)

    embedder, detector = load()
    st = StaircaseAWAREEmbedder.from_embedder(
        embedder, key=key, n_bands=n_bands, segment_mode=seg, tolerance_db=tol)
    wm = embed_watermark(audio, sample_rate=WORK_SR, watermark_bits=bits, model=st)

    # trim to common length, then STFT both
    n = min(len(audio), len(wm))
    a, w = audio[:n].astype(np.float32), wm[:n].astype(np.float32)
    Sa = np.abs(librosa.stft(a, n_fft=N_FFT, hop_length=HOP))
    Sw = np.abs(librosa.stft(w, n_fft=N_FFT, hop_length=HOP))
    diff = Sw - Sa                                     # signed: added (+) / removed (-)

    # the key-driven mask, aligned to this STFT
    M = st._build_staircase_mask(a, WORK_SR, Sa.shape[0], Sa.shape[1]).astype(float)

    _, conf = detect_watermark(wm, WORK_SR, detector)
    dur = n / WORK_SR
    extent = [0, dur, 0, WORK_SR / 2]
    vlim = np.percentile(np.abs(diff), 99.5) + 1e-9

    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True)

    for i, (S, title, cmap) in enumerate([
        (librosa.amplitude_to_db(Sa, ref=np.max), "1. original spectrogram", "magma"),
        (librosa.amplitude_to_db(Sw, ref=np.max), "2. watermarked spectrogram", "magma"),
    ]):
        im = ax[i].imshow(S, origin="lower", aspect="auto", extent=extent, cmap=cmap)
        ax[i].set_title(title)
        ax[i].set_ylabel("Hz")
        fig.colorbar(im, ax=ax[i], format="%+2.0f dB")

    im2 = ax[2].imshow(diff, origin="lower", aspect="auto", extent=extent,
                       cmap="seismic", vmin=-vlim, vmax=vlim)
    ax[2].set_title(f"3. difference |wm| - |orig|  =  THE WATERMARK   (conf={conf:.2f})")
    ax[2].set_ylabel("Hz")
    fig.colorbar(im2, ax=ax[2], label="magnitude added / removed")

    im3 = ax[3].imshow(M, origin="lower", aspect="auto", extent=extent,
                       cmap="Greens", vmin=0, vmax=1)
    ax[3].set_title(f"4. key-driven mask M  (where writing was allowed; N={n_bands}, {seg})")
    ax[3].set_ylabel("Hz")
    ax[3].set_xlabel("time (s)")
    fig.colorbar(im3, ax=ax[3], label="allowed (1) / blocked (0)")

    for a_ in ax:
        a_.set_ylim(0, 5000)                           # zoom to the interesting band
        a_.axhline(500, color="cyan", lw=0.6, ls=":")
        a_.axhline(4000, color="cyan", lw=0.6, ls=":")

    path = os.path.join(OUT, "watermark_location.png")
    fig.suptitle(f"where the watermark lives  |  {os.path.basename(clip)}  |  "
                 f"N={n_bands} tol={tol}  |  cyan dotted = (500,4000) band", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"clip: {clip}")
    print(f"detection conf={conf:.3f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
