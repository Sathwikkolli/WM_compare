"""
fsss/exp_v8_sensitivity.py -- detector SENSITIVITY (input-gradient) map: which
frequencies does AWARE's random detector actually respond to?

Measures |d(detector outputs)/d(magnitude)| at every frequency -- the "grip" the
per-file optimizer has when embedding. High grip = many good spots to nudge (easy
to embed there); ~0 grip = the detector is deaf to nudges there (can't embed) no
matter the loudness budget. This is the DIRECT proof of whether 500-1000 Hz is
reachable for the frozen random detector -- ONE forward+backward pass per clip,
NO training, NO embedding/optimization.

AWARE's detector is RANDOM (seed 328656719) and never trained, so the grip map is a
property of THAT random net. We ALSO re-seed the detector to a few other seeds to
see whether any random detector happens to have strong grip in 500-1000 -- the free
Path-A test (unlock the low band with a different seed, no training).

Per seed, two curves:
  raw grip(f) = mean_t || d(out)/d(mag[f,t]) ||_2   (L2 over the 20 bit-outputs)
  eff grip(f) = raw grip weighted by the allowed nudge (mag * 10^(-tol/20)) --
                "effective steering power" under the perceptual budget.
Reported relative to each seed's own 1000-4000 grip, so bands/seeds are comparable.

Run in wmcompare (CPU is fine -- it's one backprop per clip):
    conda activate wmcompare
    python -m fsss.exp_v8_sensitivity
    python -m fsss.exp_v8_sensitivity --seeds 328656719,1,7,42,123 --nclips 5
"""

import os
import sys
import csv
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(BASE, "cascade"))

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aware.utils.models import load
from aware.utils.utils import to_tensor
from fsss.exp_a_repeatability import load_16k, pick_clips, EMILIA_CSV, WORK_SR

DEFAULT_SEEDS = [328656719, 1, 7, 42, 123]     # shipped seed first, then a few others
BANDS = {"500-1000": (500, 1000), "1000-4000": (1000, 4000), "4000-6000": (4000, 6000)}
OUT_DIR = os.path.join(BASE, "fsss_out")


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def get_list(argv, flag, default, cast):
    if flag in argv:
        return [cast(x) for x in argv[argv.index(flag) + 1].split(",") if x.strip()]
    return default


def bar(x, width=30):
    n = int(round(max(0.0, min(1.0, x)) * width))
    return "#" * n + "." * (width - n)


def magnitude_of(embedder, audio):
    """Run AWARE's own preprocess pipeline -> STFT magnitude [n_freq, n_frames]."""
    x = to_tensor(audio).to(embedder.device)
    for p in embedder.audio_preprocess_pipeline:
        x = p(x)
    magnitude, _phase = x
    return magnitude.detach()


def grip_map(embedder, magnitude):
    """|d(out)/d(mag)| L2-combined over the 20 outputs -> grip [n_freq, n_frames]."""
    net = embedder.detection_net
    mag = magnitude.clone().requires_grad_(True)
    out = net(mag.unsqueeze(0)).squeeze()          # [20]
    n_out = int(out.numel())
    g2 = torch.zeros_like(mag)
    for i in range(n_out):
        gi = torch.autograd.grad(out[i], mag, retain_graph=(i < n_out - 1))[0]
        g2 = g2 + gi * gi
    return torch.sqrt(g2).detach(), mag.detach()


def reseed(embedder, seed):
    """Re-randomize the detector's weights with a new seed (mel buffer untouched)."""
    net = embedder.detection_net
    torch.manual_seed(int(seed))
    net.apply(net._init_weights)
    net.eval()


def main(argv):
    seeds = get_list(argv, "--seeds", DEFAULT_SEEDS, int)
    nclips = get_arg(argv, "--nclips", 3, int)
    tol = get_arg(argv, "--tol", -6.0, float)
    clip = get_arg(argv, "--clip", None)

    clips = [clip] if clip else pick_clips(EMILIA_CSV, nclips)
    if not clips:
        print("no Emilia clip found; pass --clip PATH")
        return
    audios = [load_16k(c)[:int(round(5.0 * WORK_SR))] for c in clips]

    embedder, _detector = load()
    freqs = librosa.fft_frequencies(sr=WORK_SR, n_fft=embedder.frame_length)   # [n_freq]
    budget = 10.0 ** (-tol / 20.0)

    def bandmean(v, lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return float(v[m].mean()) if m.any() else float("nan")

    print(f"clips    : {len(audios)}  (5s each)   tol={tol}")
    print(f"seeds    : {seeds}   (328656719 = AWARE shipped)")
    print("metric   : detector input-gradient 'grip' per freq, relative to 1000-4000=1.00\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}                                   # seed -> (raw[n_freq], eff[n_freq])
    for s in seeds:
        reseed(embedder, s)
        raws, effs = [], []
        for a in audios:
            mag = magnitude_of(embedder, a)
            grip, magd = grip_map(embedder, mag)
            delta = magd * budget                  # perceptually-allowed nudge
            raws.append(grip.mean(dim=1).cpu().numpy())
            effs.append((grip * delta).mean(dim=1).cpu().numpy())
        raw = np.mean(raws, axis=0)
        eff = np.mean(effs, axis=0)
        results[s] = (raw, eff)

        ref_raw = bandmean(raw, 1000, 4000)
        ref_eff = bandmean(eff, 1000, 4000)
        print(f"seed {s}{'  (shipped)' if s == 328656719 else ''}:")
        print(f"  {'band':10s} {'raw':>6s} {'eff':>6s}   raw grip (rel to 1000-4000)")
        for name, (lo, hi) in BANDS.items():
            rr = bandmean(raw, lo, hi) / ref_raw if ref_raw else float("nan")
            re = bandmean(eff, lo, hi) / ref_eff if ref_eff else float("nan")
            print(f"  {name:10s} {rr:6.2f} {re:6.2f}   {bar(rr)}")
        print()

    # ---- CSV --------------------------------------------------------------- #
    csv_path = os.path.join(OUT_DIR, "exp_v8_sensitivity.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        head = ["freq_hz"] + [f"raw_s{s}" for s in seeds] + [f"eff_s{s}" for s in seeds]
        w.writerow(head)
        for j, fz in enumerate(freqs):
            row = [round(float(fz), 2)]
            row += [round(float(results[s][0][j]), 8) for s in seeds]
            row += [round(float(results[s][1][j]), 8) for s in seeds]
            w.writerow(row)

    # ---- plot: raw grip vs freq, each seed normalized to its 1000-4000 mean - #
    plt.figure(figsize=(10, 5))
    for s in seeds:
        raw = results[s][0]
        norm = bandmean(raw, 1000, 4000) or 1.0
        plt.semilogy(freqs, raw / norm, lw=1.3,
                     label=f"seed {s}" + (" (shipped)" if s == 328656719 else ""))
    for lo, hi in BANDS.values():
        plt.axvspan(lo, hi, color="gray", alpha=0.06)
    plt.axvline(1000, color="k", ls=":", lw=0.6)
    plt.axvline(4000, color="k", ls=":", lw=0.6)
    plt.xlim(0, 8000)
    plt.xlabel("frequency (Hz)")
    plt.ylabel("detector grip (relative to 1000-4000)")
    plt.title("AWARE detector sensitivity vs frequency  |  which freqs can it be nudged in?")
    plt.legend(fontsize=8)
    plt.tight_layout()
    png = os.path.join(OUT_DIR, "exp_v8_sensitivity.png")
    plt.savefig(png, dpi=130)
    plt.close()

    print("=" * 66)
    print(f"wrote {csv_path}")
    print(f"wrote {png}")
    print("read: for the SHIPPED seed, if 500-1000 raw grip << 1.0 (e.g. <0.2) that is "
          "the PROOF the frozen detector can barely nudge there. If ANY other seed shows "
          "500-1000 grip ~1.0, the low band is reachable with NO training (Path A win).")


if __name__ == "__main__":
    main(sys.argv[1:])
