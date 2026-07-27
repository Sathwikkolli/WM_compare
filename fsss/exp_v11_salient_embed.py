"""
fsss/exp_v11_salient_embed.py -- embed the AWARE watermark with KEY-DRIVEN SUBBAND
HOPPING whose hop boundaries come from SALIENT POINTS, and detect it CLEAN (no
attacks). Compares the three salient-point methods {fsss, librosa_flux, ssl_wavlm}
plus a fixed-0.5s baseline, across three bands {1000-4000, 500-4000, 500-5000},
on three Emilia clips, at N=3 sub-bands.

WHAT VARIES: the segmentation (WHEN the key is allowed to hop). The key
(HMAC(key, seg_i) % N) still chooses WHICH sub-band each segment gets, and N=3 so a
segment boundary can actually change the band -- otherwise (N=1) the salient method
would leave no fingerprint (see the accompanying diagram).

DETECTOR-BAND FIX applied (v10): embedding_bands set on BOTH embedder and detector
so every band is embedded and read consistently.

Outputs:
  fsss_out/exp_v11_salient_embed.csv          per (clip, band, method) clean result
  fsss_out/v11_salient_points_<clip>.png      salient points on each clip
  fsss_out/v11_hopmask_<clip>.png             the key-driven staircase per method
  fsss_out/v11_summary_heatmap.png            method x band mean clean confidence

Run on a GPU node (ssl_wavlm pulls WavLM; it SKIPs gracefully if unavailable):
    conda activate wmcompare
    python -m fsss.exp_v11_salient_embed
    python -m fsss.exp_v11_salient_embed --nbands 2 --tol -6 --no-viz
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

from aware.utils.models import load
from aware.service import embed_watermark, detect_watermark

from fsss.staircase import StaircaseAWAREEmbedder
from fsss.detectors import DETECTORS
from fsss.exp_a_repeatability import load_16k, pick_clips, EMILIA_CSV, WORK_SR
from fsss import viz_salient as viz

WM_BITS = 20
DEFAULT_BANDS = [(1000, 4000), (500, 4000), (500, 5000)]
# (name, segment_mode, anchor). fixed = clock baseline; others = anchor-driven hops.
METHODS = [
    ("fixed",        "fixed",   None),
    ("fsss",         "librosa", "fsss"),
    ("librosa_flux", "librosa", "librosa_flux"),
    ("ssl_wavlm",    "librosa", "ssl_wavlm"),
]
MASK_BAND = (500, 4000)     # band used for the hop-mask figure
DET_CONF = 0.5
OUT_DIR = os.path.join(BASE, "fsss_out")


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def get_bands(argv):
    if "--bands" in argv:
        out = []
        for tok in argv[argv.index("--bands") + 1].split(","):
            lo, hi = tok.split("-")
            out.append((int(lo), int(hi)))
        return out
    return DEFAULT_BANDS


def bit_acc(bits, pattern):
    p = np.asarray(pattern).astype(int).ravel()
    b = np.asarray(bits).astype(int).ravel()
    n = min(len(b), len(p))
    return float(np.mean(b[:n] == p[:n])) if n else float("nan")


def available_methods(probe_audio, sr, rate):
    """Drop any anchor method whose detector dependency is missing on this box."""
    out = []
    for (name, mode, anchor) in METHODS:
        if anchor is None:
            out.append((name, mode, anchor))
            continue
        try:
            DETECTORS[anchor](probe_audio, sr, target_rate=rate)
            out.append((name, mode, anchor))
        except Exception as e:
            print(f"  [skip] anchor '{name}' unavailable ({type(e).__name__}: {e})")
    return out


def main(argv):
    seed = get_arg(argv, "--seed", 0, int)
    dur = get_arg(argv, "--dur", 5.0, float)
    nclips = get_arg(argv, "--nclips", 3, int)
    tol = get_arg(argv, "--tol", -6.0, float)
    nbands = get_arg(argv, "--nbands", 3, int)
    seglen = get_arg(argv, "--seglen", 0.5, float)
    rate = get_arg(argv, "--rate", 3.5, float)
    key = get_arg(argv, "--key", "thesis")
    do_viz = "--no-viz" not in argv
    bands = [(f"{lo}-{hi}", (lo, hi)) for (lo, hi) in get_bands(argv)]

    clips = pick_clips(EMILIA_CSV, nclips)
    if not clips:
        print("no Emilia clips found")
        return
    audios = [load_16k(c)[:int(round(dur * WORK_SR))] for c in clips]
    names = [os.path.splitext(os.path.basename(c))[0] for c in clips]
    bits = np.random.default_rng(seed).integers(0, 2, size=WM_BITS, dtype=np.int32)

    print(f"clips    : {len(audios)}   bands: {[b[0] for b in bands]}")
    print(f"N        : {nbands}   seg methods: {[m[0] for m in METHODS]}")
    print(f"tol      : {tol}   seglen: {seglen}   rate: {rate}/s   key: '{key}'")
    print("resolving anchor availability...")

    embedder, detector = load()
    methods = available_methods(audios[0], WORK_SR, rate)
    anchors = [a for (_, mode, a) in methods if a is not None]
    n_fft, hop = embedder.frame_length, embedder.hop_length

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "exp_v11_salient_embed.csv")
    fout = open(csv_path, "w", newline="")
    writer = csv.writer(fout)
    writer.writerow(["clip", "clip_name", "band", "method", "n_segments",
                     "bit_acc", "conf", "detected", "status"])

    agg = {}    # (method, band_name) -> [(acc, conf, det), ...]

    for ci, (name, audio) in enumerate(zip(names, audios)):
        nfr = viz._n_frames(audio, n_fft, hop)
        print(f"\nclip {ci+1}/{len(audios)}: {name}")
        for bn, band in bands:
            detector.embedding_bands = band
            for (mname, mode, anchor) in methods:
                try:
                    st = StaircaseAWAREEmbedder.from_embedder(
                        embedder, key=key, n_bands=nbands, band_range=band,
                        segment_mode=mode, anchor=(anchor or "librosa_flux"),
                        anchor_rate=rate, segment_len_s=seglen, tolerance_db=tol)
                    st.embedding_bands = band
                    nseg = len(st._segments(audio, WORK_SR, nfr))
                    wm = embed_watermark(audio, sample_rate=WORK_SR,
                                         watermark_bits=bits, model=st)
                    pat, conf = detect_watermark(wm, WORK_SR, detector)
                    acc, conf = bit_acc(bits, pat), float(conf)
                    det = int(conf >= DET_CONF)
                    status = "ok"
                except Exception as e:
                    nseg, acc, conf, det, status = 0, float("nan"), float("nan"), 0, \
                        f"err:{type(e).__name__}"
                    print(f"    [{mname:12s} {bn:9s}] FAILED: {e}")
                agg.setdefault((mname, bn), []).append((acc, conf, det))
                writer.writerow([ci, name, bn, mname, nseg, round(acc, 4),
                                 round(conf, 4), det, status])
                print(f"    {mname:12s} {bn:9s}  seg={nseg:2d}  "
                      f"acc={acc:.3f}  conf={conf:.3f}  {'DET' if det else 'miss'}")
    fout.close()

    # ---- printed summary: method x band mean conf (detected/clips) --------- #
    method_names = [m[0] for m in methods]
    band_names = [bn for bn, _ in bands]

    def cmean(m, b):
        rows = agg.get((m, b), [])
        cs = [r[1] for r in rows if not np.isnan(r[1])]
        return (np.mean(cs) if cs else float("nan"),
                sum(r[2] for r in rows), len(rows))

    def amean(m, b):
        rows = agg.get((m, b), [])
        a = [r[0] for r in rows if not np.isnan(r[0])]
        return np.mean(a) if a else float("nan")

    print("\n" + "=" * (16 + 15 * len(band_names)))
    print("CLEAN detection — mean conf (detected/clips)")
    print(f"  {'method':14s}" + "".join(f"{b:>15s}" for b in band_names))
    for m in method_names:
        line = f"  {m:14s}"
        for b in band_names:
            c, d, t = cmean(m, b)
            line += f"{c:.2f} {d}/{t}".rjust(15)
        print(line)

    # ---- visualizations ---------------------------------------------------- #
    if do_viz:
        print("\nrendering figures...")
        conf_mean = {(m, b): cmean(m, b)[0] for m in method_names for b in band_names}
        acc_mean = {(m, b): amean(m, b) for m in method_names for b in band_names}
        for name, audio in zip(names, audios):
            p1 = os.path.join(OUT_DIR, f"v11_salient_points_{name}.png")
            p2 = os.path.join(OUT_DIR, f"v11_hopmask_{name}.png")
            try:
                viz.fig_salient_points(audio, WORK_SR, name, anchors, p1,
                                       n_fft=n_fft, hop=hop, anchor_rate=rate)
                print(f"  wrote {p1}")
            except Exception as e:
                print(f"  [viz] salient_points {name} failed: {e}")
            try:
                viz.fig_hop_masks(embedder, audio, WORK_SR, name, methods, p2,
                                  band=MASK_BAND, key=key, n_bands=nbands,
                                  seglen=seglen, anchor_rate=rate)
                print(f"  wrote {p2}")
            except Exception as e:
                print(f"  [viz] hopmask {name} failed: {e}")
        p3 = os.path.join(OUT_DIR, "v11_summary_heatmap.png")
        try:
            viz.fig_summary_heatmap(conf_mean, acc_mean, method_names, band_names, p3)
            print(f"  wrote {p3}")
        except Exception as e:
            print(f"  [viz] heatmap failed: {e}")

    print(f"\nwrote {csv_path}")
    print("read: heatmap = does the salient method change clean detection? (expect all "
          "detect clean at N=3). hopmask = how each method reshapes the staircase. "
          "salient_points = where the anchors land and how much they agree.")


if __name__ == "__main__":
    main(sys.argv[1:])
