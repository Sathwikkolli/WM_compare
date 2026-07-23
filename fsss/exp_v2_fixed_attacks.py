"""
fsss/exp_v2_fixed_attacks.py -- key-driven subband hopping WITHOUT librosa, under
the everyday-edit (METAPXYL-proxy) attack chain.

Step 1 of the ablation: hop bands by the secret key over FIXED equal time chunks
(no anchors, no content analysis). Embed the 20-bit payload, then attack the
watermarked audio with programmatic proxies of the real Pro-Tools mastering chain
and see how many bits survive. (The true METAPXYL stages are manual DAW renders we
can't re-apply to fresh audio, so we use the vox_attacks equivalents.)

  dynamic_compression  ~ RCompressor + L1 limiter
  echo                 ~ D-Verb reverb
  mp3                  ~ 256k MP3 export
  quantization         ~ 16-bit dither
  lowpass, gaussian_noise ~ extra everyday edits

N=4 came out too faint at the default budget, so we embed at a louder budget
(tolerance_db=12) and both N=2 (wider slices) and N=4. Detection uses the STOCK
AWARE detector (unchanged). Output: config x attack -> bit accuracy + confidence.

Run on a GPU node in wmcompare:
    conda activate wmcompare
    python -m fsss.exp_v2_fixed_attacks
    python -m fsss.exp_v2_fixed_attacks --clip audio/client_original_16k.wav --key thesis
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
from aware.metrics.audio import PESQ

import vox_attacks
from fsss.staircase import StaircaseAWAREEmbedder
from fsss.exp_a_repeatability import load_16k, pick_clips, EMILIA_CSV, WORK_SR

WM_BITS = 20
OUT_DIR = os.path.join(BASE, "fsss_out")

# METAPXYL-proxy everyday-edit chain (programmatic equivalents in vox_attacks)
ATTACKS = ["dynamic_compression", "echo", "mp3", "quantization", "lowpass", "gaussian_noise"]

# key-driven, fixed-segment (no librosa) configs to embed.
# tolerance_db is an IMPERCEPTIBILITY dial: LOWER = louder/stronger watermark
# (delta = coeff * 10**(-tolerance_db/20)). Stock uses 6; the staircase writes
# fewer bins so it needs more per-bin strength -> lower (even negative) tolerance.
CONFIGS = [
    ("N2_t3",  dict(n_bands=2, tolerance_db=3)),
    ("N2_t0",  dict(n_bands=2, tolerance_db=0)),
    ("N2_t-6", dict(n_bands=2, tolerance_db=-6)),
    ("N4_t0",  dict(n_bands=4, tolerance_db=0)),
]


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def bit_acc(bits, pattern):
    p = np.asarray(pattern).astype(int).ravel()
    b = np.asarray(bits).astype(int).ravel()
    n = min(len(b), len(p))
    return float(np.mean(b[:n] == p[:n])) if n else float("nan")


def main(argv):
    clip = get_arg(argv, "--clip", None)
    key = get_arg(argv, "--key", "thesis")
    seed = get_arg(argv, "--seed", 0, int)
    clean_only = "--clean" in argv           # skip attacks; just find the detectable strength
    if clip is None:
        clips = pick_clips(EMILIA_CSV, 1)
        if not clips:
            print("no clip found; pass --clip PATH")
            return
        clip = clips[0]

    audio = load_16k(clip)
    bits = np.random.default_rng(seed).integers(0, 2, size=WM_BITS, dtype=np.int32)
    print(f"clip: {clip}\npayload: {bits.tolist()}")

    embedder, detector = load()
    pesq_metric = PESQ()
    os.makedirs(OUT_DIR, exist_ok=True)
    rows_path = os.path.join(OUT_DIR, "exp_v2_rows.csv")
    fout = open(rows_path, "w", newline="")
    writer = csv.writer(fout)
    writer.writerow(["config", "attack", "param", "bit_acc", "conf", "detected"])

    for cname, cfg in CONFIGS:
        print(f"\n########## {cname} ##########")
        st = StaircaseAWAREEmbedder.from_embedder(
            embedder, key=key, segment_mode="fixed", **cfg)
        wm = embed_watermark(audio, sample_rate=WORK_SR, watermark_bits=bits, model=st)

        # clean baseline
        pat, conf = detect_watermark(wm, WORK_SR, detector)
        acc = bit_acc(bits, pat)
        try:
            pesq = pesq_metric(wm, audio, WORK_SR)
        except Exception:
            pesq = float("nan")
        print(f"  clean            bit_acc={acc:.3f} conf={conf:.3f} PESQ={pesq:.3f}")
        writer.writerow([cname, "clean", "", round(acc, 4), round(float(conf), 4),
                         int(conf >= 0.5)])

        if clean_only:                        # fast strength-finding: no attacks
            continue

        # attack sweep
        for attack in ATTACKS:
            for label, param in vox_attacks.VOX_GRID.get(attack, []):
                try:
                    wa = vox_attacks.apply(attack, param, wm.astype("float32"), WORK_SR)
                except Exception as e:
                    wa = None
                    print(f"  {attack}/{label}: ERROR {e}")
                if wa is None:
                    continue
                pat, conf = detect_watermark(wa, WORK_SR, detector)
                acc = bit_acc(bits, pat)
                det = int(conf >= 0.5)
                print(f"  {attack:20s}[{label:8s}] bit_acc={acc:.3f} conf={conf:.3f} "
                      f"{'DET' if det else 'miss'}")
                writer.writerow([cname, attack, label, round(acc, 4),
                                 round(float(conf), 4), det])

    fout.close()
    print(f"\nwrote {rows_path}")
    print("read: want clean conf>=0.5 first (detectable), then high bit_acc under attacks.")


if __name__ == "__main__":
    main(sys.argv[1:])
