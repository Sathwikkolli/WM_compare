"""
fsss/exp_v1_staircase.py -- v1 milestone: does key-driven subband hopping still
embed a readable watermark at good quality?

Embeds the SAME 20-bit payload three ways into one clip, then detects all three
with the STOCK AWARE detector (unchanged), reporting clean bit-accuracy,
confidence, and PESQ:

  stock    -- unmodified AWAREEmbedder (baseline)
  bands1   -- StaircaseAWAREEmbedder with n_bands=1  => full (500,4000) stripe on
              every frame == stock. EQUIVALENCE CHECK (should match `stock`).
  bands4   -- StaircaseAWAREEmbedder with n_bands=4  => key-chosen 875 Hz sub-band
              hopping per anchor segment. THE ACTUAL TEST.

The make-or-break question: does bands4 keep bit-accuracy high (detector still
reads the payload) at a PESQ close to stock? If yes, the staircase design holds
and we move to per-band robustness (Exp D). If bit-accuracy collapses, each frame
has too little writable room -> raise tolerance_db or use fewer/wider bands.

Run on a GPU node in wmcompare:
    conda activate wmcompare
    python -m fsss.exp_v1_staircase                    # first Emilia clip
    python -m fsss.exp_v1_staircase --clip audio/client_original_16k.wav
    python -m fsss.exp_v1_staircase --key thesis --bands 4 --seed 0
"""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(BASE, "cascade"))

from aware.utils.models import load
from aware.service import embed_watermark, detect_watermark
from aware.metrics.audio import BER, PESQ

from fsss.staircase import StaircaseAWAREEmbedder
from fsss.exp_a_repeatability import load_16k, pick_clips, EMILIA_CSV, WORK_SR

WM_BITS = 20


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def bit_accuracy(bits, pattern):
    p = np.asarray(pattern).astype(int).ravel()
    b = np.asarray(bits).astype(int).ravel()
    n = min(len(b), len(p))
    return float(np.mean(b[:n] == p[:n])) if n else float("nan")


def main(argv):
    clip = get_arg(argv, "--clip", None)
    key = get_arg(argv, "--key", "thesis")
    n_bands = get_arg(argv, "--bands", 4, int)
    seed = get_arg(argv, "--seed", 0, int)

    if clip is None:
        clips = pick_clips(EMILIA_CSV, 1)
        if not clips:
            print("no clip found; pass --clip PATH")
            return
        clip = clips[0]

    audio = load_16k(clip)
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=WM_BITS, dtype=np.int32)
    print(f"clip: {clip}")
    print(f"payload ({WM_BITS} bits): {bits.tolist()}")

    embedder, detector = load()
    ber_metric, pesq_metric = BER(), PESQ()

    configs = [
        ("stock", embedder),
        ("bands1", StaircaseAWAREEmbedder.from_embedder(embedder, key=key, n_bands=1)),
        (f"bands{n_bands}", StaircaseAWAREEmbedder.from_embedder(embedder, key=key, n_bands=n_bands)),
    ]

    rows = []
    for name, model in configs:
        print(f"\n=== embedding: {name} ===")
        wm = embed_watermark(audio, sample_rate=WORK_SR, watermark_bits=bits, model=model)
        pattern, conf = detect_watermark(wm, WORK_SR, detector)
        acc = bit_accuracy(bits, pattern)
        try:
            pesq = pesq_metric(wm, audio, WORK_SR)
        except Exception as e:
            pesq = float("nan")
            print(f"  PESQ failed: {e}")
        rows.append((name, acc, float(conf), pesq))
        print(f"  bit_acc={acc:.3f}  conf={conf:.3f}  PESQ={pesq:.3f}")

    print("\n" + "=" * 52)
    print(f"{'config':10s} {'bit_acc':>8s} {'conf':>8s} {'PESQ':>8s}")
    for name, acc, conf, pesq in rows:
        print(f"{name:10s} {acc:8.3f} {conf:8.3f} {pesq:8.3f}")
    print("=" * 52)
    print("read: bands1 should ~= stock (equivalence); bands4 is the real test "
          "(want high bit_acc + PESQ near stock).")


if __name__ == "__main__":
    main(sys.argv[1:])
