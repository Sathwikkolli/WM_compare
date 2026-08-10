"""
ab_aware/embed.py -- build the two arms on disk, before any scoring happens.

Reads clips.json, writes ab_aware/work/:

    work/src/<clip_id>.wav     all 40 clips, 16 kHz mono float32 -- the PESQ/SNR
                               reference and the clean arm's audio
    work/wm/<clip_id>.wav      the 20 watermarked positives
    work/payloads.json         clip_id -> the 20 bits actually embedded

RUN THIS ON A LOGIN NODE BEFORE sbatch. Same reasoning as align_bench's
clips.json: 40 array tasks racing to embed the same files would corrupt them,
and a broken AWARE checkpoint would fail all 40 tasks instead of failing once,
interactively, in ten seconds.

PER-CLIP RANDOM PAYLOAD, not one shared payload. A shared payload leaks into the
statistics -- every clip would share the same bit pattern, so a detector biased
toward that pattern would post inflated bit accuracy across the whole arm and
nothing in the numbers would reveal it. Payloads are drawn from a seeded RNG, so
the run is still reproducible.

The clean arm is deliberately written through the SAME resample-and-write path
as the watermarked arm. If negatives went through a different code path, any
arm difference could be an artefact of file handling rather than the watermark.

Usage:
    python embed.py                    # after make_clips.py
    python embed.py --seed 0
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)
sys.path.insert(0, ROOT)

WORK_SR = 16000
N_BITS = 20                 # AWARE carries 20 bits -- see fsss/exp1_critical.py
AWARE_MODEL = os.environ.get("AWARE_MODEL", "AWARE")

CLIPS_JSON = os.path.join(HERE, "clips.json")
WORK = os.path.join(HERE, "work")
SRC_DIR = os.path.join(WORK, "src")
WM_DIR = os.path.join(WORK, "wm")
PAYLOADS = os.path.join(WORK, "payloads.json")


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def load_16k(path):
    import librosa
    y, _ = librosa.load(path, sr=WORK_SR, mono=True)
    return y.astype("float32")


def write(path, y):
    import soundfile as sf
    sf.write(path, np.asarray(y, dtype="float32"), WORK_SR, subtype="FLOAT")


def main(argv):
    seed = get_arg(argv, "--seed", 0, int)

    if not os.path.exists(CLIPS_JSON):
        raise SystemExit("clips.json missing -- run 'python make_clips.py' first.")
    spec = json.load(open(CLIPS_JSON))
    wm_clips, clean_clips = spec["wm_clips"], spec["clean_clips"]

    os.makedirs(SRC_DIR, exist_ok=True)
    os.makedirs(WM_DIR, exist_ok=True)

    from aware.utils.models import load
    from aware.service import embed_watermark, detect_watermark
    embedder, detector = load(name=AWARE_MODEL)
    print(f"loaded AWARE model {AWARE_MODEL!r}")

    rng = np.random.RandomState(seed)
    payloads, rows = {}, []

    print(f"\nembedding {len(wm_clips)} positives...")
    for i, path in enumerate(wm_clips):
        cid = f"wm{i:02d}"
        y = load_16k(path)
        write(os.path.join(SRC_DIR, cid + ".wav"), y)

        bits = rng.randint(0, 2, size=N_BITS).astype(int)
        z = np.asarray(embed_watermark(y, WORK_SR, bits, embedder), dtype="float32")
        write(os.path.join(WM_DIR, cid + ".wav"), z)
        payloads[cid] = bits.tolist()

        # Round-trip check on the undistorted file. If AWARE cannot read its own
        # watermark back out of clean audio, every downstream number is noise --
        # better to see it here than to explain a dead table later.
        res = detect_watermark(z, WORK_SR, detector)
        pat, conf = (res[0], res[1])
        pat = np.asarray(pat).astype(int).ravel()[:N_BITS]
        n = min(len(pat), N_BITS)
        acc = float(np.mean(pat[:n] == bits[:n])) if n else float("nan")
        rows.append((cid, float(conf), acc))
        print(f"  {cid}  {os.path.basename(path):<40s} conf={conf:.4f}  bit_acc={acc:.3f}")

    print(f"\nwriting {len(clean_clips)} negatives (no embedding)...")
    for i, path in enumerate(clean_clips):
        cid = f"cl{i:02d}"
        write(os.path.join(SRC_DIR, cid + ".wav"), load_16k(path))
        print(f"  {cid}  {os.path.basename(path)}")

    json.dump(
        {"seed": seed, "n_bits": N_BITS, "model": AWARE_MODEL, "payloads": payloads},
        open(PAYLOADS, "w"), indent=2,
    )

    confs = np.array([r[1] for r in rows])
    accs = np.array([r[2] for r in rows])
    print(f"\nclean-embed round trip over {len(rows)} positives:")
    print(f"  conf     min={confs.min():.4f}  median={np.median(confs):.4f}  max={confs.max():.4f}")
    print(f"  bit_acc  min={accs.min():.3f}  median={np.median(accs):.3f}  max={accs.max():.3f}")
    weak = [r[0] for r in rows if r[1] < 0.5]
    if weak:
        print(f"  WARNING: {len(weak)} positive(s) undetected at conf>=0.5 BEFORE any "
              f"attack: {weak}\n  Investigate before spending cluster time.")
    print(f"\nwrote {WORK}/")


if __name__ == "__main__":
    main(sys.argv[1:])
