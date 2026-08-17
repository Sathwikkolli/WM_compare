"""
damage_ab/embed_pairs.py -- build the matched arms on disk.

    work/src/<cid>.wav     the unwatermarked clip   (PESQ reference for both arms)
    work/wm/<cid>.wav      the same clip, AWARE-embedded
    work/payloads.json     cid -> the 20 bits embedded

RUN ON A LOGIN NODE before sbatch. Same reasoning as ab_aware/embed.py: a broken
AWARE checkpoint should fail once, interactively, in ten seconds -- not inside a
batch job after the queue wait.

BOTH ARMS GO THROUGH THE SAME WRITE PATH. The src file is not the original
Emilia file; it is the resampled-to-16k, float32 WAV. If the two arms were
written differently, any arm difference downstream could be file handling rather
than the watermark. This is the same guarantee ab_aware/embed.py makes.

Prints the clean round-trip check and warns if any clip is undetected BEFORE any
attack. If that fires, stop -- a clip that AWARE cannot read back from clean
audio contributes nothing but noise to a robustness sweep.

Usage:
    python embed_pairs.py              # after make_pairs.py
    python embed_pairs.py --seed 0
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

WORK_SR = 16000
N_BITS = 20                    # AWARE's payload -- see fsss/exp1_critical.py
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
        raise SystemExit("clips.json missing -- run 'python make_pairs.py' first.")
    clips = json.load(open(CLIPS_JSON))["clips"]

    os.makedirs(SRC_DIR, exist_ok=True)
    os.makedirs(WM_DIR, exist_ok=True)

    from aware.utils.models import load
    from aware.service import embed_watermark, detect_watermark
    embedder, detector = load(name=AWARE_MODEL)
    print(f"loaded AWARE model {AWARE_MODEL!r}")

    # Per-clip random payloads, same reasoning as ab_aware/embed.py: one shared
    # bit pattern across clips would let a detector biased toward that pattern
    # post inflated bit accuracy invisibly.
    rng = np.random.RandomState(seed)
    payloads, rows = {}, []

    print(f"\nbuilding {len(clips)} matched pairs...")
    for i, path in enumerate(clips):
        cid = f"cl{i:02d}"
        y = load_16k(path)
        write(os.path.join(SRC_DIR, cid + ".wav"), y)

        bits = rng.randint(0, 2, size=N_BITS).astype(int)
        z = np.asarray(embed_watermark(y, WORK_SR, bits, embedder), dtype="float32")
        write(os.path.join(WM_DIR, cid + ".wav"), z)
        payloads[cid] = bits.tolist()

        res = detect_watermark(z, WORK_SR, detector)
        pat, conf = res[0], res[1]
        pat = np.asarray(pat).astype(int).ravel()[:N_BITS]
        m = min(len(pat), N_BITS)
        acc = float(np.mean(pat[:m] == bits[:m])) if m else float("nan")
        rows.append((cid, float(conf), acc))
        print(f"  {cid}  {os.path.basename(path):<40s} "
              f"conf={conf:.4f}  bit_acc={acc:.3f}  {len(y)/WORK_SR:.1f}s")

    json.dump({"seed": seed, "n_bits": N_BITS, "model": AWARE_MODEL,
               "payloads": payloads}, open(PAYLOADS, "w"), indent=2)

    confs = np.array([r[1] for r in rows])
    accs = np.array([r[2] for r in rows])
    print(f"\nclean round trip over {len(rows)} clips:")
    print(f"  conf     min={confs.min():.4f}  median={np.median(confs):.4f}  max={confs.max():.4f}")
    print(f"  bit_acc  min={accs.min():.3f}  median={np.median(accs):.3f}  max={accs.max():.3f}")
    weak = [r[0] for r in rows if r[1] < 0.5]
    if weak:
        print(f"\n  WARNING: {len(weak)} clip(s) undetected at conf>=0.5 BEFORE any "
              f"attack: {weak}\n  Every sweep row from these clips would be noise. "
              f"Investigate before sbatch.")
    print(f"\nwrote {WORK}/")


if __name__ == "__main__":
    main(sys.argv[1:])
