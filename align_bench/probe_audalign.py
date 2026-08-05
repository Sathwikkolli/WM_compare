"""
align_bench/probe_audalign.py -- print what audalign ACTUALLY returns.

audalign aligns fine (the logs say "2 out of 2 found and aligned") but
methods._dig_audalign cannot locate the offset in the dict, so every audalign
row comes back UNUSABLE. Rather than guess at the shape across versions, dump
it once and patch the digger against reality.

Builds the same synthetic pair calibrate() uses: 8 s reference, 1.0 s cropped
off the head, so the true answer is known -- the number we are hunting for in
the dict is 1.0 (seconds) or -1.0, depending on their sign convention.

Usage:
    python probe_audalign.py
    python probe_audalign.py --recognizer fingerprint
"""
import os
import pprint
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SR = 16000
DUR_S = 8.0
SHIFT_S = 1.0

CTORS = {
    "fingerprint": "FingerprintRecognizer",
    "correlation": "CorrelationRecognizer",
    "spectrogram": "CorrelationSpectrogramRecognizer",
}


def synth(sr=SR, dur_s=DUR_S, seed=0):
    rng = np.random.RandomState(seed)
    n = int(dur_s * sr)
    y = rng.randn(n).astype("float32") * 0.1
    t = np.arange(n) / sr
    for f in (220.0, 440.0, 1300.0, 2700.0):
        y += 0.15 * np.sin(2 * np.pi * f * t).astype("float32")
    y *= np.linspace(0.4, 1.0, n).astype("float32")
    return y


def walk(obj, path="result", depth=0, hits=None):
    """Recursively hunt for anything close to +/-1.0 -- the known true offset."""
    if hits is None:
        hits = []
    if depth > 6:
        return hits
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk(v, f"{path}[{k!r}]", depth + 1, hits)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj[:20]):
            walk(v, f"{path}[{i}]", depth + 1, hits)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if abs(abs(float(obj)) - SHIFT_S) < 0.05:
            hits.append((path, float(obj)))
    return hits


def main(argv):
    try:
        import audalign as ad
        import soundfile as sf
    except Exception as e:
        raise SystemExit(f"audalign/soundfile not importable: {e}")

    print(f"audalign version: {getattr(ad, '__version__', 'unknown')}")
    print(f"top-level callables: "
          f"{[n for n in dir(ad) if not n.startswith('_') and n[0].isalpha()][:40]}\n")

    which = list(CTORS)
    if "--recognizer" in argv:
        which = [argv[argv.index("--recognizer") + 1]]

    ref = synth()
    dist = ref[int(SHIFT_S * SR):].copy()

    d = tempfile.mkdtemp(prefix="probe_")
    fr, fd = os.path.join(d, "ref.wav"), os.path.join(d, "dist.wav")
    sf.write(fr, ref, SR)
    sf.write(fd, dist, SR)
    print(f"ref  = {DUR_S}s, dist = ref with {SHIFT_S}s cropped off the HEAD")
    print(f"so the number we want in the dict is {SHIFT_S} or {-SHIFT_S}\n")

    for name in which:
        ctor = CTORS[name]
        if not hasattr(ad, ctor):
            print(f"=== {ctor}: NOT PRESENT in this audalign ===\n")
            continue
        print("=" * 78)
        print(f"=== {ctor} ===")
        print("=" * 78)
        try:
            rec = getattr(ad, ctor)()
            res = ad.align_files(fr, fd, recognizer=rec)
        except Exception as e:
            print(f"align_files raised: {type(e).__name__}: {e}\n")
            continue

        print(f"\ntype: {type(res)}")
        if isinstance(res, dict):
            print(f"top-level keys: {list(res.keys())}")
        print("\n--- full structure ---")
        pprint.pprint(res, width=110, depth=5, compact=True)

        hits = walk(res)
        print(f"\n--- values near +/-{SHIFT_S} (the true offset) ---")
        if hits:
            for p, v in hits:
                print(f"  {p} = {v}")
        else:
            print("  NONE FOUND -- offset may be in samples, ms, or frames.")
            print("  Scan the dump above for a number near "
                  f"{int(SHIFT_S * SR)} (samples) or {int(SHIFT_S * 1000)} (ms).")
        print()

    for p in (fr, fd):
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.rmdir(d)
    except OSError:
        pass


if __name__ == "__main__":
    main(sys.argv[1:])
