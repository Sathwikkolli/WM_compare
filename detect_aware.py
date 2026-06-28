import sys, numpy as np, librosa
from aware.utils.models import load
from aware.service import detect_watermark

embedder, detector = load()

def decode(path):
    audio, sr = librosa.load(path, sr=None, mono=True)
    if sr != 16000:
        from scipy.signal import resample_poly
        audio = resample_poly(audio, 16000, sr); sr = 16000
    pattern, conf = detect_watermark(audio, sr, detector)
    return float(conf), np.asarray(pattern).astype(int).tolist()

ref, tests = sys.argv[1], sys.argv[2:]
rc, rb = decode(ref)
print(f"REF  {ref}\n     conf={rc:.4f}  bits={rb}\n")
for t in tests:
    c, b = decode(t)
    n = min(len(rb), len(b))
    acc = np.mean([rb[i] == b[i] for i in range(n)]) if n else float('nan')
    flag = "DETECTED" if c >= 0.5 else "NOT detected"
    print(f"{t}\n     conf={c:.4f} ({flag})  bit_acc_vs_ref={acc:.3f}\n     bits={b}")
