import sys, wave, numpy as np, torch
from audioseal import AudioSeal

def load_wav(path):
    w = wave.open(path, 'rb'); sr = w.getframerate(); ch = w.getnchannels()
    raw = w.readframes(w.getnframes()); w.close()
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1: x = x.reshape(-1, ch).mean(1)
    return torch.from_numpy(x).view(1, 1, -1), sr

det = AudioSeal.load_detector("audioseal_detector_16bits")

def decode(path):
    wav, sr = load_wav(path)
    prob, msg = det.detect_watermark(wav, sr)
    return float(prob), msg.squeeze().int().tolist()

ref, tests = sys.argv[1], sys.argv[2:]
rp, rb = decode(ref)
print(f"REF  {ref}\n     conf={rp:.3f}  bits={rb}\n")
for t in tests:
    p, b = decode(t)
    acc = np.mean([x == y for x, y in zip(rb, b)]) if len(b) == len(rb) else float('nan')
    flag = "DETECTED" if p >= 0.5 else "NOT detected"
    print(f"{t}\n     conf={p:.3f} ({flag})  bit_acc_vs_ref={acc:.3f}\n     bits={b}")
