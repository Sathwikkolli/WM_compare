import sys, numpy as np, librosa
from aware.utils.models import load
from aware.service import detect_watermark
_, detector = load()
ref = np.array([0,1,1,0,1,1,1,1,1,1,1,0,0,1,0,0,0,0,0,1])
audio, sr = librosa.load(sys.argv[1], sr=16000, mono=True)
win, hop = 20*sr, 10*sr
for s in range(0, max(1, len(audio)-win+1), hop):
    seg = audio[s:s+win]
    pat, conf = detect_watermark(seg, sr, detector)
    acc = (np.asarray(pat).astype(int) == ref).mean()
    print(f"{s/sr:6.1f}s   conf={conf:.3f}   bit_acc={acc:.2f}")
