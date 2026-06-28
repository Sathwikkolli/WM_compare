import numpy as np, librosa
from scipy.signal import resample_poly
from aware.utils.models import load
from aware.service import detect_watermark
_, detector = load()
ref = np.array([0,1,1,0,1,1,1,1,1,1,1,0,0,1,0,0,0,0,0,1])
audio, sr = librosa.load("rerecord/aware_test3.wav", sr=16000, mono=True)
seg = audio[40*sr:90*sr]                      # the dead region
for ppm in [-3000,-2000,-1000,-500,-200,0,200,500,1000,2000,3000]:
    up, down = 1000000, 1000000 + ppm         # tiny time-scale correction
    s2 = resample_poly(seg, up, down)
    pat, conf = detect_watermark(s2, sr, detector)
    acc = (np.asarray(pat).astype(int) == ref).mean()
    print(f"ppm={ppm:+5d}  conf={conf:.3f}  bit_acc={acc:.2f}")
