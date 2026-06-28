import sys, wave, numpy as np
p = sys.argv[1]
w = wave.open(p,'rb'); sr=w.getframerate()
x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)/32768.0
w.close()
win = 10*sr
for s in range(0, len(x)-1, win):
    seg = x[s:s+win]
    if len(seg) < sr: break
    peak=np.abs(seg).max(); rms=np.sqrt((seg**2).mean())
    clip=np.mean(np.abs(seg)>0.99)*100
    # crest factor: low = compressed/clipped or noisy; high = clean dynamics
    crest = peak/(rms+1e-9)
    print(f"{s/sr:5.0f}s  peak={peak:.3f} rms={rms:.4f} clip={clip:.3f}% crest={crest:5.2f}")
