"""
real_attacks_experiment.py -- reproduce the client's Stage-06 failure with REAL,
independent audio, and measure AWARE's breaking points.

Real inputs (downloaded on first run):
  speech : Poe "The Black Cat", LibriVox (public domain) -> trimmed to 25 s
  music  : "March For Honor", CC0 Instruments (public domain) -> real stereo track

Pipeline:
  1. Embed AWARE into the real speech (known message -> real bit-accuracy).
  2. Baseline detect (should be strong).
  3. TEST 1  music-bed SNR sweep  -> SNR where conf crosses 0.50 (AWARE threshold).
  4. TEST 2  real ffmpeg stereo wideners on the watermarked speech -> detect after
     the detector's mono downmix (isolates phase-cancellation).

Usage (repo root, wmcompare env):
    python real_attacks_experiment.py
"""
import os, sys, csv, math, subprocess, numpy as np

BASE = os.environ.get('WM_COMPARE_BASE', os.path.expanduser('~/wm_compare'))
CASCADE = os.path.join(BASE, 'cascade')
DATA = os.path.join(BASE, 'real_audio'); os.makedirs(DATA, exist_ok=True)

SPEECH_URL = "https://archive.org/download/stories_001_librivox/black_cat_poe_ty.mp3"
MUSIC_URL  = "https://archive.org/download/MarchForHonor/March_For_Honor.mp3"
SPEECH_RAW = os.path.join(DATA, 'speech_raw.mp3')
MUSIC_RAW  = os.path.join(DATA, 'music_stereo.mp3')
SPEECH_WAV = os.path.join(DATA, 'speech_25s_16k.wav')   # trimmed clean speech
WM_WAV     = os.path.join(DATA, 'speech_wm.wav')        # AWARE-watermarked

SNRS = [30, 25, 20, 15, 10, 6, 5, 4, 3.8, 3, 2, 1, 0, -5, -10]  # fine points near the threshold

sys.path.insert(0, CASCADE)
import cascade_lib as cl
import user_key as uk
SR = cl.SR_MASTER


def sh(*a):
    subprocess.run(list(a), check=True)


def ensure_inputs():
    if not os.path.exists(SPEECH_RAW):
        print('downloading speech (LibriVox)...'); sh('curl', '-L', '-s', '-o', SPEECH_RAW, SPEECH_URL)
    if not os.path.exists(MUSIC_RAW):
        print('downloading music (CC0)...'); sh('curl', '-L', '-s', '-o', MUSIC_RAW, MUSIC_URL)
    if not os.path.exists(SPEECH_WAV):
        # skip 20 s intro, take 25 s of speech, mono 16 kHz
        sh('ffmpeg', '-y', '-loglevel', 'error', '-ss', '20', '-t', '25',
           '-i', SPEECH_RAW, '-ac', '1', '-ar', '16000', SPEECH_WAV)


def widen(cfg_name, af):
    """Apply a real ffmpeg stereo widener to the watermarked speech, return path."""
    out = os.path.join(DATA, f'wm_{cfg_name}.wav')
    sh('ffmpeg', '-y', '-loglevel', 'error', '-i', WM_WAV, '-af', af, out)
    return out


def main():
    ensure_inputs()
    adapter = cl.get_adapter('aware')

    print('embedding AWARE into real speech...')
    y = cl.read_wav(SPEECH_WAV)
    y_wm = np.asarray(adapter.embed(y), dtype='float32')
    cl.write_wav(WM_WAV, y_wm, sr=SR)
    c0, _, b0 = adapter.detect(y_wm)
    print(f'BASELINE  conf={c0:.4f}  bit_acc={b0:.4f}\n')

    rows = []

    # ---- TEST 1: music-bed SNR sweep --------------------------------------
    print('=== TEST 1: real music-bed SNR sweep ===')
    m = cl.read_wav(MUSIC_RAW)                     # detector downmixes anyway -> mono music
    if len(m) < len(y_wm):
        m = np.tile(m, int(np.ceil(len(y_wm) / len(m))))
    m = m[:len(y_wm)].astype('float32')
    Ps, Pm = float(np.mean(y_wm ** 2)), float(np.mean(m ** 2))
    print(f'{"SNR_dB":>7s} {"conf":>7s} {"bit_acc":>8s} {"detected":>9s}')
    thr, prev = None, None
    for snr in SNRS:
        a = math.sqrt(Ps / (Pm * (10 ** (snr / 10.0))))
        mix = (y_wm + a * m).astype('float32')
        cl.write_wav(os.path.join(DATA, f'mix_{snr:+05.1f}dB.wav'), mix, sr=SR)  # save for A/B listening
        conf, bits, bacc = adapter.detect(mix)
        det = 'DETECTED' if conf >= 0.5 else 'no'
        print(f'{snr:7.1f} {conf:7.4f} {bacc:8.4f} {det:>9s}')
        rows.append({'test': 'music_snr', 'config': f'{snr}dB', 'conf': round(float(conf), 4),
                     'bit_acc': round(float(bacc), 4), 'detected': det})
        if prev and thr is None and prev[1] >= 0.5 > conf:
            s1, c1 = prev; s2, c2 = snr, float(conf)
            thr = s2 + (0.5 - c2) * (s1 - s2) / (c1 - c2)
        prev = (snr, float(conf))
    print('==> music-bed threshold (conf=0.50):',
          f'{thr:.1f} dB SNR' if thr is not None else 'not crossed in range', '\n')

    # ---- TEST 2: real stereo wideners -------------------------------------
    print('=== TEST 2: real ffmpeg stereo wideners (detector downmixes to mono) ===')
    configs = {
        'dualmono':   'pan=stereo|c0=c0|c1=c0',                                  # control: no width
        'haas_12ms':  'pan=stereo|c0=c0|c1=c0,adelay=0|12',                      # Haas widening
        'stereowiden':'pan=stereo|c0=c0|c1=c0,stereowiden=delay=20:feedback=0.3:crossfeed=0.3:drymix=0.8',
    }
    print(f'{"widener":14s} {"conf":>7s} {"bit_acc":>8s} {"detected":>9s}')
    for name, af in configs.items():
        try:
            wav = widen(name, af)
            yv = cl.read_wav(wav)                  # cl.read_wav downmixes stereo -> mono
            conf, bits, bacc = adapter.detect(yv.astype('float32'))
            det = 'DETECTED' if conf >= 0.5 else 'no'
            print(f'{name:14s} {conf:7.4f} {bacc:8.4f} {det:>9s}')
            rows.append({'test': 'stereo', 'config': name, 'conf': round(float(conf), 4),
                         'bit_acc': round(float(bacc), 4), 'detected': det})
        except Exception as e:
            print(f'{name:14s} ERROR: {e}')

    out = os.path.join(BASE, 'real_attacks_results.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['test', 'config', 'conf', 'bit_acc', 'detected'])
        w.writeheader(); w.writerows(rows)
    print(f'\nbaseline conf={c0:.4f}   threshold=0.50')
    print('saved:', out)

    # optional plot of the SNR curve
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        mus = [r for r in rows if r['test'] == 'music_snr']
        xs = [float(r['config'][:-2]) for r in mus]
        plt.figure(figsize=(6, 4))
        plt.plot(xs, [r['conf'] for r in mus], 'o-', label='detection conf')
        plt.plot(xs, [r['bit_acc'] for r in mus], 's--', label='bit accuracy')
        plt.axhline(0.5, color='r', ls=':', lw=1, label='threshold 0.5')
        if thr is not None:
            plt.axvline(thr, color='g', ls=':', lw=1, label=f'{thr:.1f} dB')
        plt.gca().invert_xaxis()
        plt.xlabel('speech-to-music SNR (dB)'); plt.ylabel('score')
        plt.title('AWARE vs real music bed'); plt.legend(); plt.tight_layout()
        p = os.path.join(BASE, 'real_music_snr.png'); plt.savefig(p, dpi=130)
        print('plot:', p)
    except Exception as e:
        print('(plot skipped:', e, ')')


if __name__ == '__main__':
    main()
