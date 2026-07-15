"""
dereverb_wpe_test.py -- can dereverberation rescue the AWARE watermark, the way
Demucs rescued it from a music bed?

Why WPE
-------
WPE (Weighted Prediction Error, nara_wpe) is the established blind dereverberation
baseline: proposed 2008, the REVERB-challenge standard, integrated in ESPnet. It
models the LATE reverberation with long-term linear prediction and SUBTRACTS it --
it does not resynthesise the speech. That matters: Demucs rescued the watermark
precisely because it filters rather than regenerates. Neural dereverb (DeepFilterNet,
diffusion) scores better on speech quality but rebuilds the waveform, which is the
class of process known to destroy watermarks.

Why it may still fail
---------------------
Music is ADDITIVE (mix = speech + music) -- the watermark sat untouched underneath,
so removing the masker gave it back. Reverb is CONVOLUTIVE (rev = speech * rir) --
it deformed the watermark's own samples. Undoing it is an ill-posed inverse problem
(the room's spectral nulls destroy information outright). So this is a genuine test,
not a formality.

Conditions per clip (AWARE only, 5 Emilia clips)
------------------------------------------------
  baseline              watermarked speech            -> reference
  wpe_clean             WPE on clean watermarked      -> *** THE CONTROL ***
                        no reverb involved. If this tanks, WPE itself damages the
                        watermark and the approach is dead regardless of reverb.
  reverb_<rt60>         reverbed                      -> the damage we measured
  wpe_reverb_<rt60>     WPE on the reverbed           -> does dereverb rescue it?

Usage (repo root, wmcompare env; needs `pip install nara-wpe`):
    python dereverb_wpe_test.py
    python dereverb_wpe_test.py --n 5 --taps 10 --delay 3
"""
import os, sys, csv, numpy as np

BASE = os.environ.get('WM_COMPARE_BASE', os.path.expanduser('~/wm_compare'))
CASCADE = os.path.join(BASE, 'cascade')
EMILIA_CSV = "/nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_curated.csv"

RT60S = [0.3, 0.5, 0.8, 1.2, 2.0]     # spans AWARE's pass (0.3) -> fail (0.8+)
MIN_DUR = 9.0
THRESH = 0.5                           # AWARE detection probability
STFT_SIZE, STFT_SHIFT = 512, 128

sys.path.insert(0, CASCADE)
import cascade_lib as cl
SR = cl.SR_MASTER


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def pick_clips(csv_path, n):
    out = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                if float(row['duration_s']) >= MIN_DUR and os.path.exists(row['path']):
                    out.append(row['path'])
            except (KeyError, ValueError):
                continue
            if len(out) >= n:
                break
    return out


def make_rir(rt60, sr=SR, seed=0):
    """Dense room impulse response: decaying noise (thousands of reflections)."""
    rng = np.random.default_rng(seed)
    n = max(2, int(sr * rt60))
    decay = np.exp(-6.9078 * np.arange(n) / n)
    rir = rng.standard_normal(n).astype('float32') * decay
    rir[0] += 1.0
    rir /= np.sqrt((rir ** 2).sum())
    return rir.astype('float32')


def apply_reverb(y, rt60, seed=0):
    """Level-matched convolution, so we measure smearing not gain."""
    rir = make_rir(rt60, seed=seed)
    w = np.convolve(y, rir)[:len(y)].astype('float32')
    r_y, r_w = np.sqrt(np.mean(y ** 2)), np.sqrt(np.mean(w ** 2))
    if r_w > 0:
        w *= r_y / r_w
    return w.astype('float32')


def wpe_dereverb(y, taps=10, delay=3, iterations=5):
    """Single-channel WPE. Returns a dereverberated, level-matched signal."""
    from nara_wpe.wpe import wpe
    from nara_wpe.utils import stft, istft
    x = y[None, :]                                     # (D=1, N)
    X = stft(x, size=STFT_SIZE, shift=STFT_SHIFT).transpose(2, 0, 1)   # (F, D, T)
    Z = wpe(X, taps=taps, delay=delay, iterations=iterations, statistics_mode='full')
    z = istft(Z.transpose(1, 2, 0), size=STFT_SIZE, shift=STFT_SHIFT)  # (D, N)
    z = np.asarray(z[0], dtype='float32')
    n = min(len(z), len(y)); z = z[:n]
    r_y, r_z = np.sqrt(np.mean(y[:n] ** 2)), np.sqrt(np.mean(z ** 2))
    if r_z > 0:
        z *= r_y / r_z                                 # level-match back
    return z


def main(argv):
    try:
        import nara_wpe  # noqa: F401
    except ImportError:
        print('ERROR: nara_wpe not installed. Run:\n    pip install nara-wpe')
        return 1

    n_clips = get_arg(argv, '--n', 5, int)
    taps    = get_arg(argv, '--taps', 10, int)
    delay   = get_arg(argv, '--delay', 3, int)
    csv_in  = get_arg(argv, '--csv', EMILIA_CSV)

    clips = pick_clips(csv_in, n_clips)
    if not clips:
        print('no usable clips found in', csv_in); return 1
    print(f'model=aware | {len(clips)} Emilia clips | WPE taps={taps} delay={delay}\n')

    adapter = cl.get_adapter('aware')
    agg, rows = {}, []

    for i, path in enumerate(clips, 1):
        name = os.path.basename(path)
        y = cl.read_wav(path)
        print(f'[{i}/{len(clips)}] embedding AWARE into {name} ...')
        y_wm = np.asarray(adapter.embed(y), dtype='float32')

        def record(cond, sig):
            conf, _, bacc = adapter.detect(np.asarray(sig, dtype='float32'))
            agg.setdefault(cond, []).append(float(conf))
            rows.append({'clip': name, 'condition': cond, 'conf': round(float(conf), 4),
                         'bit_acc': round(float(bacc), 4),
                         'detected': 'DETECTED' if conf >= THRESH else 'no'})
            return float(conf)

        record('baseline', y_wm)
        # --- THE CONTROL: WPE on clean watermarked speech, no reverb ---
        c_ctrl = record('wpe_clean', wpe_dereverb(y_wm, taps, delay))
        print(f'      wpe_clean (control) = {c_ctrl:.3f}')

        for rt in RT60S:
            rev = apply_reverb(y_wm, rt, seed=i)
            c_rev = record(f'reverb_{rt}', rev)
            c_wpe = record(f'wpe_reverb_{rt}', wpe_dereverb(rev, taps, delay))
            print(f'      RT60 {rt:<4}: reverb={c_rev:.3f} -> after WPE={c_wpe:.3f} '
                  f'({"+" if c_wpe >= c_rev else ""}{c_wpe - c_rev:.3f})')

    # ---- summary -----------------------------------------------------------
    print(f'\n===== MEAN over {len(clips)} clips (threshold {THRESH}) =====')
    conds = ['baseline', 'wpe_clean'] + \
            [c for r in RT60S for c in (f'reverb_{r}', f'wpe_reverb_{r}')]
    summary = []
    print(f'{"condition":22s} {"mean":>8s} {"std":>7s} {"detected":>9s}')
    for cond in conds:
        v = np.array(agg.get(cond, []))
        if not len(v):
            continue
        det = 'DETECTED' if v.mean() >= THRESH else 'no'
        print(f'{cond:22s} {v.mean():8.4f} {v.std():7.4f} {det:>9s}')
        summary.append({'condition': cond, 'mean_conf': round(float(v.mean()), 4),
                        'std': round(float(v.std()), 4), 'detected': det})

    ctrl = float(np.mean(agg.get('wpe_clean', [0])))
    print(f'\nCONTROL wpe_clean = {ctrl:.3f} -> '
          + ('WPE is watermark-SAFE; any rescue below is real'
             if ctrl >= THRESH else
             'WPE DESTROYS the watermark by itself -- dereverb cannot rescue it'))
    print('\nrescue by RT60:')
    for rt in RT60S:
        a = float(np.mean(agg.get(f'reverb_{rt}', [0])))
        b = float(np.mean(agg.get(f'wpe_reverb_{rt}', [0])))
        verdict = 'RESCUED' if (b >= THRESH and a < THRESH) else \
                  ('improved' if b > a else 'no help')
        print(f'  RT60 {rt:<4}: {a:.3f} -> {b:.3f}   {verdict}')

    out = os.path.join(BASE, 'dereverb_wpe_results.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['clip', 'condition', 'conf', 'bit_acc', 'detected'])
        w.writeheader(); w.writerows(rows)
    out2 = os.path.join(BASE, 'dereverb_wpe_summary.csv')
    with open(out2, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['condition', 'mean_conf', 'std', 'detected'])
        w.writeheader(); w.writerows(summary)
    print('\nsaved:', out, '\n       ', out2)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        revs = [np.mean(agg[f'reverb_{r}']) for r in RT60S]
        wpes = [np.mean(agg[f'wpe_reverb_{r}']) for r in RT60S]
        plt.figure(figsize=(6.5, 4.5))
        plt.plot(RT60S, revs, 'o-', label='reverbed (damaged)')
        plt.plot(RT60S, wpes, 's-', label='after WPE dereverb')
        plt.axhline(ctrl, color='g', ls='--', label='WPE on clean (control)')
        plt.axhline(THRESH, color='r', ls=':', lw=1, label=f'threshold {THRESH}')
        plt.xlabel('RT60 (s)  -- bigger = more reverb'); plt.ylabel('detection conf')
        plt.ylim(0, 1.05); plt.title(f'AWARE: does WPE dereverb rescue the watermark? ({len(clips)} clips)')
        plt.legend(); plt.tight_layout()
        p = os.path.join(BASE, 'dereverb_wpe.png'); plt.savefig(p, dpi=130)
        print('plot:', p)
    except Exception as e:
        print('(plot skipped:', e, ')')


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
