"""
reverb_test_emilia.py -- does REAL (dense) reverb break the AWARE watermark?

Motivation
----------
Every benchmark we have says AWARE survives "echo" (bit_acc 1.0, conf ~0.99).
But `cascade/vox_attacks.py:a_echo` builds an impulse response that is all zeros
except TWO points -- the direct sound and ONE delayed copy. That is a slapback
echo, not reverb. Real reverb (the client's D-Verb, Stage 05) is a DENSE impulse
response: thousands of reflections decaying over time, which smears the signal
continuously instead of duplicating it once.

This script tests both on the same audio, over N real Emilia speech clips:
  * bench_echo_1tap : the benchmark's single-reflection echo   (expect: survives)
  * reverb_rt60_*   : dense RIR (decaying-noise), swept by RT60 (the real thing)

RT60 = time for the reverb tail to decay by 60 dB (room size / liveness).
  0.1-0.3 s small room | 0.5-0.8 s live room | 1.2-2.0 s hall / cathedral

Usage (repo root, wmcompare env):
    python reverb_test_emilia.py
    python reverb_test_emilia.py --n 7 --wet 1.0
"""
import os, sys, csv, numpy as np

BASE = os.environ.get('WM_COMPARE_BASE', os.path.expanduser('~/wm_compare'))
CASCADE = os.path.join(BASE, 'cascade')
EMILIA_CSV = "/nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_curated.csv"

RT60S = [0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0]   # seconds
MIN_DUR = 9.0                                  # skip very short clips

sys.path.insert(0, CASCADE)
import cascade_lib as cl
import user_key as uk
SR = cl.SR_MASTER


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def pick_clips(csv_path, n):
    """Take the first n Emilia clips at least MIN_DUR long."""
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
    """Dense room impulse response: exponentially-decaying noise (thousands of
    reflections) + the direct sound. This is what real reverb looks like --
    unlike the benchmark's 2-point (direct + one echo) impulse response."""
    rng = np.random.default_rng(seed)
    n = max(2, int(sr * rt60))
    decay = np.exp(-6.9078 * np.arange(n) / n)          # -60 dB across rt60
    rir = (rng.standard_normal(n).astype('float32') * decay)
    rir[0] += 1.0                                       # direct path
    rir /= np.sqrt((rir ** 2).sum())                    # unit energy
    return rir.astype('float32')


def apply_reverb(y, rt60, wet=1.0, seed=0):
    """Convolve with a dense RIR, level-matched so we measure SMEARING, not gain."""
    rir = make_rir(rt60, seed=seed)
    w = np.convolve(y, rir)[:len(y)].astype('float32')
    rms_y, rms_w = np.sqrt(np.mean(y ** 2)), np.sqrt(np.mean(w ** 2))
    if rms_w > 0:
        w *= rms_y / rms_w
    return ((1.0 - wet) * y + wet * w).astype('float32')


def bench_echo(y, duration=0.5, volume=0.4):
    """The benchmark's echo, copied from cascade/vox_attacks.py:a_echo -- a single
    reflection. Included so we can compare it against real reverb directly."""
    n = max(1, int(SR * duration))
    ir = np.zeros(n, dtype='float32'); ir[0] = 1.0; ir[-1] = volume
    return np.convolve(y, ir)[:len(y)].astype('float32')


def main(argv):
    n_clips = get_arg(argv, '--n', 7, int)
    wet     = get_arg(argv, '--wet', 1.0, float)
    csv_in  = get_arg(argv, '--csv', EMILIA_CSV)

    clips = pick_clips(csv_in, n_clips)
    if not clips:
        print('no usable clips found in', csv_in); return 1
    print(f'using {len(clips)} Emilia clips (>= {MIN_DUR}s), wet={wet}\n')

    adapter = cl.get_adapter('aware')
    per_config = {}     # config -> list of confs across clips
    rows = []

    for i, path in enumerate(clips, 1):
        name = os.path.basename(path)
        y = cl.read_wav(path)
        print(f'[{i}/{len(clips)}] embedding AWARE into {name} ...')
        y_wm = np.asarray(adapter.embed(y), dtype='float32')

        tests = {'baseline': y_wm, 'bench_echo_1tap': bench_echo(y_wm)}
        for rt in RT60S:
            tests[f'reverb_rt60_{rt}'] = apply_reverb(y_wm, rt, wet=wet, seed=i)

        for cfg, sig in tests.items():
            conf, bits, bacc = adapter.detect(sig)
            per_config.setdefault(cfg, []).append(float(conf))
            rows.append({'clip': name, 'config': cfg, 'conf': round(float(conf), 4),
                         'bit_acc': round(float(bacc), 4),
                         'detected': 'DETECTED' if conf >= 0.5 else 'no'})
        print(f'      baseline={per_config["baseline"][-1]:.3f}  '
              f'echo1tap={per_config["bench_echo_1tap"][-1]:.3f}  '
              f'rt60_2.0={per_config["reverb_rt60_2.0"][-1]:.3f}')

    # ---- aggregate across clips -------------------------------------------
    print(f'\n=== MEAN over {len(clips)} clips ===')
    hdr = f'{"config":20s} {"mean_conf":>10s} {"std":>7s} {"detected":>9s}'
    print(hdr); print('-' * len(hdr))
    order = ['baseline', 'bench_echo_1tap'] + [f'reverb_rt60_{r}' for r in RT60S]
    summary = []
    for cfg in order:
        v = np.array(per_config[cfg])
        det = 'DETECTED' if v.mean() >= 0.5 else 'no'
        print(f'{cfg:20s} {v.mean():10.4f} {v.std():7.4f} {det:>9s}')
        summary.append({'config': cfg, 'mean_conf': round(float(v.mean()), 4),
                        'std': round(float(v.std()), 4), 'n_clips': len(v), 'detected': det})

    # threshold: RT60 where mean conf crosses 0.5
    thr, prev = None, None
    for rt in RT60S:
        c = float(np.mean(per_config[f'reverb_rt60_{rt}']))
        if prev and thr is None and prev[1] >= 0.5 > c:
            r1, c1 = prev; r2, c2 = rt, c
            thr = r1 + (c1 - 0.5) * (r2 - r1) / (c1 - c2)
        prev = (rt, c)
    print('\n==> reverb threshold (conf=0.50):',
          f'RT60 ~ {thr:.2f} s' if thr is not None else 'never crossed -- reverb does NOT break AWARE')

    out = os.path.join(BASE, 'reverb_test_results.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['clip', 'config', 'conf', 'bit_acc', 'detected'])
        w.writeheader(); w.writerows(rows)
    out2 = os.path.join(BASE, 'reverb_test_summary.csv')
    with open(out2, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['config', 'mean_conf', 'std', 'n_clips', 'detected'])
        w.writeheader(); w.writerows(summary)
    print('saved:', out, '\n       ', out2)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        ys = [np.mean(per_config[f'reverb_rt60_{r}']) for r in RT60S]
        es = [np.std(per_config[f'reverb_rt60_{r}']) for r in RT60S]
        plt.figure(figsize=(6, 4))
        plt.errorbar(RT60S, ys, yerr=es, fmt='o-', capsize=3, label='dense reverb (real)')
        plt.axhline(np.mean(per_config['bench_echo_1tap']), color='orange', ls='--',
                    label='benchmark 1-tap echo')
        plt.axhline(0.5, color='r', ls=':', lw=1, label='threshold 0.5')
        plt.xlabel('RT60 (s)  -- bigger = more reverb'); plt.ylabel('detection conf')
        plt.title(f'AWARE vs real reverb ({len(clips)} Emilia clips)')
        plt.ylim(0, 1.05); plt.legend(); plt.tight_layout()
        p = os.path.join(BASE, 'reverb_test.png'); plt.savefig(p, dpi=130)
        print('plot:', p)
    except Exception as e:
        print('(plot skipped:', e, ')')


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
