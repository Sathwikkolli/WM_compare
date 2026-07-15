"""
music_test_emilia.py -- music-bed SNR sweep for ALL THREE watermarks on real
Emilia speech. Companion to reverb_test_emilia.py; together they give the full
3 models x 2 attacks matrix.

For each model, for each clip: embed -> mix real music at a controlled SNR ->
detect. The SNR where mean confidence crosses the model's threshold is that
model's music-bed breaking point.

    SNR_dB = 10*log10(P_speech / P_music)
    music scale a = sqrt(P_speech / (P_music * 10^(SNR/10)))
    mix = speech + a*music

Thresholds
    aware / audioseal : 0.50  (real detection probability)
    timbre            : 0.80  (conf == bit_acc; chance is 0.50 for 10 bits,
                               so we use the vox "survives" bar)

Usage (repo root, wmcompare env):
    python music_test_emilia.py
    python music_test_emilia.py --n 5 --models aware,audioseal,timbre
"""
import os, sys, csv, math, subprocess, numpy as np

BASE = os.environ.get('WM_COMPARE_BASE', os.path.expanduser('~/wm_compare'))
CASCADE = os.path.join(BASE, 'cascade')
DATA = os.path.join(BASE, 'real_audio'); os.makedirs(DATA, exist_ok=True)
EMILIA_CSV = "/nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_curated.csv"
MUSIC_URL = "https://archive.org/download/MarchForHonor/March_For_Honor.mp3"   # CC0 stereo
MUSIC_RAW = os.path.join(DATA, 'music_cc0.mp3')

SNRS = [30, 20, 15, 10, 8, 6, 5, 4, 3, 2, 1, 0, -5, -10]
MIN_DUR = 9.0
THRESH = {'aware': 0.5, 'audioseal': 0.5, 'timbre': 0.8}

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


def get_music(n):
    if not os.path.exists(MUSIC_RAW):
        print('downloading CC0 music ...')
        subprocess.run(['curl', '-L', '-s', '-o', MUSIC_RAW, MUSIC_URL], check=True)
    m = cl.read_wav(MUSIC_RAW)                 # detector downmixes anyway -> mono
    if len(m) < n:
        m = np.tile(m, int(np.ceil(n / len(m))))
    return m[:n].astype('float32')


def threshold_cross(snrs, confs, thresh):
    """Interpolate the SNR where mean conf crosses `thresh` (sweeping high->low)."""
    prev = None
    for s, c in zip(snrs, confs):
        if prev and prev[1] >= thresh > c:
            s1, c1 = prev; s2, c2 = s, c
            return s2 + (thresh - c2) * (s1 - s2) / (c1 - c2)
        prev = (s, c)
    return None


def main(argv):
    n_clips = get_arg(argv, '--n', 5, int)
    models  = get_arg(argv, '--models', 'aware,audioseal,timbre').split(',')
    csv_in  = get_arg(argv, '--csv', EMILIA_CSV)

    clips = pick_clips(csv_in, n_clips)
    if not clips:
        print('no usable clips found in', csv_in); return 1
    print(f'{len(clips)} Emilia clips (>= {MIN_DUR}s) | models={models}\n')

    rows, curves = [], {}
    for model in models:
        thresh = THRESH.get(model, 0.5)
        print(f'===== {model}  (threshold {thresh}) =====')
        adapter = cl.get_adapter(model)
        per_snr = {s: [] for s in SNRS}
        base_confs = []

        for i, path in enumerate(clips, 1):
            name = os.path.basename(path)
            y = cl.read_wav(path)
            print(f'  [{i}/{len(clips)}] embedding {model} into {name} ...')
            y_wm = np.asarray(adapter.embed(y), dtype='float32')
            c0, _, b0 = adapter.detect(y_wm)
            base_confs.append(float(c0))
            rows.append({'model': model, 'clip': name, 'snr_db': 'baseline',
                         'conf': round(float(c0), 4), 'bit_acc': round(float(b0), 4),
                         'detected': 'DETECTED' if c0 >= thresh else 'no'})

            m = get_music(len(y_wm))
            Ps, Pm = float(np.mean(y_wm ** 2)), float(np.mean(m ** 2))
            for snr in SNRS:
                a = math.sqrt(Ps / (Pm * (10 ** (snr / 10.0))))
                mix = (y_wm + a * m).astype('float32')
                conf, _, bacc = adapter.detect(mix)
                per_snr[snr].append(float(conf))
                rows.append({'model': model, 'clip': name, 'snr_db': snr,
                             'conf': round(float(conf), 4), 'bit_acc': round(float(bacc), 4),
                             'detected': 'DETECTED' if conf >= thresh else 'no'})

        means = [float(np.mean(per_snr[s])) for s in SNRS]
        stds  = [float(np.std(per_snr[s])) for s in SNRS]
        curves[model] = (means, stds, thresh)
        thr = threshold_cross(SNRS, means, thresh)

        print(f'\n  baseline mean conf = {np.mean(base_confs):.4f}')
        print(f'  {"SNR_dB":>7s} {"mean":>8s} {"std":>7s} {"detected":>9s}')
        for s, mu, sd in zip(SNRS, means, stds):
            print(f'  {s:7d} {mu:8.4f} {sd:7.4f} {"DETECTED" if mu >= thresh else "no":>9s}')
        print(f'  ==> {model} music threshold: '
              + (f'{thr:.1f} dB SNR' if thr is not None else
                 ('never crossed (survives whole range)' if means[-1] >= thresh
                  else f'already below {thresh} at {SNRS[0]} dB')) + '\n')

    out = os.path.join(BASE, 'music_test_results.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'clip', 'snr_db', 'conf', 'bit_acc', 'detected'])
        w.writeheader(); w.writerows(rows)

    summ = os.path.join(BASE, 'music_test_summary.csv')
    with open(summ, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['model', 'snr_db', 'mean_conf', 'std', 'threshold', 'detected'])
        for model, (means, stds, th) in curves.items():
            for s, mu, sd in zip(SNRS, means, stds):
                w.writerow([model, s, round(mu, 4), round(sd, 4), th, 'DETECTED' if mu >= th else 'no'])
    print('saved:', out, '\n       ', summ)

    print('\n===== THRESHOLD SUMMARY =====')
    for model, (means, stds, th) in curves.items():
        thr = threshold_cross(SNRS, means, th)
        print(f'{model:10s} threshold={th}  ->  '
              + (f'{thr:.1f} dB SNR' if thr is not None else 'not crossed in range'))

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4.5))
        for model, (means, stds, th) in curves.items():
            plt.errorbar(SNRS, means, yerr=stds, fmt='o-', capsize=3, label=f'{model} (thr {th})')
        plt.axhline(0.5, color='r', ls=':', lw=1, label='threshold 0.5')
        plt.axhline(0.8, color='m', ls=':', lw=1, label='threshold 0.8 (timbre)')
        plt.gca().invert_xaxis()
        plt.xlabel('speech-to-music SNR (dB)  -- lower = louder music')
        plt.ylabel('detection conf'); plt.ylim(0, 1.05)
        plt.title(f'Music bed vs 3 watermarks ({len(clips)} Emilia clips)')
        plt.legend(); plt.tight_layout()
        p = os.path.join(BASE, 'music_test.png'); plt.savefig(p, dpi=130)
        print('plot:', p)
    except Exception as e:
        print('(plot skipped:', e, ')')


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
