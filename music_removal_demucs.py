"""
music_removal_demucs.py -- can stripping the music bed back out RESCUE the watermark?

The idea
--------
Mixing music is additive and linear:  mix = watermarked_speech + a*music
Nothing is deleted -- the watermark is still physically in the waveform, just
MASKED. So removing the music should un-mask it.

The catch
---------
A neural separator (Demucs) does not subtract, it RESYNTHESISES the vocals. That
rebuild is exactly the kind of process known to destroy watermarks (the SoK
reports neural resynthesis causing up to 50% BER). So separation both helps
(removes the masker) and hurts (rebuilds the signal). Which wins is the question.

Conditions per clip / model
---------------------------
  baseline          watermarked speech                 -> reference
  demucs_clean      demucs(watermarked speech)         -> *** THE CONTROL ***
                    no music involved. If this tanks, the separator itself
                    destroys the watermark and the whole approach is dead.
  mix_<snr>         speech + music at <snr> dB         -> the failure we measured
  demucs_mix_<snr>  demucs(mix)                        -> does removal rescue it?

Read it as: compare demucs_mix_<snr> against mix_<snr> (rescue?) and against
demucs_clean (how much of the loss is the separator's own damage?).

Usage (repo root, wmcompare env; needs `pip install demucs`):
    python music_removal_demucs.py
    python music_removal_demucs.py --n 3 --models aware,timbre --device cuda
"""
import os, sys, csv, math, shutil, subprocess, numpy as np

BASE = os.environ.get('WM_COMPARE_BASE', os.path.expanduser('~/wm_compare'))
CASCADE = os.path.join(BASE, 'cascade')
WORK = os.path.join(BASE, 'demucs_work'); os.makedirs(WORK, exist_ok=True)
SEP = os.path.join(WORK, 'sep'); os.makedirs(SEP, exist_ok=True)
EMILIA_CSV = "/nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_curated.csv"
MUSIC_URL = "https://archive.org/download/MarchForHonor/March_For_Honor.mp3"
MUSIC_RAW = os.path.join(BASE, 'real_audio', 'music_cc0.mp3')

SNRS = [10, 5, 0, -5]
MIN_DUR = 9.0
THRESH = {'aware': 0.5, 'audioseal': 0.5, 'timbre': 0.8}

sys.path.insert(0, CASCADE)
import cascade_lib as cl
SR = cl.SR_MASTER


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def check_demucs():
    if shutil.which('demucs') is None:
        print('ERROR: demucs not found. Install it first:\n    pip install demucs')
        return False
    return True


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
    os.makedirs(os.path.dirname(MUSIC_RAW), exist_ok=True)
    if not os.path.exists(MUSIC_RAW):
        subprocess.run(['curl', '-L', '-s', '-o', MUSIC_RAW, MUSIC_URL], check=True)
    m = cl.read_wav(MUSIC_RAW)
    if len(m) < n:
        m = np.tile(m, int(np.ceil(n / len(m))))
    return m[:n].astype('float32')


def separate(sig, tag, device='cpu'):
    """Write sig -> run Demucs two-stem -> return the separated VOCALS as mono@SR."""
    src = os.path.join(WORK, f'{tag}.wav')
    cl.write_wav(src, sig.astype('float32'), sr=SR)
    cmd = ['demucs', '--two-stems=vocals', '-o', SEP, '-d', device, src]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # demucs writes {SEP}/{model}/{tag}/vocals.wav
    for model_dir in sorted(os.listdir(SEP)):
        p = os.path.join(SEP, model_dir, tag, 'vocals.wav')
        if os.path.exists(p):
            return cl.read_wav(p)          # resamples to SR, mono
    raise FileNotFoundError(f'demucs produced no vocals stem for {tag}')


def main(argv):
    if not check_demucs():
        return 1
    n_clips = get_arg(argv, '--n', 3, int)
    models = get_arg(argv, '--models', 'aware,timbre').split(',')
    device = get_arg(argv, '--device', 'cpu')
    csv_in = get_arg(argv, '--csv', EMILIA_CSV)

    clips = pick_clips(csv_in, n_clips)
    if not clips:
        print('no usable clips found in', csv_in); return 1
    print(f'{len(clips)} clips | models={models} | device={device}')
    print(f'demucs runs: {len(models) * len(clips) * (1 + len(SNRS))} '
          f'(slow on CPU -- use --device cuda on a GPU node)\n')

    rows, agg = [], {}
    for model in models:
        thresh = THRESH.get(model, 0.5)
        adapter = cl.get_adapter(model)
        print(f'===== {model} (threshold {thresh}) =====')
        for i, path in enumerate(clips, 1):
            name = os.path.basename(path)
            base_tag = f'{model}_{i}'
            y = cl.read_wav(path)
            print(f'  [{i}/{len(clips)}] embedding {model} into {name} ...')
            y_wm = np.asarray(adapter.embed(y), dtype='float32')

            def record(cond, sig):
                conf, _, bacc = adapter.detect(np.asarray(sig, dtype='float32'))
                agg.setdefault((model, cond), []).append(float(conf))
                rows.append({'model': model, 'clip': name, 'condition': cond,
                             'conf': round(float(conf), 4), 'bit_acc': round(float(bacc), 4),
                             'detected': 'DETECTED' if conf >= thresh else 'no'})
                return float(conf)

            record('baseline', y_wm)
            # --- THE CONTROL: separator on clean watermarked speech, no music ---
            c_ctrl = record('demucs_clean', separate(y_wm, f'{base_tag}_clean', device))
            print(f'      demucs_clean (control) = {c_ctrl:.3f}')

            m = get_music(len(y_wm))
            Ps, Pm = float(np.mean(y_wm ** 2)), float(np.mean(m ** 2))
            for snr in SNRS:
                a = math.sqrt(Ps / (Pm * (10 ** (snr / 10.0))))
                mix = (y_wm + a * m).astype('float32')
                c_mix = record(f'mix_{snr}dB', mix)
                c_sep = record(f'demucs_mix_{snr}dB',
                               separate(mix, f'{base_tag}_mix{snr}', device))
                print(f'      {snr:>3} dB: mix={c_mix:.3f} -> after removal={c_sep:.3f} '
                      f'({"+" if c_sep >= c_mix else ""}{c_sep - c_mix:.3f})')

    # ---- summary -----------------------------------------------------------
    print('\n===== MEAN over clips =====')
    conds = ['baseline', 'demucs_clean'] + \
            [c for s in SNRS for c in (f'mix_{s}dB', f'demucs_mix_{s}dB')]
    summary = []
    for model in models:
        th = THRESH.get(model, 0.5)
        print(f'\n-- {model} (threshold {th}) --')
        print(f'{"condition":22s} {"mean":>8s} {"std":>7s} {"detected":>9s}')
        for cond in conds:
            v = np.array(agg.get((model, cond), []))
            if not len(v):
                continue
            det = 'DETECTED' if v.mean() >= th else 'no'
            print(f'{cond:22s} {v.mean():8.4f} {v.std():7.4f} {det:>9s}')
            summary.append({'model': model, 'condition': cond,
                            'mean_conf': round(float(v.mean()), 4),
                            'std': round(float(v.std()), 4),
                            'threshold': th, 'detected': det})
        # verdict
        ctrl = np.mean(agg.get((model, 'demucs_clean'), [0]))
        print(f'\n  CONTROL demucs_clean = {ctrl:.3f} -> '
              + ('separator is watermark-SAFE; removal can help'
                 if ctrl >= th else
                 'separator DESTROYS the watermark by itself -- removal cannot rescue it'))
        for s in SNRS:
            mix = np.mean(agg.get((model, f'mix_{s}dB'), [0]))
            sep = np.mean(agg.get((model, f'demucs_mix_{s}dB'), [0]))
            verdict = 'RESCUED' if (sep >= th and mix < th) else \
                      ('improved' if sep > mix else 'no help')
            print(f'  {s:>3} dB: {mix:.3f} -> {sep:.3f}   {verdict}')

    out = os.path.join(BASE, 'music_removal_results.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'clip', 'condition', 'conf', 'bit_acc', 'detected'])
        w.writeheader(); w.writerows(rows)
    out2 = os.path.join(BASE, 'music_removal_summary.csv')
    with open(out2, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'condition', 'mean_conf', 'std', 'threshold', 'detected'])
        w.writeheader(); w.writerows(summary)
    print('\nsaved:', out, '\n       ', out2)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4), squeeze=False)
        for ax, model in zip(axes[0], models):
            th = THRESH.get(model, 0.5)
            mixes = [np.mean(agg.get((model, f'mix_{s}dB'), [0])) for s in SNRS]
            seps = [np.mean(agg.get((model, f'demucs_mix_{s}dB'), [0])) for s in SNRS]
            ax.plot(SNRS, mixes, 'o-', label='with music (masked)')
            ax.plot(SNRS, seps, 's-', label='after demucs removal')
            ax.axhline(np.mean(agg.get((model, 'demucs_clean'), [0])), color='g', ls='--',
                       label='demucs on clean (control)')
            ax.axhline(th, color='r', ls=':', lw=1, label=f'threshold {th}')
            ax.invert_xaxis(); ax.set_ylim(0, 1.05)
            ax.set_xlabel('speech-to-music SNR (dB)'); ax.set_ylabel('detection conf')
            ax.set_title(model); ax.legend(fontsize=8)
        plt.tight_layout()
        p = os.path.join(BASE, 'music_removal.png'); plt.savefig(p, dpi=130)
        print('plot:', p)
    except Exception as e:
        print('(plot skipped:', e, ')')


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
