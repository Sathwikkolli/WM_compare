"""
stereo_roundtrip_test.py -- does a mono -> stereo -> mono round trip damage the
watermark? Run for all three models on one clip.

Chain (real ffmpeg channel conversions, the same ones a DAW / RX batch would do):

    1. embed          mono watermarked speech            -> baseline
    2. mono -> stereo `ffmpeg -ac 2`  (duplicates the channel: L = R)
       detect          (the detector must downmix stereo back to mono to read it)
    3. stereo -> mono  `ffmpeg -ac 1` (averages the channels: (L+R)/2)
       detect

Why this matters
----------------
The client's Stage 05 exported stereo and Stage 07B/08B stayed stereo. This test
separates "the file became stereo" from "a stereo WIDENER decorrelated it".
A plain dual-mono conversion is mathematically lossless -- (L+R)/2 == the original
when L == R -- so any confidence change here would point to a codec/IO artifact
rather than the channel count itself.

Usage (repo root, wmcompare env):
    python stereo_roundtrip_test.py
    python stereo_roundtrip_test.py --clip /path/to.wav --models aware,audioseal,timbre
"""
import os, sys, csv, subprocess, numpy as np

BASE = os.environ.get('WM_COMPARE_BASE', os.path.expanduser('~/wm_compare'))
CASCADE = os.path.join(BASE, 'cascade')
WORK = os.path.join(BASE, 'stereo_work'); os.makedirs(WORK, exist_ok=True)
EMILIA_CSV = "/nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_curated.csv"
MIN_DUR = 9.0
THRESH = {'aware': 0.5, 'audioseal': 0.5, 'timbre': 0.8}

sys.path.insert(0, CASCADE)
import cascade_lib as cl
SR = cl.SR_MASTER


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def sh(*a):
    subprocess.run(list(a), check=True)


def first_clip(csv_path):
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                if float(row['duration_s']) >= MIN_DUR and os.path.exists(row['path']):
                    return row['path']
            except (KeyError, ValueError):
                continue
    return None


def probe_channels(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                          '-show_entries', 'stream=channels', '-of', 'csv=p=0', path],
                         capture_output=True, text=True).stdout.strip()
    return out or '?'


def main(argv):
    clip = get_arg(argv, '--clip', None) or first_clip(get_arg(argv, '--csv', EMILIA_CSV))
    models = get_arg(argv, '--models', 'aware,audioseal,timbre').split(',')
    if not clip:
        print('no clip found'); return 1
    print(f'clip   : {os.path.basename(clip)}')
    print(f'models : {models}\n')

    y = cl.read_wav(clip)
    rows = []

    for model in models:
        thresh = THRESH.get(model, 0.5)
        adapter = cl.get_adapter(model)
        print(f'===== {model} (threshold {thresh}) =====')
        y_wm = np.asarray(adapter.embed(y), dtype='float32')

        p_mono = os.path.join(WORK, f'{model}_1_mono.wav')
        p_st   = os.path.join(WORK, f'{model}_2_stereo.wav')
        p_back = os.path.join(WORK, f'{model}_3_mono_again.wav')
        cl.write_wav(p_mono, y_wm, sr=SR)

        # real channel conversions, exactly what a DAW/RX batch export does
        sh('ffmpeg', '-y', '-loglevel', 'error', '-i', p_mono, '-ac', '2', p_st)
        sh('ffmpeg', '-y', '-loglevel', 'error', '-i', p_st, '-ac', '1', p_back)

        steps = [('1_baseline_mono', p_mono), ('2_stereo', p_st), ('3_mono_again', p_back)]
        confs = {}
        print(f'{"step":18s} {"ch":>3s} {"conf":>8s} {"bit_acc":>8s} {"detected":>9s}')
        for label, path in steps:
            sig = cl.read_wav(path)            # read_wav downmixes stereo -> mono
            conf, _, bacc = adapter.detect(np.asarray(sig, dtype='float32'))
            confs[label] = float(conf)
            det = 'DETECTED' if conf >= thresh else 'no'
            ch = probe_channels(path)
            print(f'{label:18s} {ch:>3s} {conf:8.4f} {bacc:8.4f} {det:>9s}')
            rows.append({'model': model, 'step': label, 'channels': ch,
                         'conf': round(float(conf), 4), 'bit_acc': round(float(bacc), 4),
                         'detected': det})

        # is the round trip numerically lossless?
        a = cl.read_wav(p_mono); b = cl.read_wav(p_back)
        n = min(len(a), len(b))
        maxdiff = float(np.max(np.abs(a[:n] - b[:n]))) if n else float('nan')
        d_st = confs['2_stereo'] - confs['1_baseline_mono']
        d_bk = confs['3_mono_again'] - confs['1_baseline_mono']
        print(f'\n  delta vs baseline:  stereo {d_st:+.4f}   mono_again {d_bk:+.4f}')
        print(f'  max sample diff (baseline vs round-tripped): {maxdiff:.2e}')
        print('  -> ' + ('round trip is LOSSLESS for the watermark'
                         if abs(d_bk) < 0.01 else
                         'round trip CHANGED the watermark confidence') + '\n')

    out = os.path.join(BASE, 'stereo_roundtrip_results.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'step', 'channels', 'conf', 'bit_acc', 'detected'])
        w.writeheader(); w.writerows(rows)
    print('saved:', out)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
