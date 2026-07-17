"""
client_correlation_all.py -- correlation (XCorr) score for EVERY client stage
(02 -> 08B) against the verified embedded message.

Reference payload REF was confirmed on the source file (conf 0.998, 20/20).
    xcorr = 2*bit_accuracy - 1   (recovered bits vs REF, range -1..+1)

For the MUSIC-branch stages (06, 07B, 08B) we first strip the music with Demucs,
then detect -- the correlation reported for those stages is the POST-removal value
(the "with music" value is shown too for reference). All other stages are scored
directly.

Note: Demucs removes music but not the Stage-05 reverb these stages inherit, so
their ceiling is ~Stage 05 (0.46), not full recovery.

Usage (repo root, wmcompare env; demucs installed):
    python client_correlation_all.py
"""
import os, sys, csv, glob, shutil, subprocess, numpy as np

BASE = os.environ.get('WM_COMPARE_BASE', os.path.expanduser('~/wm_compare'))
CASCADE = os.path.join(BASE, 'cascade')
CLIENT = os.path.join(BASE, 'client_processed')
WORK = os.path.join(BASE, 'client_demucs'); os.makedirs(WORK, exist_ok=True)
SEP = os.path.join(WORK, 'sep'); os.makedirs(SEP, exist_ok=True)
REF = '01101111111001000001'
STAGES = ['02', '03', '04', '05', '06', '07A', '07B', '08A', '08B']
MUSIC_STAGES = {'06', '07B', '08B'}

sys.path.insert(0, CASCADE)
import cascade_lib as cl
SR = cl.SR_MASTER


def find_stage(stage):
    hits = glob.glob(os.path.join(CLIENT, '**', f'*Stage {stage}.*'), recursive=True)
    return hits[0] if hits else None


def score(bits):
    m = sum(a == b for a, b in zip(bits, REF)); ba = m / 20.0
    return ba, 2 * ba - 1


def separate(path, tag):
    subprocess.run(['demucs', '--two-stems=vocals', '-o', SEP, '-d', 'cpu', path],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for md in sorted(os.listdir(SEP)):
        p = os.path.join(SEP, md, tag, 'vocals.wav')
        if os.path.exists(p):
            return cl.read_wav(p)
    raise FileNotFoundError(f'no vocals stem for {tag}')


def main():
    have_demucs = shutil.which('demucs') is not None
    if not have_demucs:
        print('WARNING: demucs not installed -- music stages will be scored WITH music.\n')
    ad = cl.get_adapter('aware')
    rows = []
    print('reference message:', REF, '\n')
    hdr = f'{"stage":7s} {"conf":>7s} {"bit_acc":>8s} {"xcorr":>7s}  {"note":s}'
    print(hdr); print('-' * (len(hdr) + 8))
    for st in STAGES:
        f = find_stage(st)
        if not f:
            print(f'{("St"+st):7s}  file not found'); continue
        y = cl.read_wav(f)
        conf, bits, _ = ad.detect(y); ba, xc = score(bits)
        note = ''
        if st in MUSIC_STAGES and have_demucs:
            xc_before = xc
            try:
                tag = os.path.splitext(os.path.basename(f))[0]
                v = separate(f, tag)
                conf, bits, _ = ad.detect(np.asarray(v, dtype='float32')); ba, xc = score(bits)
                note = f'music removed (was xcorr {xc_before:.2f})'
            except Exception as e:
                note = f'demucs failed: {e}'
        print(f'{("St"+st):7s} {conf:7.3f} {ba:8.3f} {xc:7.3f}  {note}')
        rows.append({'stage': f'Stage {st}', 'conf': round(conf, 4),
                     'bit_acc': round(ba, 3), 'xcorr': round(xc, 3),
                     'music_removed': st in MUSIC_STAGES and have_demucs, 'note': note})

    out = os.path.join(BASE, 'client_correlation_all.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print('\nsaved:', out)


if __name__ == '__main__':
    raise SystemExit(main())
