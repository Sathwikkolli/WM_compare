"""
client_music_removal.py -- strip the music bed off the client's music-branch
stages (06, 07B, 08B) with Demucs, then re-detect the AWARE watermark.

We verified the embedded message is REF below (conf 0.998, 20/20 on the source).
So here we can report, before vs after music removal, both the detection
confidence and a real correlation score against the known payload.

    xcorr = 2*bit_accuracy - 1   (recovered bits vs REF, range -1..+1)

IMPORTANT expectation: Demucs removes the MUSIC, but NOT the reverb these stages
inherited from Stage 05. Reverb is not reversible (see the WPE test), and Stage 05
(reverb, no music) already sat at ~0.46. So the realistic ceiling here is roughly
Stage 05's level, not full recovery.

Usage (repo root, wmcompare env; demucs installed):
    python client_music_removal.py
"""
import os, sys, csv, glob, shutil, subprocess, numpy as np

BASE = os.environ.get('WM_COMPARE_BASE', os.path.expanduser('~/wm_compare'))
CASCADE = os.path.join(BASE, 'cascade')
CLIENT = os.path.join(BASE, 'client_processed')
WORK = os.path.join(BASE, 'client_demucs'); os.makedirs(WORK, exist_ok=True)
SEP = os.path.join(WORK, 'sep'); os.makedirs(SEP, exist_ok=True)
REF = '01101111111001000001'
MUSIC_STAGES = ['06', '07B', '08B']

sys.path.insert(0, CASCADE)
import cascade_lib as cl
SR = cl.SR_MASTER


def find_stage(stage):
    hits = glob.glob(os.path.join(CLIENT, '**', f'*Stage {stage}.*'), recursive=True)
    return hits[0] if hits else None


def score(bits):
    m = sum(a == b for a, b in zip(bits, REF))
    ba = m / 20.0
    return ba, 2 * ba - 1


def separate(path, tag):
    """Demucs two-stem; return the vocals stem as mono@SR."""
    subprocess.run(['demucs', '--two-stems=vocals', '-o', SEP, '-d', 'cpu', path],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for md in sorted(os.listdir(SEP)):
        p = os.path.join(SEP, md, tag, 'vocals.wav')
        if os.path.exists(p):
            return cl.read_wav(p)
    raise FileNotFoundError(f'no vocals stem for {tag}')


def main():
    if shutil.which('demucs') is None:
        print('demucs not installed:  pip install demucs'); return 1
    ad = cl.get_adapter('aware')
    rows = []
    print('reference message:', REF, '\n')
    hdr = f'{"stage":8s} | {"conf_before":>11s} {"xcorr_b":>8s} {"det":>4s} | ' \
          f'{"conf_after":>10s} {"xcorr_a":>8s} {"det":>4s}'
    print(hdr); print('-' * len(hdr))
    for st in MUSIC_STAGES:
        f = find_stage(st)
        if not f:
            print(f'St{st:6s} | file not found in client_processed/'); continue
        # before: with music
        y = cl.read_wav(f)
        cb, bb, _ = ad.detect(y); bab, xcb = score(bb)
        # after: music stripped by demucs
        tag = os.path.splitext(os.path.basename(f))[0]
        try:
            v = separate(f, tag)
            ca, ba_bits, _ = ad.detect(np.asarray(v, dtype='float32')); baa, xca = score(ba_bits)
        except Exception as e:
            print(f'St{st:6s} | demucs failed: {e}'); continue
        print(f'St{st:6s} | {cb:11.3f} {xcb:8.3f} {"Y" if cb>=0.5 else "N":>4s} | '
              f'{ca:10.3f} {xca:8.3f} {"Y" if ca>=0.5 else "N":>4s}')
        rows.append({'stage': f'Stage {st}', 'conf_before': round(cb, 4),
                     'bitacc_before': round(bab, 3), 'xcorr_before': round(xcb, 3),
                     'conf_after': round(ca, 4), 'bitacc_after': round(baa, 3),
                     'xcorr_after': round(xca, 3)})

    if rows:
        out = os.path.join(BASE, 'client_music_removal.csv')
        with open(out, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print('\nsaved:', out)
        print('\nNote: Demucs removes music but NOT the Stage-05 reverb these stages')
        print('carry, so the ceiling here is ~Stage 05 (0.46), not full recovery.')


if __name__ == '__main__':
    raise SystemExit(main())
