"""
detect_client_aware.py -- AWARE-only detection across the client's production
stages. AudioSeal and Timbre read empty on these files (only AWARE was embedded),
so this runs just the AWARE detector for a clean, fast, final report.

Reference points from the control run:
    AWARE noise floor (no watermark present) ~= 0.05
    AWARE healthy (clean watermark)          ~= 0.998
    AWARE decision threshold                  = 0.50  (matches detect_aware.py)

Usage (repo root, `wmcompare` env):
    python detect_client_aware.py
    python detect_client_aware.py "client_processed/EP_FAMILY NOW_Stage 06.wav"
"""
import os, sys, csv, glob

BASE = os.environ.get('WM_COMPARE_BASE', os.path.expanduser('~/wm_compare'))
CASCADE = os.path.join(BASE, 'cascade')
CLIENT_DIR = os.path.join(BASE, 'client_processed')
THRESH = 0.5

sys.path.insert(0, CASCADE)
import cascade_lib as cl
import user_key as uk


def stage_label(path):
    b = os.path.basename(path)
    i = b.lower().find('stage')
    return b[i:b.rfind('.')].strip() if i >= 0 else b


def main(argv):
    targets = argv or sorted(glob.glob(os.path.join(CLIENT_DIR, '*.wav')) +
                             glob.glob(os.path.join(CLIENT_DIR, '*.mp3')))
    targets = sorted(targets, key=stage_label)
    if not targets:
        print('no files found in', CLIENT_DIR); return 1

    print('loading AWARE detector...')
    adapter = cl.get_adapter('aware')

    rows = []
    hdr = f'{"stage":12s} {"AW_conf":>8s} {"detected":>9s} {"rec_id":>7s}  bits'
    print('\n' + hdr); print('-' * (len(hdr) + 10))
    for path in targets:
        try:
            y = cl.read_wav(path)                 # mono @ 22.05k
            conf, bits, _ = adapter.detect(y)
            rec_id = uk.recover_id(bits)
            det = 'DETECTED' if conf >= THRESH else 'no'
            rows.append({'stage': stage_label(path), 'file': os.path.basename(path),
                         'aware_conf': round(conf, 4), 'detected': det,
                         'recovered_id': rec_id, 'bits': bits})
            print(f'{stage_label(path):12s} {conf:8.4f} {det:>9s} {rec_id:7d}  {bits}')
        except Exception as e:
            print(f'{stage_label(path):12s} ERROR: {e}')

    out = os.path.join(BASE, 'client_aware_results.csv')
    if rows:
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print('\nfloor~0.05  healthy~0.998  threshold=0.50')
        print('saved:', out)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
