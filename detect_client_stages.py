"""
detect_client_stages.py -- run the 3-watermark detector on the client's
production-pipeline files (Stages 02..08) and report, per stage, whether our
watermark is still PRESENT.

Presence needs no user_id: AudioSeal and AWARE output a detection probability
that is independent of the embedded message ("the message has no influence on
the detection output" -- AudioSeal). So conf >= THRESH => watermark present.
The recovered user_id is reported too (for free), but is only meaningful once
you compare it against the id you actually embedded.

Usage (run from repo root, on Great Lakes with the `wmcompare` env):
    python detect_client_stages.py                 # all files in client_processed/
    python detect_client_stages.py "client_processed/EP_FAMILY NOW_Stage 02.wav"
    python detect_client_stages.py --baseline audio/native_wm_full.wav   # add a ref row first

If client_processed/ is empty it auto-downloads the client's Dropbox folder.
"""
import os, sys, csv, glob, subprocess

BASE = os.environ.get('WM_COMPARE_BASE', os.path.expanduser('~/wm_compare'))
CASCADE = os.path.join(BASE, 'cascade')
CLIENT_DIR = os.path.join(BASE, 'client_processed')
DROPBOX_ZIP = ("https://www.dropbox.com/scl/fo/o1adko6sziyqo5i0n2hfs/"
               "AJDnEsqZyOWV4Qu07LbZQhM?rlkey=h6juuyn1crmvntc04jutv1uql&dl=1")
THRESH = 0.5   # AudioSeal / AWARE presence threshold

sys.path.insert(0, CASCADE)   # cascade modules import each other bare (service, cascade_lib, user_key)
import service                 # noqa: E402  (uses cascade_lib + user_key internally)


def ensure_files():
    """Make sure the client stage files are present; fetch + unzip if not."""
    os.makedirs(CLIENT_DIR, exist_ok=True)
    have = glob.glob(os.path.join(CLIENT_DIR, '*.wav')) + glob.glob(os.path.join(CLIENT_DIR, '*.mp3'))
    if have:
        return have
    print('client_processed/ empty -> downloading Dropbox folder...')
    zip_path = os.path.join(CLIENT_DIR, 'client_files.zip')
    subprocess.run(['curl', '-L', '-s', '-o', zip_path, DROPBOX_ZIP], check=True)
    # -j junk paths: the archive has a leading "/" entry that trips unzip otherwise
    subprocess.run(['unzip', '-o', '-j', '-q', zip_path, '-d', CLIENT_DIR])
    return sorted(glob.glob(os.path.join(CLIENT_DIR, '*.wav')) +
                  glob.glob(os.path.join(CLIENT_DIR, '*.mp3')))


def stage_label(path):
    """'.../EP_FAMILY NOW_Stage 07A.wav' -> 'Stage 07A'."""
    b = os.path.basename(path)
    i = b.lower().find('stage')
    return b[i:b.rfind('.')].strip() if i >= 0 else b


def run_one(path):
    """Detect on one file; return a result row dict."""
    r = service.detect(path)          # {'consensus_user_id','agreement','per_model':{...}}
    pm = r['per_model']
    a, w, t = pm['audioseal'], pm['aware'], pm['timbre']
    # Presence decision from the two keyless models that expose a real prob.
    present = (a['confidence'] >= THRESH) or (w['confidence'] >= THRESH)
    return {
        'file': os.path.basename(path),
        'stage': stage_label(path),
        'audioseal_conf': a['confidence'], 'audioseal_id': a['user_id'],
        'aware_conf':     w['confidence'], 'aware_id':     w['user_id'],
        'timbre_conf':    t['confidence'], 'timbre_id':    t['user_id'],
        'consensus_id':   r['consensus_user_id'], 'agreement': r['agreement'],
        'present': 'YES' if present else 'NO',
    }


def main(argv):
    args = [a for a in argv if not a.startswith('--')]
    baseline = None
    if '--baseline' in argv:
        baseline = argv[argv.index('--baseline') + 1]

    targets = args if args else ensure_files()
    targets = sorted(targets, key=lambda p: stage_label(p))
    if baseline:
        targets = [baseline] + targets

    print('warming up the three detectors (first load is slow)...')
    service.warmup()

    rows = []
    hdr = f'{"stage":12s} {"present":7s} {"AS_conf":>8s} {"AW_conf":>8s} ' \
          f'{"TB_conf":>8s} {"AS_id":>6s} {"AW_id":>6s} {"TB_id":>6s} {"agree":>6s}'
    print('\n' + hdr); print('-' * len(hdr))
    for path in targets:
        try:
            row = run_one(path)
            rows.append(row)
            print(f'{row["stage"]:12s} {row["present"]:7s} '
                  f'{row["audioseal_conf"]:8.3f} {row["aware_conf"]:8.3f} '
                  f'{row["timbre_conf"]:8.3f} {row["audioseal_id"]:6d} '
                  f'{row["aware_id"]:6d} {row["timbre_id"]:6d} {row["agreement"]:>6s}')
        except Exception as e:
            print(f'{stage_label(path):12s} ERROR: {e}')

    out = os.path.join(BASE, 'client_stage_results.csv')
    if rows:
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print('\nsaved:', out)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
