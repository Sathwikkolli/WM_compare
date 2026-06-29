"""
run_vox.py  --  VoxWatermark no-box benchmark on AudioSeal, AWARE, Timbre (each solo)
and on the combined AudioSeal+AWARE+Timbre file.

For every target we embed the watermark(s) into the one clean clip, then for every
attack x strength in vox_attacks.VOX_GRID we apply the perturbation and detect every
watermark present, recording bit-accuracy, BER (=1-bit_acc), confidence and a
detected flag (bit_acc >= 0.8). CSV is written incrementally and the run RESUMES.

Targets:
  audioseal / aware / timbre  -> one watermark each
  combined                    -> all three cascaded (order is irrelevant, proven)

Usage:
  python run_vox.py --selftest      # apply every attack once to the combined file
  python run_vox.py                 # full sweep (all targets, full 17-attack grid)
  python run_vox.py --attacks encodec,time_stretch,mp3   # subset
  python run_vox.py --targets combined                   # one target
"""
import os, csv, sys, argparse, traceback
import numpy as np

import cascade_lib as L
from cascade_lib import (BASE, AUDIO, SR_MASTER, TOOLS, TOOL_NAME, get_adapter,
                         read_wav, write_wav)
import vox_attacks as V

OUT = os.path.join(BASE, 'vox_out'); os.makedirs(OUT, exist_ok=True)
DEFAULT_SRC = os.path.join(AUDIO, 'client_original_16k.wav')
COMBINED_ORDER = ['audioseal', 'aware', 'timbre']     # any order works
CSV = os.path.join(OUT, 'vox_results.csv')
HEADER = ['target', 'attack', 'strength_label', 'strength_x', 'watermark',
          'bit_acc', 'ber', 'conf', 'detected']

TARGETS = {
    'audioseal': ['audioseal'],
    'aware':     ['aware'],
    'timbre':    ['timbre'],
    'combined':  COMBINED_ORDER,
}


def build_target(name, master):
    """Embed the watermark(s) for a target into the clean master; return wav @22050."""
    cur = master.copy()
    for code in TARGETS[name]:
        cur = get_adapter(code).embed(cur)
    return cur


def detect_all(y, present):
    """Detect every watermark in `present` on wav y@22050 -> list of result tuples."""
    out = []
    for code in present:
        try:
            conf, bits, acc = get_adapter(code).detect(y)
            out.append((TOOL_NAME[code], acc, conf))
        except Exception as e:
            out.append((TOOL_NAME[code], None, None))
    return out


def selftest(master):
    print('=== VOX SELFTEST (apply each attack once to the combined file) ===', flush=True)
    y = build_target('combined', master)
    ok = True
    for name, grid in V.VOX_GRID.items():
        label, param = grid[0]
        try:
            y2 = V.apply(name, param, y, SR_MASTER)
            if y2 is None:
                print(f'  {name:20s} SKIP (dependency/noise missing)', flush=True); continue
            print(f'  {name:20s} OK   ({label}) len={len(y2)}', flush=True)
        except Exception as e:
            ok = False
            print(f'  {name:20s} FAIL  {str(e)[:70]}', flush=True)
    print('=== SELFTEST', 'PASSED' if ok else 'HAD FAILURES', '===', flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=DEFAULT_SRC)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--targets', default='audioseal,aware,timbre,combined')
    ap.add_argument('--attacks', default='all')
    args = ap.parse_args()

    src = args.src if os.path.isabs(args.src) else os.path.join(BASE, args.src)
    master = read_wav(src, SR_MASTER)
    write_wav(os.path.join(OUT, 'clean_master_22k.wav'), master, SR_MASTER)
    print(f'master: {src}  ({len(master)} samp @ {SR_MASTER})', flush=True)

    if args.selftest:
        sys.exit(0 if selftest(master) else 1)

    targets = [t for t in args.targets.split(',') if t in TARGETS]
    attacks = list(V.VOX_GRID) if args.attacks == 'all' else \
              [a for a in args.attacks.split(',') if a in V.VOX_GRID]

    # resume support: skip (target,attack) pairs already fully in the CSV
    done = set()
    if os.path.exists(CSV):
        with open(CSV) as f:
            for r in csv.DictReader(f):
                done.add((r['target'], r['attack']))
    new = (not os.path.exists(CSV)) or os.path.getsize(CSV) == 0
    fh = open(CSV, 'a', newline=''); w = csv.writer(fh)
    if new: w.writerow(HEADER); fh.flush()

    for tname in targets:
        present = TARGETS[tname]
        print(f'\n#### TARGET {tname}  (watermarks: {", ".join(TOOL_NAME[c] for c in present)})', flush=True)
        y_embed = build_target(tname, master)
        # sanity: clean detection
        for nm, acc, conf in detect_all(y_embed, present):
            print(f'   clean  {nm:10s} bit_acc={None if acc is None else round(acc,3)}', flush=True)
        for attack in attacks:
            if (tname, attack) in done:
                print(f'   skip {attack} (already done)', flush=True); continue
            for label, param in V.VOX_GRID[attack]:
                try:
                    y2 = V.apply(attack, param, y_embed, SR_MASTER)
                except Exception as e:
                    print(f'   {attack:18s} {label:8s} ERROR {str(e)[:50]}', flush=True); continue
                if y2 is None:
                    w.writerow([tname, attack, label, V.strength_x(attack, param), 'ALL', '', '', '', 'SKIP'])
                    continue
                xs = V.strength_x(attack, param)
                line = f'   {attack:18s} {label:8s} |'
                for nm, acc, conf in detect_all(y2, present):
                    ber = '' if acc is None else round(1 - acc, 3)
                    det = '' if acc is None else int(acc >= 0.8)
                    w.writerow([tname, attack, label, xs, nm,
                                '' if acc is None else round(acc, 3),
                                ber, '' if conf is None else round(conf, 3), det])
                    line += f' {nm[:2]}={"ERR" if acc is None else round(acc,2)}'
                fh.flush()
                print(line, flush=True)

    fh.close()
    print('\nALL DONE ->', CSV, '\nNow run:  python vox_report.py', flush=True)


if __name__ == '__main__':
    main()
