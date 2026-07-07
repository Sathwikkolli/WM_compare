"""
run_vox_aura.py  --  VoxWatermark no-box benchmark for AURA.

Runs AURA through the *exact same* 17-attack VoxWatermark grid used for
AudioSeal / AWARE / Timbre (vox_attacks.VOX_GRID), writing results in the same
CSV schema as run_vox.py so the two can be merged for comparison.

Output: vox_out/vox_results_aura.csv   (target='aura', watermark='AURA')

Usage:
  python run_vox_aura.py --selftest                 # apply each attack once
  python run_vox_aura.py                            # full grid
  python run_vox_aura.py --attacks encodec,mp3      # subset
  AURA_CKPT=/path/to/step.pt python run_vox_aura.py # override checkpoint
"""
import os, csv, sys, argparse

import cascade_lib as L
from cascade_lib import BASE, AUDIO, SR_MASTER, read_wav, write_wav
import vox_attacks as V
from aura_adapter import AuraAdapter

OUT = os.path.join(BASE, 'vox_out'); os.makedirs(OUT, exist_ok=True)
DEFAULT_SRC = os.path.join(AUDIO, 'client_original_16k.wav')
CSV = os.path.join(OUT, 'vox_results_aura.csv')
HEADER = ['target', 'attack', 'strength_label', 'strength_x', 'watermark',
          'bit_acc', 'ber', 'conf', 'detected']
NAME = 'AURA'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=DEFAULT_SRC)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--attacks', default='all')
    ap.add_argument('--checkpoint', default=None)
    args = ap.parse_args()

    src = args.src if os.path.isabs(args.src) else os.path.join(BASE, args.src)
    master = read_wav(src, SR_MASTER)
    print(f'master: {src}  ({len(master)} samp @ {SR_MASTER})', flush=True)

    ad = AuraAdapter(checkpoint=args.checkpoint)
    ad.load()
    print(f'AURA loaded (checkpoint: {ad.checkpoint})', flush=True)

    y_embed = ad.embed(master)
    write_wav(os.path.join(OUT, 'aura_wm_master_22k.wav'), y_embed, SR_MASTER)
    conf, bits, acc = ad.detect(y_embed)
    print(f'   clean  AURA  bit_acc={round(acc, 3)}  bits={bits}', flush=True)

    attacks = list(V.VOX_GRID) if args.attacks == 'all' else \
              [a for a in args.attacks.split(',') if a in V.VOX_GRID]

    if args.selftest:
        print('=== AURA VOX SELFTEST ===', flush=True)
        for name in attacks:
            label, param = V.VOX_GRID[name][0]
            y2 = V.apply(name, param, y_embed, SR_MASTER)
            if y2 is None:
                print(f'  {name:20s} SKIP (dep/noise missing)', flush=True); continue
            c, b, a = ad.detect(y2)
            print(f'  {name:20s} OK   ({label}) acc={round(a,3)}', flush=True)
        return

    # resume support
    done = set()
    if os.path.exists(CSV):
        with open(CSV) as f:
            for r in csv.DictReader(f):
                done.add((r['target'], r['attack']))
    new = (not os.path.exists(CSV)) or os.path.getsize(CSV) == 0
    fh = open(CSV, 'a', newline=''); w = csv.writer(fh)
    if new:
        w.writerow(HEADER); fh.flush()

    for attack in attacks:
        if ('aura', attack) in done:
            print(f'   skip {attack} (already done)', flush=True); continue
        for label, param in V.VOX_GRID[attack]:
            try:
                y2 = V.apply(attack, param, y_embed, SR_MASTER)
            except Exception as e:
                print(f'   {attack:18s} {label:8s} ERROR {str(e)[:50]}', flush=True); continue
            if y2 is None:
                w.writerow(['aura', attack, label, V.strength_x(attack, param), 'ALL',
                            '', '', '', 'SKIP'])
                fh.flush(); continue
            conf, bits, acc = ad.detect(y2)
            ber = round(1 - acc, 3)
            det = int(acc >= 0.8)
            w.writerow(['aura', attack, label, V.strength_x(attack, param), NAME,
                        round(acc, 3), ber, round(conf, 3), det])
            fh.flush()
            print(f'   {attack:18s} {label:8s} | AURA={round(acc, 2)}', flush=True)

    fh.close()
    print('\nAURA DONE ->', CSV, '\nNow run:  python compare_vox.py', flush=True)


if __name__ == '__main__':
    main()
