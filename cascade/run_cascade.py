"""
run_cascade.py  --  multi-watermark cascade benchmark orchestrator.

Workflow
--------
ONE clean input audio -> for every ordering of {AudioSeal, AWARE, Timbre} of
length 1, 2 and 3 (15 configs total) we:

  1. CASCADE EMBED, stage by stage. After each embed step, run PROGRESSIVE
     DETECTION of every watermark embedded SO FAR on the clean (un-attacked)
     intermediate. This isolates *embedding interference* -- e.g. "how much did
     adding AWARE on top damage the AudioSeal watermark" -- with no attack noise.

  2. ATTACK ROBUSTNESS. The final cascaded file is rendered to SR_MASTER and run
     through the full attack suite; for each attack we detect every watermark
     present in that config and record bit-accuracy / confidence / quality.

Controls
--------
  * clean_negative : the raw clean master -> all 3 detectors (should NOT fire;
    measures false-positive / cross-talk floor).
  * resample_only  : clean master -> 16k -> 22.05k round trip -> all detectors +
    quality. Isolates the cost of the inter-stage resampling the cascade performs,
    so interference numbers can be read net of resampling.

Outputs (in BASE/cascade_out/)
  cascade_interference.csv   progressive (clean) detections, per config/stage
  cascade_robustness.csv     attack x watermark detections, per config
  cascade_quality.csv        cumulative audio quality after each embed stage
  cascade_controls.csv       negative + resample-only controls
  wavs/<config>/...          intermediate + final + attacked wavs

Usage
  python run_cascade.py --selftest          # quick adapter sanity check (seconds)
  python run_cascade.py                      # full run (singles+pairs+triples)
  python run_cascade.py --sizes 3            # only the 6 triples
  python run_cascade.py --src audio/foo.wav  # choose the single input audio
"""

import os, sys, csv, json, argparse, itertools, traceback
import numpy as np

import cascade_lib as L
from cascade_lib import (BASE, AUDIO, SR_MASTER, SR_16K, TOOLS, TOOL_NAME,
                         get_adapter, read_wav, write_wav, resample,
                         build_attacks, ATTACK_ORDER, quality_metrics)

OUT   = os.path.join(BASE, 'cascade_out')
WAVES = os.path.join(OUT, 'wavs')

# default single input audio (clean, un-watermarked)
DEFAULT_SRC = os.path.join(AUDIO, 'client_original_16k.wav')


def make_master(src):
    """Build the canonical clean master at SR_MASTER and a clean 16k reference."""
    os.makedirs(OUT, exist_ok=True)
    y = read_wav(src, SR_MASTER)
    master22 = write_wav(os.path.join(OUT, 'clean_master_22k.wav'), y, SR_MASTER)
    y16 = resample(y, SR_MASTER, SR_16K)
    master16 = write_wav(os.path.join(OUT, 'clean_master_16k.wav'), y16, SR_16K)
    return master22, master16


def configs_for(sizes):
    """Ordered tool sequences: permutations of TOOLS for each requested length."""
    out = []
    for k in sizes:
        out += [list(p) for p in itertools.permutations(TOOLS, k)]
    return out

def cfg_id(seq):
    return '_'.join(s[:2].upper() for s in seq)   # e.g. ['audioseal','aware'] -> 'AU_AW'


# --------------------------------------------------------------------------- #
#  Self-test : solo embed + detect each tool on the master. Fails fast.
# --------------------------------------------------------------------------- #
def selftest(master22):
    y = read_wav(master22, SR_MASTER)
    print('=== SELFTEST (solo embed -> detect round trip) ===', flush=True)
    ok = True
    for code in TOOLS:
        try:
            a = get_adapter(code)
            yw = a.embed(y)
            conf, bits, acc = a.detect(yw)
            status = 'OK ' if acc >= 0.99 else ('weak' if acc >= 0.8 else 'FAIL')
            if acc < 0.8: ok = False
            print(f'  {TOOL_NAME[code]:10s} truth={a.truth}  decoded={bits}  '
                  f'bit_acc={acc:.3f}  conf={conf:.3f}  [{status}]', flush=True)
        except Exception as e:
            ok = False
            print(f'  {TOOL_NAME[code]:10s} ERROR: {e}', flush=True)
            traceback.print_exc()
    print('=== SELFTEST', 'PASSED' if ok else 'FAILED', '===', flush=True)
    return ok


# --------------------------------------------------------------------------- #
#  Controls
# --------------------------------------------------------------------------- #
def run_controls(master22, master16):
    rows = []
    y = read_wav(master22, SR_MASTER)
    # negative: clean master, all detectors
    for code in TOOLS:
        try:
            conf, bits, acc = get_adapter(code).detect(y)
        except Exception as e:
            conf, bits, acc = 'ERR', '', str(e)[:40]
        rows.append(['clean_negative', TOOL_NAME[code], conf, acc, bits])
    # resample-only: 22k -> 16k -> 22k round trip
    y_rt = resample(resample(y, SR_MASTER, SR_16K), SR_16K, SR_MASTER)
    rt_path = write_wav(os.path.join(OUT, 'resample_only_22k.wav'), y_rt, SR_MASTER)
    for code in TOOLS:
        try:
            conf, bits, acc = get_adapter(code).detect(y_rt)
        except Exception as e:
            conf, bits, acc = 'ERR', '', str(e)[:40]
        rows.append(['resample_only', TOOL_NAME[code], conf, acc, bits])
    q = quality_metrics(master16, rt_path)
    rows.append(['resample_only_quality', '-', q['pesq'], q['stoi'],
                 f"snr={q['snr_db']} sisnr={q['si_snr_db']}"])
    with open(os.path.join(OUT, 'cascade_controls.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['control', 'watermark', 'conf', 'bit_acc', 'bits']); w.writerows(rows)
    print('controls done ->', os.path.join(OUT, 'cascade_controls.csv'), flush=True)


# --------------------------------------------------------------------------- #
#  One cascade config
# --------------------------------------------------------------------------- #
def run_config(seq, master22, master16, attacks, inter_rows, qual_rows, robust_rows):
    cid = cfg_id(seq)
    wdir = os.path.join(WAVES, cid); os.makedirs(wdir, exist_ok=True)
    print(f'\n#### CONFIG {cid}  ({" -> ".join(TOOL_NAME[s] for s in seq)})', flush=True)

    cur = read_wav(master22, SR_MASTER)          # current cascaded audio @22k
    embedded_so_far = []

    for stage, code in enumerate(seq, 1):
        a = get_adapter(code)
        cur = a.embed(cur)                       # embed this watermark on top
        embedded_so_far.append(code)
        stage_path = write_wav(os.path.join(wdir, f'stage{stage}_{code}.wav'), cur, SR_MASTER)

        # ---- progressive detection of ALL watermarks embedded so far ----
        for wcode in embedded_so_far:
            try:
                conf, bits, acc = get_adapter(wcode).detect(cur)
            except Exception as e:
                conf, bits, acc = 'ERR', '', str(e)[:40]
            inter_rows.append([cid, stage, code, TOOL_NAME[wcode], conf, acc])
            print(f'   stage{stage} after +{TOOL_NAME[code]:9s} | detect {TOOL_NAME[wcode]:9s}'
                  f' -> bit_acc={acc if isinstance(acc,str) else round(acc,3)} conf='
                  f'{conf if isinstance(conf,str) else round(conf,3)}', flush=True)

        # ---- cumulative audio quality after this stage ----
        q = quality_metrics(master16, stage_path)
        qual_rows.append([cid, stage, code, len(embedded_so_far),
                          q['pesq'], q['snr_db'], q['si_snr_db'], q['stoi']])

    final_path = os.path.join(wdir, 'final.wav'); write_wav(final_path, cur, SR_MASTER)

    # ---- attack robustness on the FINAL cascaded file ----
    for attack in ATTACK_ORDER:
        fn, do_q = attacks[attack]
        try:
            apath = fn(final_path, wdir)
        except Exception as e:
            for wcode in seq:
                robust_rows.append([cid, attack, TOOL_NAME[wcode], 'ERR', 'ERR', '', None])
            print(f'   attack {attack:14s} ERROR {str(e)[:50]}', flush=True); continue
        q = quality_metrics(master16, apath) if do_q else {'pesq': None}
        ya = read_wav(apath, SR_MASTER)
        line = f'   attack {attack:14s} |'
        for wcode in seq:                        # only watermarks present in this config
            try:
                conf, bits, acc = get_adapter(wcode).detect(ya)
            except Exception as e:
                conf, bits, acc = 'ERR', '', str(e)[:40]
            robust_rows.append([cid, attack, TOOL_NAME[wcode], conf, acc, bits, q.get('pesq')])
            line += f' {wcode[:2].upper()}={acc if isinstance(acc,str) else round(acc,2)}'
        print(line + (f'  pesq={q.get("pesq")}' if do_q else ''), flush=True)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=DEFAULT_SRC, help='single clean input audio')
    ap.add_argument('--sizes', default='1,2,3', help='cascade lengths, e.g. "3" or "1,2,3"')
    ap.add_argument('--selftest', action='store_true', help='solo round-trip check then exit')
    args = ap.parse_args()

    src = args.src if os.path.isabs(args.src) else os.path.join(BASE, args.src)
    os.makedirs(WAVES, exist_ok=True)
    master22, master16 = make_master(src)
    print(f'master: {src}  ->  {master22} (22k) / {master16} (16k)', flush=True)

    if args.selftest:
        sys.exit(0 if selftest(master22) else 1)

    sizes = [int(x) for x in args.sizes.split(',')]
    cfgs  = configs_for(sizes)
    attacks = build_attacks(SR_MASTER)

    # ---- CSV files written INCREMENTALLY (one append per finished config) so a
    #      timeout/crash never loses completed work, and a re-run RESUMES. ----
    files = {
        'inter':  (os.path.join(OUT, 'cascade_interference.csv'),
                   ['config','stage','embedded','detected_wm','conf','bit_acc']),
        'qual':   (os.path.join(OUT, 'cascade_quality.csv'),
                   ['config','stage','embedded','depth','pesq','snr_db','si_snr_db','stoi']),
        'robust': (os.path.join(OUT, 'cascade_robustness.csv'),
                   ['config','attack','watermark','conf','bit_acc','bits','pesq']),
    }
    def appender(key):
        path, header = files[key]
        new = (not os.path.exists(path)) or os.path.getsize(path) == 0
        f = open(path, 'a', newline=''); w = csv.writer(f)
        if new: w.writerow(header); f.flush()
        return f, w

    # a config counts as DONE if its last attack (pitch_up) is already in robustness CSV
    done = set()
    rpath = files['robust'][0]
    if os.path.exists(rpath):
        with open(rpath) as f:
            for r in csv.DictReader(f):
                if r.get('attack') == ATTACK_ORDER[-1]:
                    done.add(r['config'])
    if done:
        print(f'resuming -- already complete: {sorted(done)}', flush=True)

    run_controls(master22, master16)

    fi, wi = appender('inter'); fq, wq = appender('qual'); fr, wr = appender('robust')
    todo = [s for s in cfgs if cfg_id(s) not in done]
    print(f'{len(cfgs)} configs x {len(ATTACK_ORDER)} attacks  ({len(todo)} to run)', flush=True)
    try:
        for seq in todo:
            ir, qr, rr = [], [], []
            try:
                run_config(seq, master22, master16, attacks, ir, qr, rr)
            except Exception as e:
                print(f'CONFIG {cfg_id(seq)} FATAL: {e}', flush=True); traceback.print_exc()
            wi.writerows(ir); wq.writerows(qr); wr.writerows(rr)   # persist after EACH config
            fi.flush(); fq.flush(); fr.flush()
    finally:
        fi.close(); fq.close(); fr.close()
    print('\nALL DONE. Now run:  python make_report.py', flush=True)


if __name__ == '__main__':
    main()
