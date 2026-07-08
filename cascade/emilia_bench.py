"""
emilia_bench.py  --  4-watermark evaluation on a frozen Emilia set.

Tests AudioSeal / AWARE / Timbre / AURA on the SAME 50 Emilia clips (trimmed to a
fixed length), each with K random messages, under the VoxWatermark attack grid, and
reports per-cell bit-accuracy + detection-rate with Wilson 95% CIs, plus false-positive
rate on clean clips and McNemar pairwise model comparisons.

Design (paired, statistically honest with few clips):
  * ONE frozen clip set used identically by all models  -> emilia_eval50.csv
  * K messages/clip from one 32-bit seed per (clip,msg), sliced to each model's
    native payload (AudioSeal 16 / AWARE 20 / Timbre 10 / AURA 32) -> paired trials
  * embeds cached to disk (AWARE embedding is the slow step; never recomputed)
  * N_clips x K messages trials per (model, attack) cell -> Wilson 95% CI

Subcommands (each resumable):
  python emilia_bench.py select                       # freeze the 50-clip set + trials
  python emilia_bench.py embed  [--models a,b,c,d]    # embed + cache wm wavs (+ quality)
  python emilia_bench.py attack [--models ...] [--attacks all|enc,mp3]
  python emilia_bench.py fpr    [--models ...]        # detect on clean (unwatermarked)
  python emilia_bench.py aggregate                    # summary CSVs + Wilson CI + McNemar
  python emilia_bench.py all                          # select->embed->attack->fpr->aggregate

Env: WM_COMPARE_BASE (checkout), AURA_CKPT (AURA checkpoint), EMILIA_CSV (override manifest).
"""
import os, sys, csv, json, argparse, random
import numpy as np

import cascade_lib as L
from cascade_lib import BASE, SR_MASTER, read_wav, write_wav, quality_metrics, get_adapter
import vox_attacks as V
from aura_adapter import AuraAdapter

# --------------------------------------------------------------------------- #
#  Config / paths
# --------------------------------------------------------------------------- #
# Default to AURA's speaker-disjoint held-out Emilia val split (no leakage for AURA).
# Override with $EMILIA_CSV. Note: val.csv mixes Emilia+FMA; stage_select filters to Emilia.
EMILIA_CSV = os.environ.get(
    'EMILIA_CSV', os.path.expanduser('~/projects/aura_watermark/data/val.csv'))
OUT      = os.path.join(BASE, 'emilia_out')
EMB_DIR  = os.path.join(OUT, 'embeds')
EVAL_CSV = os.path.join(OUT, 'emilia_eval50.csv')
TRIALS   = os.path.join(OUT, 'trials.csv')
GTRUTH   = os.path.join(OUT, 'groundtruth_messages.csv')
EMB_LOG  = os.path.join(OUT, 'embed_quality.csv')
RESULTS  = os.path.join(OUT, 'results_long.csv')
FPR_CSV  = os.path.join(OUT, 'fpr.csv')

MODELS   = ['audioseal', 'aware', 'timbre', 'aura']
N_CLIPS  = 50
N_MSGS   = 3
SECONDS  = 10.0                 # fixed clip length (decision: trim to 10 s)
DNSMOS_MIN = 3.0
SEED     = 1234

# --------------------------------------------------------------------------- #
#  Adapter registry (cascade_lib has 3; add AURA) + per-trial message setting
# --------------------------------------------------------------------------- #
_AURA = None
def adapter(code):
    global _AURA
    if code == 'aura':
        if _AURA is None:
            _AURA = AuraAdapter(); _AURA.load()
        return _AURA
    return get_adapter(code)

def payload_len(ad):
    return ad.NB if getattr(ad, 'code', '') == 'timbre' else len(ad.truth)

def set_message(ad, bits):
    """Point an adapter at a new ground-truth message (adapters hard-code one)."""
    ad.truth = bits
    if ad.code == 'timbre':
        ad._wm = [int(b) for b in bits]; ad.NB = len(ad._wm)
    elif ad.code == 'aura':
        import torch
        ad._msg = torch.tensor([int(b) for b in bits], dtype=torch.long)
    # audioseal / aware read self.truth on every embed()/detect() -> nothing else

def message_for(clip_id, msg_idx, length):
    """Deterministic paired message: one 32-bit draw per (clip,msg), sliced to length."""
    rng = random.Random(f'{clip_id}|{msg_idx}|{SEED}')
    bits = ''.join(rng.choice('01') for _ in range(32))
    return bits[:length]

# --------------------------------------------------------------------------- #
#  Stage 0: select the frozen 50-clip set + trial list
# --------------------------------------------------------------------------- #
def stage_select():
    import pandas as pd
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(EMILIA_CSV)
    # AURA's val.csv mixes Emilia (speech) + FMA (music); keep only held-out Emilia.
    if 'dataset' in df.columns:
        df = df[df['dataset'] == 'emilia'].copy()
    df = df[(df['dnsmos'] >= DNSMOS_MIN) & (df['duration_s'] >= SECONDS)].copy()
    if len(df) < N_CLIPS:
        raise SystemExit(f'only {len(df)} clips pass filters; need {N_CLIPS}')
    # speaker-stratified: round-robin across speakers for max voice diversity
    rng = random.Random(SEED)
    by_spk = {}
    for _, r in df.sample(frac=1.0, random_state=SEED).iterrows():
        by_spk.setdefault(r['speaker'], []).append(r)
    order = list(by_spk); rng.shuffle(order)
    picked, i = [], 0
    while len(picked) < N_CLIPS and any(by_spk.values()):
        spk = order[i % len(order)]; i += 1
        if by_spk[spk]:
            picked.append(by_spk[spk].pop())
    sel = pd.DataFrame(picked).head(N_CLIPS).reset_index(drop=True)
    sel.insert(0, 'clip_id', [f'c{i:03d}' for i in range(len(sel))])
    sel.to_csv(EVAL_CSV, index=False)
    print(f'selected {len(sel)} clips, {sel["speaker"].nunique()} speakers -> {EVAL_CSV}')

    # payload lengths (load adapters once to read them)
    lengths = {}
    for m in MODELS:
        try:
            lengths[m] = payload_len(adapter(m))
        except Exception as e:
            print(f'  WARN could not load {m} for payload length ({str(e)[:50]}); using default')
            lengths[m] = {'audioseal': 16, 'aware': 20, 'timbre': 10, 'aura': 32}[m]
    print('payload lengths:', lengths)

    with open(TRIALS, 'w', newline='') as fh, open(GTRUTH, 'w', newline='') as gh:
        tw = csv.writer(fh); gw = csv.writer(gh)
        tw.writerow(['clip_id', 'clip_path', 'speaker', 'model', 'msg_idx', 'truth', 'wm_path'])
        gw.writerow(['clip_id', 'model', 'msg_idx', 'truth'])
        for _, r in sel.iterrows():
            for m in MODELS:
                for k in range(N_MSGS):
                    truth = message_for(r['clip_id'], k, lengths[m])
                    wm = os.path.join(EMB_DIR, m, f'{r["clip_id"]}__m{k}.wav')
                    tw.writerow([r['clip_id'], r['path'], r['speaker'], m, k, truth, wm])
                    gw.writerow([r['clip_id'], m, k, truth])
    print(f'wrote {N_CLIPS}x{len(MODELS)}x{N_MSGS} = {N_CLIPS*len(MODELS)*N_MSGS} trials -> {TRIALS}')

def _read_trials():
    with open(TRIALS) as f:
        return list(csv.DictReader(f))

def _clean_path(clip_id):
    return os.path.join(EMB_DIR, 'clean', f'{clip_id}.wav')

def _load_clip(path):
    y = read_wav(path, SR_MASTER)
    n = int(SECONDS * SR_MASTER)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y[:n].astype('float32')

# --------------------------------------------------------------------------- #
#  Stage 1: embed (cache wm wavs) + clean-detection + imperceptibility
# --------------------------------------------------------------------------- #
def stage_embed(models):
    os.makedirs(os.path.join(EMB_DIR, 'clean'), exist_ok=True)
    new = not os.path.exists(EMB_LOG)
    fh = open(EMB_LOG, 'a', newline=''); w = csv.writer(fh)
    if new:
        w.writerow(['clip_id', 'model', 'msg_idx', 'clean_bit_acc', 'pesq', 'snr_db', 'stoi'])
    for m in models:
        os.makedirs(os.path.join(EMB_DIR, m), exist_ok=True)
        ad = adapter(m)
        for t in _read_trials():
            if t['model'] != m:
                continue
            if os.path.exists(t['wm_path']):
                continue                                  # resume
            clip = _load_clip(t['clip_path'])
            cp = _clean_path(t['clip_id'])
            if not os.path.exists(cp):
                write_wav(cp, clip, SR_MASTER)
            set_message(ad, t['truth'])
            wm = ad.embed(clip)
            n = min(len(wm), len(clip)); wm = wm[:n]
            write_wav(t['wm_path'], wm, SR_MASTER)
            conf, bits, acc = ad.detect(wm)
            q = quality_metrics(cp, t['wm_path'])
            w.writerow([t['clip_id'], m, t['msg_idx'], round(acc, 3),
                        q['pesq'], q['snr_db'], q['stoi']]); fh.flush()
            print(f'  embed {m:9s} {t["clip_id"]} m{t["msg_idx"]}  clean_acc={acc:.2f} '
                  f'pesq={q["pesq"]}', flush=True)
    fh.close()

# --------------------------------------------------------------------------- #
#  Stage 2: attack + detect
# --------------------------------------------------------------------------- #
def stage_attack(models, attacks):
    grid = list(V.VOX_GRID) if attacks == 'all' else [a for a in attacks.split(',') if a in V.VOX_GRID]
    done = set()
    new = not os.path.exists(RESULTS)
    if not new:
        with open(RESULTS) as f:
            for r in csv.DictReader(f):
                done.add((r['clip_id'], r['model'], r['msg_idx'], r['attack'], r['strength_label']))
    fh = open(RESULTS, 'a', newline=''); w = csv.writer(fh)
    if new:
        w.writerow(['clip_id', 'speaker', 'model', 'msg_idx', 'attack', 'strength_label',
                    'strength_x', 'bit_acc', 'ber', 'conf', 'detected'])
    for m in models:
        ad = adapter(m)
        for t in _read_trials():
            if t['model'] != m or not os.path.exists(t['wm_path']):
                continue
            set_message(ad, t['truth'])
            wm = read_wav(t['wm_path'], SR_MASTER)
            for attack in grid:
                for label, param in V.VOX_GRID[attack]:
                    key = (t['clip_id'], m, t['msg_idx'], attack, label)
                    if key in done:
                        continue
                    try:
                        y2 = V.apply(attack, param, wm, SR_MASTER)
                    except Exception as e:
                        print(f'   {m} {t["clip_id"]} {attack} {label} ERR {str(e)[:40]}'); continue
                    if y2 is None:
                        w.writerow([t['clip_id'], t['speaker'], m, t['msg_idx'], attack, label,
                                    V.strength_x(attack, param), '', '', '', 'SKIP']); fh.flush(); continue
                    conf, bits, acc = ad.detect(y2)
                    w.writerow([t['clip_id'], t['speaker'], m, t['msg_idx'], attack, label,
                                V.strength_x(attack, param), round(acc, 3), round(1 - acc, 3),
                                round(conf, 3), int(acc >= 0.8)]); fh.flush()
            print(f'  attacked {m:9s} {t["clip_id"]} m{t["msg_idx"]}', flush=True)
    fh.close()

# --------------------------------------------------------------------------- #
#  Stage 3: false-positive rate (detect on clean, unwatermarked clips)
# --------------------------------------------------------------------------- #
def stage_fpr(models):
    new = not os.path.exists(FPR_CSV)
    fh = open(FPR_CSV, 'a', newline=''); w = csv.writer(fh)
    if new:
        w.writerow(['clip_id', 'model', 'bit_acc_vs_ref', 'false_positive'])
    seen = set()
    if not new:
        with open(FPR_CSV) as f:
            for r in csv.DictReader(f):
                seen.add((r['clip_id'], r['model']))
    for m in models:
        ad = adapter(m)
        for t in _read_trials():
            if t['model'] != m or t['msg_idx'] != '0':
                continue
            if (t['clip_id'], m) in seen:
                continue
            cp = _clean_path(t['clip_id'])
            if not os.path.exists(cp):
                write_wav(cp, _load_clip(t['clip_path']), SR_MASTER)
            set_message(ad, t['truth'])                  # reference message
            conf, bits, acc = ad.detect(read_wav(cp, SR_MASTER))
            w.writerow([t['clip_id'], m, round(acc, 3), int(acc >= 0.8)]); fh.flush()
    fh.close()
    print('FPR ->', FPR_CSV)

# --------------------------------------------------------------------------- #
#  Stage 4: aggregate  (Wilson 95% CI, detection rate, imperceptibility, McNemar)
# --------------------------------------------------------------------------- #
def _wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (round(p, 3), round(max(0, c - h), 3), round(min(1, c + h), 3))

def stage_aggregate():
    import pandas as pd
    df = pd.read_csv(RESULTS)
    df = df[df['detected'].isin(['0', '1', 0, 1])].copy()
    df['detected'] = df['detected'].astype(int)
    df['bit_acc'] = pd.to_numeric(df['bit_acc'], errors='coerce')

    rows = []
    for (m, atk, lab), g in df.groupby(['model', 'attack', 'strength_label']):
        p, lo, hi = _wilson(int(g['detected'].sum()), len(g))
        rows.append(dict(model=m, attack=atk, strength=lab, n=len(g),
                         mean_bit_acc=round(g['bit_acc'].mean(), 3),
                         det_rate=p, det_ci_lo=lo, det_ci_hi=hi))
    summ = pd.DataFrame(rows).sort_values(['attack', 'strength', 'model'])
    summ.to_csv(os.path.join(OUT, 'summary_by_attack.csv'), index=False)
    print('summary ->', os.path.join(OUT, 'summary_by_attack.csv'))

    # per-model overall detection rate with Wilson CI
    over = []
    for m, g in df.groupby('model'):
        p, lo, hi = _wilson(int(g['detected'].sum()), len(g))
        over.append(dict(model=m, n=len(g), det_rate=p, ci_lo=lo, ci_hi=hi,
                         mean_bit_acc=round(g['bit_acc'].mean(), 3)))
    pd.DataFrame(over).to_csv(os.path.join(OUT, 'summary_overall.csv'), index=False)
    print('overall  ->', os.path.join(OUT, 'summary_overall.csv'))
    print(pd.DataFrame(over).to_string(index=False))

    # imperceptibility summary
    if os.path.exists(EMB_LOG):
        q = pd.read_csv(EMB_LOG)
        imp = q.groupby('model')[['clean_bit_acc', 'pesq', 'snr_db', 'stoi']].mean().round(3)
        imp.to_csv(os.path.join(OUT, 'imperceptibility.csv'))
        print('\nimperceptibility (mean):'); print(imp.to_string())

    # FPR summary
    if os.path.exists(FPR_CSV):
        fp = pd.read_csv(FPR_CSV)
        fpr = fp.groupby('model')['false_positive'].mean().round(3)
        print('\nfalse-positive rate:'); print(fpr.to_string())

    # McNemar pairwise (overall paired detected vectors)
    try:
        from statsmodels.stats.contingency_tables import mcnemar
        piv = df.pivot_table(index=['clip_id', 'msg_idx', 'attack', 'strength_label'],
                             columns='model', values='detected')
        print('\nMcNemar pairwise (p-value; <0.05 = models differ):')
        ms = [c for c in MODELS if c in piv.columns]
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                a, b = piv[ms[i]], piv[ms[j]]
                ok = a.notna() & b.notna()
                t = [[((a == 1) & (b == 1) & ok).sum(), ((a == 1) & (b == 0) & ok).sum()],
                     [((a == 0) & (b == 1) & ok).sum(), ((a == 0) & (b == 0) & ok).sum()]]
                try:
                    pv = mcnemar(t, exact=False, correction=True).pvalue
                    print(f'  {ms[i]:9s} vs {ms[j]:9s}  p={pv:.4g}')
                except Exception as e:
                    print(f'  {ms[i]} vs {ms[j]}: {str(e)[:40]}')
    except ImportError:
        print('\n(statsmodels not installed -> skipping McNemar; pip install statsmodels)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['select', 'embed', 'attack', 'fpr', 'aggregate', 'all'])
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--attacks', default='all')
    args = ap.parse_args()
    models = [m for m in args.models.split(',') if m in MODELS]

    if args.stage in ('select', 'all'):
        stage_select()
    if args.stage in ('embed', 'all'):
        stage_embed(models)
    if args.stage in ('attack', 'all'):
        stage_attack(models, args.attacks)
    if args.stage in ('fpr', 'all'):
        stage_fpr(models)
    if args.stage in ('aggregate', 'all'):
        stage_aggregate()


if __name__ == '__main__':
    main()
