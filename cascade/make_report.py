"""
make_report.py  --  turn the cascade CSVs into visualizations + an HTML report.

Reads BASE/cascade_out/*.csv (produced by run_cascade.py) and writes:

  figs/interference_<config>.png   triangular heatmap: as you stack watermarks,
                                    how each earlier watermark's bit_acc decays
  figs/robust_<config>.png         attack x watermark bit_acc heatmap (per config)
  figs/order_effect.png            for each watermark, final solo-vs-stacked bit_acc
                                    across every ordering it appears in
  figs/quality_vs_depth.png        PESQ / STOI as a function of stack depth
  report.html                      everything assembled with the data tables

Headless-safe (Agg backend) -- meant to run on the cluster, then you scp the
cascade_out/ folder back. No interactivity required.
"""

import os, csv, base64, html
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.expanduser('~/wm_compare')
OUT  = os.path.join(BASE, 'cascade_out')
FIGS = os.path.join(OUT, 'figs'); os.makedirs(FIGS, exist_ok=True)

WM_ORDER = ['AudioSeal', 'AWARE', 'Timbre']


def load_csv(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))

def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


# --------------------------------------------------------------------------- #
#  Heatmap helper
# --------------------------------------------------------------------------- #
def heatmap(matrix, rows, cols, title, path, fmt='{:.2f}', cmap='RdYlGn',
            vmin=0.0, vmax=1.0, xlabel='', ylabel=''):
    fig, ax = plt.subplots(figsize=(max(5, 0.7*len(cols)+2), max(3, 0.5*len(rows)+1.5)))
    M = np.array(matrix, dtype=float)
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, fmt.format(v), ha='center', va='center',
                        fontsize=7, color='black')
    ax.set_title(title, fontsize=10); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# --------------------------------------------------------------------------- #
#  1. Interference heatmaps (one per config)
# --------------------------------------------------------------------------- #
def fig_interference(inter):
    by_cfg = defaultdict(list)
    for r in inter:
        by_cfg[r['config']].append(r)
    made = []
    for cid, rows in by_cfg.items():
        stages = sorted({int(r['stage']) for r in rows})
        wms = []
        for r in rows:
            if r['detected_wm'] not in wms:
                wms.append(r['detected_wm'])
        M = np.full((len(stages), len(wms)), np.nan)
        rlabel = []
        for si, s in enumerate(stages):
            emb = next(r['embedded'] for r in rows if int(r['stage']) == s)
            rlabel.append(f'stage{s}: +{emb}')
            for r in rows:
                if int(r['stage']) == s:
                    M[si, wms.index(r['detected_wm'])] = fnum(r['bit_acc'])
        p = os.path.join(FIGS, f'interference_{cid}.png')
        heatmap(M, rlabel, wms, f'Interference  {cid}  (bit_acc on clean audio)',
                p, xlabel='detected watermark', ylabel='after embedding stage')
        made.append((cid, p))
    return made


# --------------------------------------------------------------------------- #
#  2. Robustness heatmaps (one per config)
# --------------------------------------------------------------------------- #
def fig_robustness(robust):
    by_cfg = defaultdict(list)
    for r in robust:
        by_cfg[r['config']].append(r)
    made = []
    for cid, rows in by_cfg.items():
        attacks = []
        for r in rows:
            if r['attack'] not in attacks:
                attacks.append(r['attack'])
        wms = []
        for r in rows:
            if r['watermark'] not in wms:
                wms.append(r['watermark'])
        M = np.full((len(attacks), len(wms)), np.nan)
        for r in rows:
            M[attacks.index(r['attack']), wms.index(r['watermark'])] = fnum(r['bit_acc'])
        p = os.path.join(FIGS, f'robust_{cid}.png')
        heatmap(M, attacks, wms, f'Attack robustness  {cid}  (bit_acc)',
                p, xlabel='watermark', ylabel='attack')
        made.append((cid, p))
    return made


# --------------------------------------------------------------------------- #
#  3. Order effect : final-stage bit_acc of each watermark across all orderings
# --------------------------------------------------------------------------- #
def fig_order_effect(inter):
    # final-stage detection per config = last stage rows
    by_cfg = defaultdict(list)
    for r in inter:
        by_cfg[r['config']].append(r)
    # for each watermark, collect (config, final bit_acc)
    series = defaultdict(list)
    for cid, rows in by_cfg.items():
        last = max(int(r['stage']) for r in rows)
        depth = last
        for r in rows:
            if int(r['stage']) == last:
                series[r['detected_wm']].append((cid, depth, fnum(r['bit_acc'])))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    width = 0.25
    all_cfgs = sorted({c for v in series.values() for c, _, _ in v})
    x = np.arange(len(all_cfgs))
    for i, wm in enumerate([w for w in WM_ORDER if w in series]):
        vals = {c: a for c, d, a in series[wm]}
        ys = [vals.get(c, np.nan) for c in all_cfgs]
        ax.bar(x + i*width, ys, width, label=wm)
    ax.set_xticks(x + width); ax.set_xticklabels(all_cfgs, rotation=60, ha='right', fontsize=7)
    ax.set_ylabel('final bit_acc (clean)'); ax.set_ylim(0, 1.05)
    ax.axhline(0.8, ls='--', c='gray', lw=0.8, label='detect thresh 0.8')
    ax.set_title('Order effect: each watermark\'s survival vs. who was stacked on top of it')
    ax.legend(fontsize=8, ncol=4)
    p = os.path.join(FIGS, 'order_effect.png')
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
#  4. Quality vs stack depth
# --------------------------------------------------------------------------- #
def fig_quality(qual):
    by_depth = defaultdict(lambda: defaultdict(list))
    for r in qual:
        d = int(r['depth'])
        by_depth[d]['pesq'].append(fnum(r['pesq']))
        by_depth[d]['stoi'].append(fnum(r['stoi']))
        by_depth[d]['si_snr_db'].append(fnum(r['si_snr_db']))
    depths = sorted(by_depth)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    pesq = [np.nanmean(by_depth[d]['pesq']) for d in depths]
    stoi = [np.nanmean(by_depth[d]['stoi']) for d in depths]
    ax1.plot(depths, pesq, 'o-', color='tab:blue', label='PESQ (mean)')
    ax1.set_xlabel('stack depth (# watermarks embedded)')
    ax1.set_ylabel('PESQ', color='tab:blue'); ax1.set_xticks(depths)
    ax2 = ax1.twinx()
    ax2.plot(depths, stoi, 's--', color='tab:red', label='STOI (mean)')
    ax2.set_ylabel('STOI', color='tab:red')
    ax1.set_title('Audio quality degradation vs. number of stacked watermarks')
    p = os.path.join(FIGS, 'quality_vs_depth.png')
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
#  HTML assembly
# --------------------------------------------------------------------------- #
def img_tag(path, width=560):
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'<img src="data:image/png;base64,{b64}" width="{width}">'

def table(rows, cols):
    if not rows:
        return '<p><i>no data</i></p>'
    th = ''.join(f'<th>{html.escape(c)}</th>' for c in cols)
    body = ''
    for r in rows:
        body += '<tr>' + ''.join(f'<td>{html.escape(str(r.get(c,"")))}</td>' for c in cols) + '</tr>'
    return f'<table border=1 cellspacing=0 cellpadding=3 style="border-collapse:collapse;font-size:12px">' \
           f'<tr>{th}</tr>{body}</table>'


def main():
    inter   = load_csv('cascade_interference.csv')
    robust  = load_csv('cascade_robustness.csv')
    qual    = load_csv('cascade_quality.csv')
    ctrl    = load_csv('cascade_controls.csv')

    inter_figs  = fig_interference(inter)  if inter  else []
    robust_figs = fig_robustness(robust)   if robust else []
    order_fig   = fig_order_effect(inter)  if inter  else None
    qual_fig    = fig_quality(qual)        if qual   else None

    parts = ['<html><head><meta charset="utf-8"><title>WM Cascade Report</title>',
             '<style>body{font-family:system-ui,Arial;margin:24px;max-width:1100px}'
             'h2{border-bottom:2px solid #333;padding-bottom:4px;margin-top:32px}'
             'h3{color:#444} .grid{display:flex;flex-wrap:wrap;gap:16px}</style></head><body>']
    parts.append('<h1>Multi-watermark cascade benchmark</h1>')
    parts.append('<p>AudioSeal + AWARE + Timbre embedded in sequence on one audio. '
                 'Singles, ordered pairs, and ordered triples (15 configs). '
                 '<b>bit_acc &ge; 0.8 = detected.</b></p>')

    parts.append('<h2>Controls</h2>')
    parts.append(table(ctrl, ['control','watermark','conf','bit_acc','bits']))
    parts.append('<p><i>clean_negative should be ~random (no watermark present); '
                 'resample_only shows the cost of inter-stage 22k&harr;16k resampling alone.</i></p>')

    if order_fig:
        parts.append('<h2>Order effect (headline result)</h2>')
        parts.append(img_tag(order_fig, 900))
        parts.append('<p>Each watermark\'s bit-accuracy in the <b>final</b> cascaded file, '
                     'across every ordering. Lower bars = that watermark got damaged by '
                     'whatever was stacked on top of it. The earliest-embedded watermark '
                     'usually suffers most.</p>')

    if qual_fig:
        parts.append('<h2>Audio quality vs stack depth</h2>')
        parts.append(img_tag(qual_fig, 640))

    if inter_figs:
        parts.append('<h2>Interference matrices (per config)</h2><div class="grid">')
        for cid, p in inter_figs:
            parts.append(f'<div>{img_tag(p)}</div>')
        parts.append('</div>')

    if robust_figs:
        parts.append('<h2>Attack robustness (per config)</h2><div class="grid">')
        for cid, p in robust_figs:
            parts.append(f'<div>{img_tag(p)}</div>')
        parts.append('</div>')

    parts.append('<h2>Raw: interference</h2>')
    parts.append(table(inter, ['config','stage','embedded','detected_wm','conf','bit_acc']))
    parts.append('<h2>Raw: robustness</h2>')
    parts.append(table(robust, ['config','attack','watermark','conf','bit_acc','pesq']))
    parts.append('<h2>Raw: quality</h2>')
    parts.append(table(qual, ['config','stage','embedded','depth','pesq','snr_db','si_snr_db','stoi']))

    parts.append('</body></html>')
    out = os.path.join(OUT, 'report.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(parts))
    print('wrote', out)
    print('figures in', FIGS)


if __name__ == '__main__':
    main()
