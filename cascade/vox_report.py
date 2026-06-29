"""
vox_report.py  --  visualizations for the VoxWatermark sweep (run_vox.py output).

Reads BASE/vox_out/vox_results.csv and writes:

  figs/curve_<attack>.png   robustness-vs-strength: x = strength, y = bit_acc,
                            one line per watermark, SOLID = solo file,
                            DASHED = same watermark inside the combined file.
                            (shows whether being stacked changes robustness)
  figs/heatmap_solo.png     attack:strength (rows) x watermark (cols) bit_acc grid
  figs/heatmap_combined.png same, for the combined file (3 watermarks)
  vox_report.html           everything assembled

Headless (Agg). bit_acc >= 0.8 = detected (dashed gray line on curves).
"""
import os, csv
from collections import defaultdict, OrderedDict
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.expanduser('~/wm_compare')
OUT  = os.path.join(BASE, 'vox_out')
FIGS = os.path.join(OUT, 'figs'); os.makedirs(FIGS, exist_ok=True)
WMS  = ['AudioSeal', 'AWARE', 'Timbre']
WCOL = {'AudioSeal': '#2a78d6', 'AWARE': '#1baf7a', 'Timbre': '#eda100'}

def load():
    p = os.path.join(OUT, 'vox_results.csv')
    rows = list(csv.DictReader(open(p))) if os.path.exists(p) else []
    for r in rows:
        try: r['x'] = float(r['strength_x'])
        except: r['x'] = 0.0
        try: r['acc'] = float(r['bit_acc'])
        except: r['acc'] = np.nan
    return rows


def fig_curves(rows):
    attacks = list(OrderedDict.fromkeys(r['attack'] for r in rows))
    made = []
    for atk in attacks:
        sub = [r for r in rows if r['attack'] == atk and not np.isnan(r['acc'])]
        if not sub:
            continue
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        for wm in WMS:
            for target, style, mk in [(wm.lower(), '-', 'o'), ('combined', '--', 's')]:
                pts = sorted([(r['x'], r['acc']) for r in sub
                              if r['watermark'] == wm and r['target'] == target])
                if not pts:
                    continue
                xs, ys = zip(*pts)
                lbl = f'{wm} ' + ('solo' if style == '-' else 'in combo')
                ax.plot(xs, ys, style, marker=mk, ms=4, color=WCOL[wm], label=lbl, lw=1.6)
        ax.axhline(0.8, ls=':', c='gray', lw=1)
        ax.set_ylim(-0.02, 1.05); ax.set_xlabel('strength'); ax.set_ylabel('bit_acc')
        ax.set_title(f'{atk}  (solid=solo, dashed=combined)', fontsize=10)
        ax.legend(fontsize=7, ncol=2)
        p = os.path.join(FIGS, f'curve_{atk}.png')
        fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig); made.append((atk, p))
    return made


def fig_heatmap(rows, target, fname, title):
    sub = [r for r in rows if r['target'] == target]
    conds = list(OrderedDict.fromkeys(f"{r['attack']}:{r['strength_label']}" for r in sub))
    wms = [w for w in WMS if any(r['watermark'] == w for r in sub)]
    if not conds or not wms:
        return None
    M = np.full((len(conds), len(wms)), np.nan)
    for r in sub:
        if r['watermark'] in wms:
            c = conds.index(f"{r['attack']}:{r['strength_label']}")
            M[c, wms.index(r['watermark'])] = r['acc']
    fig, ax = plt.subplots(figsize=(max(4, 1.3*len(wms)+2), max(6, 0.22*len(conds)+1)))
    im = ax.imshow(M, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(wms))); ax.set_xticklabels(wms, fontsize=8)
    ax.set_yticks(range(len(conds))); ax.set_yticklabels(conds, fontsize=6)
    for i in range(len(conds)):
        for j in range(len(wms)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f'{M[i,j]:.2f}', ha='center', va='center', fontsize=5.5)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    p = os.path.join(FIGS, fname)
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); return p


def img(p, w=560):
    import base64
    return f'<img src="data:image/png;base64,{base64.b64encode(open(p,"rb").read()).decode()}" width="{w}">'


def main():
    rows = load()
    if not rows:
        print('no vox_results.csv found'); return
    curves = fig_curves(rows)
    hsolo  = fig_heatmap(rows, 'audioseal', 'heatmap_AU.png', 'AudioSeal solo')
    # combined heatmap (all 3 watermarks in one file)
    hcomb  = fig_heatmap(rows, 'combined', 'heatmap_combined.png', 'Combined file (3 watermarks)')

    parts = ['<html><head><meta charset="utf-8"><title>VoxWatermark sweep</title>',
             '<style>body{font-family:system-ui,Arial;margin:24px;max-width:1100px}'
             'h2{border-bottom:2px solid #333;padding-bottom:4px;margin-top:30px}'
             '.grid{display:flex;flex-wrap:wrap;gap:14px}</style></head><body>']
    parts.append('<h1>VoxWatermark no-box robustness sweep</h1>')
    parts.append('<p>17 perturbation families at swept strengths, on AudioSeal / AWARE / '
                 'Timbre solo and on the combined 3-watermark file. bit_acc &ge; 0.8 = detected. '
                 'On the curves: <b>solid = watermark alone</b>, <b>dashed = same watermark inside '
                 'the combined file</b> &mdash; overlapping lines mean stacking didn\'t change robustness.</p>')
    if hcomb:
        parts.append('<h2>Combined file — full grid</h2>' + img(hcomb, 520))
    parts.append('<h2>Robustness vs. strength (per attack)</h2><div class="grid">')
    for atk, p in curves:
        parts.append(f'<div>{img(p, 430)}</div>')
    parts.append('</div>')
    out = os.path.join(OUT, 'vox_report.html')
    open(out, 'w', encoding='utf-8').write('\n'.join(parts))
    print('wrote', out); print('figures in', FIGS)


if __name__ == '__main__':
    main()
