"""
vox_heatmap_aura.py  --  reproduce the "Combined file (3 watermarks)" heatmap
with AURA added as a 4th column.

Rows  = attack:strength (taken from the combined-file run, so the layout matches
        the existing figure exactly).
Cols  = AudioSeal, AWARE, Timbre  (from vox_out/vox_results.csv, target='combined')
        + AURA                     (from vox_out/vox_results_aura.csv, target='aura')
Values = bit_acc, RdYlGn 0..1, annotated — same style as vox_report.fig_heatmap.

Output: vox_out/figs/heatmap_combined_aura.png

Run on Great Lakes after run_vox_aura.py:
    python vox_heatmap_aura.py
"""
import os, csv
from collections import OrderedDict
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.expanduser('~/wm_compare')
OUT  = os.path.join(BASE, 'vox_out')
FIGS = os.path.join(OUT, 'figs'); os.makedirs(FIGS, exist_ok=True)
MAIN = os.path.join(OUT, 'vox_results.csv')
AURA = os.path.join(OUT, 'vox_results_aura.csv')

COLS = ['AudioSeal', 'AWARE', 'Timbre', 'AURA']


def _load(path):
    if not os.path.exists(path):
        return []
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        try: r['acc'] = float(r['bit_acc'])
        except Exception: r['acc'] = np.nan
    return rows


def main():
    main_rows = _load(MAIN)
    aura_rows = _load(AURA)
    if not main_rows:
        raise SystemExit('missing vox_results.csv — run run_vox.py first')
    if not aura_rows:
        raise SystemExit('missing vox_results_aura.csv — run run_vox_aura.py first')

    # row order from the combined-file run (matches the existing figure)
    comb = [r for r in main_rows if r['target'] == 'combined']
    conds = list(OrderedDict.fromkeys(
        f"{r['attack']}:{r['strength_label']}" for r in comb))
    # AURA may add conds the combined run lacks (or vice-versa); append any extras
    for r in aura_rows:
        key = f"{r['attack']}:{r['strength_label']}"
        if key not in conds:
            conds.append(key)

    M = np.full((len(conds), len(COLS)), np.nan)

    def put(rows, wm_name, col):
        for r in rows:
            if r['watermark'] != wm_name:
                continue
            key = f"{r['attack']}:{r['strength_label']}"
            if key in conds:
                M[conds.index(key), col] = r['acc']

    put(comb, 'AudioSeal', 0)
    put(comb, 'AWARE', 1)
    put(comb, 'Timbre', 2)
    put(aura_rows, 'AURA', 3)

    fig, ax = plt.subplots(figsize=(1.3 * len(COLS) + 2, max(6, 0.22 * len(conds) + 1)))
    im = ax.imshow(M, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(COLS))); ax.set_xticklabels(COLS, fontsize=8)
    ax.set_yticks(range(len(conds))); ax.set_yticklabels(conds, fontsize=6)
    for i in range(len(conds)):
        for j in range(len(COLS)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=5.5)
    # thin separator before the AURA column to set it apart
    ax.axvline(2.5, color='black', lw=1.2)
    ax.set_title('Combined file + AURA (4 watermarks)', fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    p = os.path.join(FIGS, 'heatmap_combined_aura.png')
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    print('wrote', p)


if __name__ == '__main__':
    main()
