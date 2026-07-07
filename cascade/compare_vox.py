"""
compare_vox.py  --  merge AURA's VoxWatermark results with the existing
AudioSeal / AWARE / Timbre results and emit a side-by-side comparison.

Reads:
  vox_out/vox_results.csv        (AudioSeal, AWARE, Timbre, combined)
  vox_out/vox_results_aura.csv   (AURA)

Writes:
  vox_out/vox_comparison.csv     mean bit-accuracy per (model x attack)
  vox_out/vox_comparison.md      the same as a readable table + robustness rank

Bit-accuracy is averaged over each attack's strength sweep (SKIP/blank rows are
ignored). A model "survives" an attack at the same >=0.8 bit-acc bar used by the
benchmark's `detected` flag.
"""
import os, csv
from collections import defaultdict

HERE = os.path.dirname(__file__)
BASE = os.path.abspath(os.path.join(HERE, '..'))
OUT = os.path.join(BASE, 'vox_out')
MAIN = os.path.join(OUT, 'vox_results.csv')
AURA = os.path.join(OUT, 'vox_results_aura.csv')
DETECT_BAR = 0.8

# which watermark names to surface as columns, in display order
MODEL_ORDER = ['AudioSeal', 'AWARE', 'Timbre', 'AURA']


def _read(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    rows = _read(MAIN) + _read(AURA)
    if not rows:
        raise SystemExit('No results found. Run run_vox.py and run_vox_aura.py first.')

    # (model, attack) -> list of bit_acc across strengths.
    # Use SOLO targets only (skip the cascaded 'combined' target, whose rows carry
    # the same watermark names and would double-count).
    acc = defaultdict(list)
    attacks = []
    for r in rows:
        model = r['watermark']
        if r.get('target') == 'combined':
            continue
        if model in ('ALL', '') or r['bit_acc'] in ('', None):
            continue
        try:
            a = float(r['bit_acc'])
        except ValueError:
            continue
        acc[(model, r['attack'])].append(a)
        if r['attack'] not in attacks:
            attacks.append(r['attack'])

    models = [m for m in MODEL_ORDER if any((m, at) in acc for at in attacks)]

    # per-cell mean
    def cell(m, at):
        v = acc.get((m, at))
        return sum(v) / len(v) if v else None

    # ---- CSV ----
    with open(os.path.join(OUT, 'vox_comparison.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['attack'] + models)
        for at in attacks:
            w.writerow([at] + [('' if cell(m, at) is None else round(cell(m, at), 3))
                               for m in models])
        # overall mean row
        w.writerow(['MEAN'] + [
            ('' if not [cell(m, at) for at in attacks if cell(m, at) is not None]
             else round(sum(c for at in attacks if (c := cell(m, at)) is not None) /
                        len([at for at in attacks if cell(m, at) is not None]), 3))
            for m in models])

    # ---- Markdown ----
    lines = ['# VoxWatermark no-box benchmark — model comparison', '',
             f'Mean bit-accuracy across each attack\'s strength sweep. '
             f'**Bold** = survives (mean bit-acc >= {DETECT_BAR}).', '',
             '| Attack | ' + ' | '.join(models) + ' |',
             '|' + '---|' * (len(models) + 1)]
    means = {m: [] for m in models}
    for at in attacks:
        cells = []
        for m in models:
            c = cell(m, at)
            if c is None:
                cells.append('—')
            else:
                means[m].append(c)
                cells.append(f'**{c:.2f}**' if c >= DETECT_BAR else f'{c:.2f}')
        lines.append(f'| {at} | ' + ' | '.join(cells) + ' |')
    # overall mean
    mean_cells = []
    for m in models:
        vals = means[m]
        mean_cells.append('—' if not vals else f'{sum(vals) / len(vals):.2f}')
    lines.append('| **MEAN** | ' + ' | '.join(mean_cells) + ' |')

    # robustness ranking by overall mean
    ranked = sorted(
        ((m, sum(means[m]) / len(means[m])) for m in models if means[m]),
        key=lambda kv: kv[1], reverse=True)
    lines += ['', '## Overall robustness ranking', '']
    for i, (m, v) in enumerate(ranked, 1):
        lines.append(f'{i}. **{m}** — mean bit-acc {v:.3f}')

    with open(os.path.join(OUT, 'vox_comparison.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print('Wrote:')
    print('  ', os.path.join(OUT, 'vox_comparison.csv'))
    print('  ', os.path.join(OUT, 'vox_comparison.md'))
    print()
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
