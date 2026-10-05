"""
run_probe.py  --  does watermarking make real speech look like a deepfake?

For every clip, three versions are built at 22.05 kHz (cascade_lib's canonical rate):
    clean       original, passed through the same read/write path (control)
    aware_only  AWARE alone
    all_three   AudioSeal -> AWARE -> Timbre, stacked (cascade order)

Then for each version:
    * every watermark is detected (on all versions, so clean is the false-positive floor)
    * PESQ / SNR / SI-SNR / STOI vs clean
    * three deepfake detectors (AASIST, RawNet2, XLS-R) give P(spoof) at 16 kHz

Output -> results/<date>_deepfake-probe/{data/*.csv, summary.md, params.json}
Audio  -> deepfake_probe/work/<clip>/<variant>.wav   (gitignored)

    python run_probe.py                 # all of clean_01..06
    python run_probe.py --reuse-wavs    # skip embedding if wavs already exist
"""
import os, sys, glob, json, argparse, datetime, platform, subprocess
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
os.environ.setdefault('WM_COMPARE_BASE', REPO)
sys.path.insert(0, os.path.join(REPO, 'cascade'))
import cascade_lib as cl                                       # noqa: E402
import detectors as D                                          # noqa: E402

VARIANTS = {                     # variant -> watermarks embedded, in order
    'clean':      (),
    'aware_only': ('aware',),
    'all_three':  ('audioseal', 'aware', 'timbre'),
}
THRESH = 0.5                     # P(spoof) > THRESH -> verdict "fake"


def build_variants(clip, work, reuse):
    stem = os.path.splitext(os.path.basename(clip))[0]
    d = os.path.join(work, stem)
    y = cl.read_wav(clip, cl.SR_MASTER)
    paths = {}
    for var, chain in VARIANTS.items():
        p = os.path.join(d, f'{var}.wav')
        paths[var] = p
        if reuse and os.path.exists(p):
            continue
        z = y.copy()
        for tool in chain:
            z = cl.get_adapter(tool).embed(z)
            z = np.clip(z, -1.0, 1.0)
        cl.write_wav(p, z, cl.SR_MASTER)
        print(f'  [{stem}] wrote {var}', flush=True)
    return stem, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', default=os.path.join(REPO, 'audio', 'clean_set',
                                                    'clean_0[1-6]_speech_*.wav'))
    ap.add_argument('--work', default=os.path.join(HERE, 'work'))
    ap.add_argument('--out',  default=os.path.join(
        REPO, 'results', f'{datetime.date.today():%Y-%m-%d}_deepfake-probe'))
    ap.add_argument('--reuse-wavs', action='store_true')
    a = ap.parse_args()

    clips = sorted(glob.glob(a.clips))
    if not clips:
        sys.exit(f'no clips match {a.clips}')
    data = os.path.join(a.out, 'data')
    if os.path.exists(os.path.join(data, 'deepfake_scores.csv')):
        sys.exit(f'{data} already has results -- results/ is never overwritten; use --out')
    os.makedirs(data, exist_ok=True)

    # ---- 1. embed ---------------------------------------------------------- #
    built = [build_variants(c, a.work, a.reuse_wavs) for c in clips]

    # ---- 2. watermark check + quality ------------------------------------- #
    wm_rows, q_rows = [], []
    for stem, paths in built:
        for var, p in paths.items():
            y = cl.read_wav(p, cl.SR_MASTER)
            for tool in cl.TOOLS:
                conf, bits, acc = cl.get_adapter(tool).detect(y)
                wm_rows.append(dict(clip=stem, variant=var, watermark=tool,
                                    embedded=tool in VARIANTS[var], conf=round(conf, 4),
                                    bit_acc=round(acc, 4), detected=acc >= 0.8))
            if var != 'clean':
                q_rows.append(dict(clip=stem, variant=var,
                                   **cl.quality_metrics(paths['clean'], p)))
    pd.DataFrame(wm_rows).to_csv(os.path.join(data, 'watermark_check.csv'), index=False)
    pd.DataFrame(q_rows).to_csv(os.path.join(data, 'quality.csv'), index=False)

    # free watermark models before loading the detectors (XLS-R is ~1.2 GB)
    cl._ADAPTERS.clear()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- 3. deepfake detectors -------------------------------------------- #
    dets = D.load_all()
    rows = []
    for stem, paths in built:
        for var, p in paths.items():
            y16 = cl.read_wav(p, cl.SR_16K)
            for det in dets:
                ps, per = det.p_spoof(y16)
                rows.append(dict(clip=stem, variant=var, detector=det.name,
                                 p_spoof=round(ps, 4), verdict='fake' if ps > THRESH else 'real',
                                 n_windows=len(per), p_spoof_min=round(min(per), 4),
                                 p_spoof_max=round(max(per), 4)))
                print(f'  {stem:28s} {var:10s} {det.name:8s} P(spoof)={ps:.3f}', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(data, 'deepfake_scores.csv'), index=False)

    write_summary(a.out, df, pd.DataFrame(wm_rows), pd.DataFrame(q_rows))
    write_params(a.out, clips, dets)
    print(f'\ndone -> {a.out}')


def md_table(t):
    """DataFrame -> GitHub markdown (pandas' to_markdown needs tabulate, not in the env)."""
    t = t.reset_index()
    cell = lambda v: f'{v:.3f}' if isinstance(v, (float, np.floating)) else str(v)
    lines = ['| ' + ' | '.join(map(str, t.columns)) + ' |',
             '|' + '---|' * len(t.columns)]
    lines += ['| ' + ' | '.join(cell(v) for v in row) + ' |' for row in t.itertuples(index=False)]
    return '\n'.join(lines)


def write_summary(out, df, wm, q):
    order = list(VARIANTS)
    mean = df.pivot_table(index='variant', columns='detector', values='p_spoof',
                          aggfunc='mean').reindex(order)
    nfake = df.assign(f=df.verdict.eq('fake')).pivot_table(
        index='variant', columns='detector', values='f', aggfunc='sum').reindex(order)
    n = df['clip'].nunique()
    # paired change vs clean, per clip
    piv = df.pivot_table(index=['clip', 'detector'], columns='variant', values='p_spoof')
    delta = pd.DataFrame({v: (piv[v] - piv['clean']) for v in order[1:]})
    dmean = delta.groupby('detector').mean().T.rename_axis('variant')
    dmax = delta.abs().groupby('detector').max().T.rename_axis('variant')
    per_clip = df.pivot_table(index=['clip', 'variant'], columns='detector',
                              values='p_spoof').reindex(order, level='variant')

    emb = wm[wm.embedded]
    wm_t = emb.pivot_table(index='variant', columns='watermark', values='bit_acc',
                           aggfunc='mean').reindex(order[1:])
    fp = wm[(wm.variant == 'clean')].groupby('watermark').detected.sum()
    q_t = q.groupby('variant')[['pesq', 'snr_db', 'si_snr_db', 'stoi']].mean().reindex(order[1:])

    f = md_table
    md = [
        '# Deepfake detectors vs watermarked speech -- summary\n',
        f'{n} real speech clips (`clean_01`-`clean_06`, 8 kHz source). P(spoof) is the '
        f'detector\'s probability the clip is fake, averaged over 4.04 s windows. '
        f'Verdict "fake" = P(spoof) > {THRESH}.\n',
        '## Mean P(spoof)\n', f(mean), '',
        f'## Clips called "fake" (out of {n})\n', f(nfake.astype(int)), '',
        '## Change in P(spoof) vs clean (paired, per clip)\n',
        'Mean signed change:\n', f(dmean), '', 'Largest absolute change on any clip:\n', f(dmax), '',
        '## Per-clip P(spoof)\n', f(per_clip), '',
        '## Sanity checks\n',
        'Watermark bit accuracy on the variants where it was embedded (want ~1.0):\n', f(wm_t), '',
        'False positives on clean (clips where a watermark was "detected" without being embedded):\n',
        f(fp.to_frame('false_pos')), '',
        'Audio quality vs clean:\n', f(q_t), '',
        '## Conclusion\n',
        '_TODO: write in words once the numbers are in (results/README.md rule 2)._\n',
    ]
    open(os.path.join(out, 'summary.md'), 'w').write('\n'.join(md))


def write_params(out, clips, dets):
    def sh(c):
        try: return subprocess.check_output(c, cwd=REPO, text=True).strip()
        except Exception: return None
    import torch
    p = dict(
        git_commit=sh(['git', 'rev-parse', 'HEAD']), slurm_job=os.environ.get('SLURM_JOB_ID'),
        date=datetime.datetime.now().isoformat(timespec='seconds'), host=platform.node(),
        clips=[os.path.relpath(c, REPO) for c in clips], variants=VARIANTS,
        sr_master=cl.SR_MASTER, detector_sr=cl.SR_16K, window_samples=D.WIN,
        threshold=THRESH, wm_detect_threshold=0.8,
        truth_bits=dict(audioseal=cl.AUDIOSEAL_BITS, aware=cl.AWARE_BITS),
        detectors=dict(AASIST='clovaai/aasist models/weights/AASIST.pth (ASVspoof2019 LA)',
                       RawNet2='ASVspoof2021 baseline, pre_trained_DF_RawNet2.zip ('
                               + getattr(next(d for d in dets if d.name == 'RawNet2'), 'weights', '?') + ')',
                       XLSR=D.XLSR_ID),
        torch=torch.__version__, cuda=torch.cuda.is_available(),
    )
    json.dump(p, open(os.path.join(out, 'params.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
