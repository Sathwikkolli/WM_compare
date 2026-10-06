"""
run_probe.py  --  does watermarking make real speech look like a deepfake?

Input: 6 real Emilia speech clips (one per speaker, DNSMOS >= 3, trimmed to 10 s),
loaded at 16 kHz. Everything is done at 16 kHz:
    clean       original, through the same 16 kHz read/write path (control)
    aware_only  AWARE alone
    all_three   AudioSeal -> AWARE -> Timbre, stacked (cascade order)

AudioSeal and AWARE embed/detect natively at 16 kHz (no resampling). Timbre only
runs at 22.05 kHz, so its step alone is 16k -> 22.05k -> embed -> 16k.

Then for each version:
    * every watermark is detected (on all versions, so clean is the false-positive floor)
    * PESQ / SNR / SI-SNR / STOI vs clean
    * three deepfake detectors (AASIST, RawNet2, XLS-R) give P(spoof)

Output -> results/<date>_deepfake-probe-emilia16k/{data/*.csv, summary.md, params.json}
Audio  -> deepfake_probe/work/emilia16k/<clip>/<variant>.wav   (gitignored)

    python run_probe.py                 # pick 6 Emilia clips, embed, detect
    python run_probe.py --reuse-wavs    # skip embedding if wavs already exist
"""
import os, sys, json, argparse, datetime, platform, subprocess
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
os.environ.setdefault('WM_COMPARE_BASE', REPO)
sys.path.insert(0, os.path.join(REPO, 'cascade'))
import cascade_lib as cl                                       # noqa: E402
import detectors as D                                          # noqa: E402

SR = 16000                       # working rate for everything in this experiment
VARIANTS = {                     # variant -> watermarks embedded, in order
    'clean':      (),
    'aware_only': ('aware',),
    'all_three':  ('audioseal', 'aware', 'timbre'),
}
THRESH = 0.5                     # P(spoof) > THRESH -> verdict "fake"

# clip selection -- same source and filters as cascade/emilia_bench.py
EMILIA_CSV = os.environ.get(
    'EMILIA_CSV', os.path.expanduser('~/projects/aura_watermark/data/val.csv'))
N_CLIPS, SECONDS, DNSMOS_MIN, SEED = 6, 10.0, 3.0, 1234


def pick_emilia(csv_path):
    """N_CLIPS held-out Emilia clips, one per speaker, deterministic for SEED."""
    df = pd.read_csv(csv_path)
    if 'dataset' in df.columns:                 # val.csv mixes Emilia + FMA music
        df = df[df['dataset'] == 'emilia']
    df = df[(df['dnsmos'] >= DNSMOS_MIN) & (df['duration_s'] >= SECONDS)]
    sel = (df.sample(frac=1.0, random_state=SEED)
             .drop_duplicates('speaker').head(N_CLIPS).reset_index(drop=True))
    if len(sel) < N_CLIPS:
        sys.exit(f'only {len(sel)} speakers pass the filters in {csv_path}')
    sel.insert(0, 'clip_id', [f'emilia_{i:02d}' for i in range(len(sel))])
    return sel


# ---- 16 kHz embed / detect ------------------------------------------------- #
# cascade_lib's adapters take 22.05 kHz input and resample internally. Here the
# signal is already 16 kHz, so AudioSeal/AWARE are called at their native rate
# directly (same calls as the adapters, minus the resampling); only Timbre,
# which has no 16 kHz mode, goes through 22.05 kHz.
def embed16(tool, y):
    a = cl.get_adapter(tool)
    if tool == 'audioseal':
        torch = a._torch
        x = torch.from_numpy(y).view(1, 1, -1)
        msg = torch.tensor([[int(b) for b in a.truth]], dtype=torch.int32)
        with torch.no_grad():
            w = a._gen(x, sample_rate=SR, message=msg, alpha=1.0)
        z = w.squeeze().cpu().numpy()
    elif tool == 'aware':
        from aware.service import embed_watermark
        bits = np.array([int(b) for b in a.truth], dtype=np.int64)
        z = np.asarray(embed_watermark(y, SR, bits, a._emb))
    else:
        z = cl.resample(a.embed(cl.resample(y, SR, cl.SR_MASTER)), cl.SR_MASTER, SR)
    z = np.asarray(z, dtype='float32').ravel()[:len(y)]
    return np.clip(np.pad(z, (0, len(y) - len(z))), -1.0, 1.0)


def detect16(tool, y):
    """-> (conf, bits, bit_acc), same contract as the cascade_lib adapters."""
    a = cl.get_adapter(tool)
    if tool == 'audioseal':
        torch = a._torch
        with torch.no_grad():
            prob, msg = a._det.detect_watermark(torch.from_numpy(y).view(1, 1, -1),
                                                sample_rate=SR)
        bits = ''.join(map(str, (msg.squeeze() > 0.5).int().tolist()))
        return float(prob), bits, cl.bit_acc(bits, a.truth)
    if tool == 'aware':
        from aware.service import detect_watermark
        pat, conf = detect_watermark(y, SR, a._det)
        bits = ''.join(map(str, np.asarray(pat).astype(int).ravel()[:len(a.truth)].tolist()))
        return float(conf), bits, cl.bit_acc(bits, a.truth)
    return a.detect(cl.resample(y, SR, cl.SR_MASTER))


def build_variants(clip_id, src, work, reuse):
    d = os.path.join(work, clip_id)
    y = cl.read_wav(src, SR)[:int(SECONDS * SR)]
    paths = {}
    for var, chain in VARIANTS.items():
        p = os.path.join(d, f'{var}.wav')
        paths[var] = p
        if reuse and os.path.exists(p):
            continue
        z = y.copy()
        for tool in chain:
            z = embed16(tool, z)
        cl.write_wav(p, z, SR)
        print(f'  [{clip_id}] wrote {var}', flush=True)
    return clip_id, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--emilia-csv', default=EMILIA_CSV)
    ap.add_argument('--work', default=os.path.join(HERE, 'work', 'emilia16k'))
    ap.add_argument('--out',  default=os.path.join(
        REPO, 'results', f'{datetime.date.today():%Y-%m-%d}_deepfake-probe-emilia16k'))
    ap.add_argument('--reuse-wavs', action='store_true')
    a = ap.parse_args()

    data = os.path.join(a.out, 'data')
    if os.path.exists(os.path.join(data, 'deepfake_scores.csv')):
        sys.exit(f'{data} already has results -- results/ is never overwritten; use --out')
    os.makedirs(data, exist_ok=True)
    sel = pick_emilia(a.emilia_csv)
    sel.to_csv(os.path.join(data, 'clips.csv'), index=False)
    print(sel[['clip_id', 'speaker', 'dnsmos', 'duration_s', 'path']].to_string(index=False))

    # ---- 1. embed ---------------------------------------------------------- #
    built = [build_variants(r.clip_id, r.path, a.work, a.reuse_wavs)
             for r in sel.itertuples()]

    # ---- 2. watermark check + quality ------------------------------------- #
    wm_rows, q_rows = [], []
    for clip_id, paths in built:
        for var, p in paths.items():
            y = cl.read_wav(p, SR)
            for tool in cl.TOOLS:
                conf, bits, acc = detect16(tool, y)
                wm_rows.append(dict(clip=clip_id, variant=var, watermark=tool,
                                    embedded=tool in VARIANTS[var], conf=round(conf, 4),
                                    bit_acc=round(acc, 4), detected=acc >= 0.8))
            if var != 'clean':
                q_rows.append(dict(clip=clip_id, variant=var,
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
    for clip_id, paths in built:
        for var, p in paths.items():
            y16 = cl.read_wav(p, SR)
            for det in dets:
                ps, per = det.p_spoof(y16)
                rows.append(dict(clip=clip_id, variant=var, detector=det.name,
                                 p_spoof=round(ps, 4), verdict='fake' if ps > THRESH else 'real',
                                 n_windows=len(per), p_spoof_min=round(min(per), 4),
                                 p_spoof_max=round(max(per), 4)))
                print(f'  {clip_id:10s} {var:10s} {det.name:8s} P(spoof)={ps:.3f}', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(data, 'deepfake_scores.csv'), index=False)

    write_summary(a.out, df, pd.DataFrame(wm_rows), pd.DataFrame(q_rows))
    write_params(a.out, sel, a.emilia_csv, dets)
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
        f'{n} real Emilia speech clips (one per speaker, {SECONDS:.0f} s, 16 kHz throughout). '
        f'P(spoof) is the detector\'s probability the clip is fake, averaged over 4.04 s windows. '
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


def write_params(out, sel, emilia_csv, dets):
    def sh(c):
        try: return subprocess.check_output(c, cwd=REPO, text=True).strip()
        except Exception: return None
    import torch
    p = dict(
        git_commit=sh(['git', 'rev-parse', 'HEAD']), slurm_job=os.environ.get('SLURM_JOB_ID'),
        date=datetime.datetime.now().isoformat(timespec='seconds'), host=platform.node(),
        emilia_csv=emilia_csv, clips=sel[['clip_id', 'speaker', 'path']].to_dict('records'),
        clip_filter=dict(n=N_CLIPS, seconds=SECONDS, dnsmos_min=DNSMOS_MIN, seed=SEED),
        variants=VARIANTS, sr=SR, timbre_sr=cl.SR_MASTER, window_samples=D.WIN,
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
