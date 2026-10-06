"""
spectrograms.py  --  where does each watermark put its energy?

Descriptive, NOT a model comparison. For each clip and each watermark embedded
ALONE (AudioSeal, AWARE, Timbre), three constant-Q spectrograms, one PNG each:

    1_original.png      the clip before watermarking
    2_watermarked.png   the clip after watermarking
    3_diff.png          CQT of (watermarked - original) = the watermark signal itself

Constant-Q (24 bins/octave, 32.7 Hz - 7.6 kHz at 16 kHz) gives every octave the same
height, so the low frequencies that a linear STFT squashes into a few rows are readable.
All three images share one dB reference (the original's peak), so the diff's colours
say how far below the speech the watermark sits.

Plus stats per clip x watermark (data/watermark_stats.csv, explained in summary.md):
where the watermark energy sits by band, how loud it is relative to the speech in
each band, whether it follows the speech spectro-temporally, and whether it is
embedded in pauses as well as in speech.

Uses the same Emilia clips and 16 kHz wavs as run_probe.py. aware_only.wav is reused
from that run; audioseal_only / timbre_only are embedded here (fast).

    python spectrograms.py                       # first 2 clips of the latest probe run
    python spectrograms.py --n-clips 6
    python spectrograms.py --n-clips 1 --variants all_three   # stacked: all 3 in one file
"""
import os, sys, glob, argparse, datetime
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import run_probe as P                                          # noqa: E402
cl = P.cl
SR = P.SR

TOOLS = ('audioseal', 'aware', 'timbre')
STACKED = 'all_three'            # AudioSeal -> AWARE -> Timbre, from run_probe.py
NAME = {'audioseal': 'AudioSeal', 'aware': 'AWARE', 'timbre': 'Timbre',
        STACKED: 'All three (AudioSeal -> AWARE -> Timbre)'}
# CQT: C1 (32.7 Hz) upward, 24 bins/octave, stop below Nyquist so the top filters fit
FMIN, BPO, HOP = 32.70, 24, 256
N_BINS = int(BPO * np.log2(7600 / FMIN))
DYN_DB = 90                       # dB range shown in every image
BANDS = [(0, 300), (300, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
N_FFT = 1024                      # STFT used for the band statistics only


# --------------------------------------------------------------------------- #
#  audio
# --------------------------------------------------------------------------- #
def wav_pairs(clip_id, work, variants):
    """-> {variant: (reference_wav, watermarked_wav)}; embeds whatever is missing.
    variants: any of TOOLS (embedded alone) and/or STACKED."""
    d = os.path.join(work, clip_id)
    clean_p = os.path.join(d, 'clean.wav')
    if not os.path.exists(clean_p):
        sys.exit(f'{clean_p} missing -- run run_probe.py first')
    clean = cl.read_wav(clean_p, SR)
    # Timbre's step is 16k -> 22.05k -> 16k; anything containing Timbre gets a
    # reference with the same round trip, so the diff holds only the watermark,
    # not resampling error.
    rt_p = os.path.join(d, 'clean_rt22k.wav')
    if not os.path.exists(rt_p):
        rt = cl.resample(cl.resample(clean, SR, cl.SR_MASTER), cl.SR_MASTER, SR)
        cl.write_wav(rt_p, rt[:len(clean)], SR)
    out = {}
    for v in variants:
        chain = P.VARIANTS[STACKED] if v == STACKED else (v,)
        wm_p = os.path.join(d, f'{v}.wav' if v == STACKED else f'{v}_only.wav')
        if not os.path.exists(wm_p):
            z = clean.copy()
            for tool in chain:
                z = P.embed16(tool, z)
            cl.write_wav(wm_p, z, SR)
            print(f'  [{clip_id}] embedded {os.path.basename(wm_p)}', flush=True)
        out[v] = (rt_p if 'timbre' in chain else clean_p, wm_p)
    return out


def load_pair(ref_p, wm_p):
    x = cl.read_wav(ref_p, SR); y = cl.read_wav(wm_p, SR)
    n = min(len(x), len(y))
    return x[:n], y[:n]


# --------------------------------------------------------------------------- #
#  spectrograms
# --------------------------------------------------------------------------- #
def cqt_db(sig, ref):
    import librosa
    C = np.abs(librosa.cqt(sig, sr=SR, hop_length=HOP, fmin=FMIN,
                           n_bins=N_BINS, bins_per_octave=BPO))
    return 20 * np.log10(np.maximum(C, 1e-10) / ref)


def save_cqt(S_db, path, title):
    import librosa.display, matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = librosa.display.specshow(S_db, sr=SR, hop_length=HOP, fmin=FMIN,
                                  bins_per_octave=BPO, x_axis='time', y_axis='cqt_hz',
                                  cmap='magma', vmin=-DYN_DB, vmax=0, ax=ax)
    ax.set_title(title); ax.set_xlabel('Time (s)')
    fig.colorbar(im, ax=ax, format='%+.0f dB', label='dB re original peak')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def make_images(x, y, out_dir, clip_id, tool):
    import librosa
    os.makedirs(out_dir, exist_ok=True)
    ref = np.abs(librosa.cqt(x, sr=SR, hop_length=HOP, fmin=FMIN,
                             n_bins=N_BINS, bins_per_octave=BPO)).max()
    t = f'{clip_id}  |  {NAME[tool]}'
    save_cqt(cqt_db(x, ref), os.path.join(out_dir, '1_original.png'),
             f'{t}  |  original')
    save_cqt(cqt_db(y, ref), os.path.join(out_dir, '2_watermarked.png'),
             f'{t}  |  watermarked')
    save_cqt(cqt_db(y - x, ref), os.path.join(out_dir, '3_diff.png'),
             f'{t}  |  diff = watermarked - original (the watermark signal)')


# --------------------------------------------------------------------------- #
#  stats
# --------------------------------------------------------------------------- #
def db(a, b):
    return float(10 * np.log10((a + 1e-20) / (b + 1e-20)))


def stats(x, y):
    import librosa
    r = y - x
    X = np.abs(librosa.stft(x, n_fft=N_FFT, hop_length=N_FFT // 4)) ** 2
    R = np.abs(librosa.stft(r, n_fft=N_FFT, hop_length=N_FFT // 4)) ** 2
    f = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    px, pr = X.mean(1), R.mean(1)                     # average power spectra
    s = dict(snr_db=round(db(np.sum(x ** 2), np.sum(r ** 2)), 2),
             peak_wm_amp=round(float(np.abs(r).max()), 5))
    # where the watermark energy is, and how loud it is vs the speech, per band
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        tag = f'{lo}-{hi}'
        s[f'wm_energy_pct_{tag}'] = round(100 * pr[m].sum() / pr.sum(), 2)
        s[f'wm_to_speech_db_{tag}'] = round(db(pr[m].sum(), px[m].sum()), 2)
    # spectral shape of the watermark
    cdf = np.cumsum(pr) / pr.sum()
    s['wm_centroid_hz'] = round(float((f * pr).sum() / pr.sum()), 1)
    s['wm_90pct_band_lo_hz'] = round(float(f[np.searchsorted(cdf, 0.05)]), 1)
    s['wm_90pct_band_hi_hz'] = round(float(f[min(np.searchsorted(cdf, 0.95), len(f) - 1)]), 1)
    # does the watermark follow the speech? (log-power correlation over all TF bins,
    # and over time alone)
    lx, lr = 10 * np.log10(X + 1e-12).ravel(), 10 * np.log10(R + 1e-12).ravel()
    s['tf_corr_wm_vs_speech'] = round(float(np.corrcoef(lx, lr)[0, 1]), 3)
    ex, er = X.sum(0), R.sum(0)                         # per-frame energy
    # floor both at -100 dB re the loudest speech frame, so near-digital-silence
    # pauses give a bounded ratio instead of dividing by ~0
    floor = ex.max() * 1e-10
    ex, er = np.maximum(ex, floor), np.maximum(er, floor)
    s['time_corr_wm_vs_speech'] = round(float(np.corrcoef(10 * np.log10(ex + 1e-12),
                                                          10 * np.log10(er + 1e-12))[0, 1]), 3)
    # speech vs pauses: frames more than 40 dB below the loudest frame = pause
    pause = 10 * np.log10(ex / ex.max() + 1e-20) < -40
    s['pause_frames_pct'] = round(100 * pause.mean(), 1)
    if pause.any() and (~pause).any():
        s['wm_speech_minus_pause_db'] = round(db(er[~pause].mean(), er[pause].mean()), 2)
        s['wm_to_speech_db_in_pauses'] = round(db(er[pause].mean(), ex[pause].mean()), 2)
    else:
        s['wm_speech_minus_pause_db'] = s['wm_to_speech_db_in_pauses'] = None
    return s


# --------------------------------------------------------------------------- #
#  summary
# --------------------------------------------------------------------------- #
EXPLAIN = """\
## What each statistic means

All computed on the diff signal r = watermarked - original (the watermark itself),
at 16 kHz, so 0-8 kHz.

| column | meaning |
|---|---|
| `snr_db` | speech power / watermark power over the whole clip. Higher = quieter watermark. |
| `peak_wm_amp` | largest sample of the watermark signal (full scale = 1.0). |
| `wm_energy_pct_<band>` | share of the watermark's energy in that band. Rows sum to ~100. **This is "where it is embedded".** |
| `wm_to_speech_db_<band>` | watermark power vs speech power inside that band. Near 0 or positive = the watermark is as loud as (or louder than) the speech there, e.g. bands where speech has little energy. Very negative = hidden well under the speech. |
| `wm_centroid_hz` | centre of mass of the watermark's spectrum. |
| `wm_90pct_band_lo/hi_hz` | the frequency range holding the middle 90% of the watermark's energy. |
| `tf_corr_wm_vs_speech` | correlation of log-power between watermark and speech over every time-frequency cell. Near 1 = the watermark is shaped like the speech (strong where speech is strong, which is how perceptual masking works); near 0 = spread independently of the speech. |
| `time_corr_wm_vs_speech` | the same, over time only: does the watermark get louder when the speaker does? |
| `pause_frames_pct` | share of frames more than 40 dB below the loudest frame (pauses). Frame energies are floored at -100 dB re the loudest frame, so the two pause columns below are capped at about +/-100 dB. |
| `wm_speech_minus_pause_db` | watermark energy during speech minus during pauses. Large positive = embedded mainly in speech; near 0 = embedded at the same level in pauses too. Empty if the clip has no pauses. |
| `wm_to_speech_db_in_pauses` | watermark vs (near-silent) background in pauses. Positive = the watermark is the loudest thing during pauses, which is where it is most audible. |
"""


def write_summary(out, df, clips):
    bands = [f'{lo}-{hi}' for lo, hi in BANDS]
    mean = df.groupby('watermark', sort=False).mean(numeric_only=True)
    f = P.md_table
    t_where = mean[[f'wm_energy_pct_{b}' for b in bands]].rename(
        columns=lambda c: c.replace('wm_energy_pct_', '') + ' Hz')
    t_loud = mean[[f'wm_to_speech_db_{b}' for b in bands]].rename(
        columns=lambda c: c.replace('wm_to_speech_db_', '') + ' Hz')
    rest = ['snr_db', 'peak_wm_amp', 'wm_centroid_hz', 'wm_90pct_band_lo_hz',
            'wm_90pct_band_hi_hz', 'tf_corr_wm_vs_speech', 'time_corr_wm_vs_speech',
            'pause_frames_pct', 'wm_speech_minus_pause_db', 'wm_to_speech_db_in_pauses']
    md = [
        '# Where each watermark is embedded -- summary\n',
        f'Clips: {", ".join(clips)} (Emilia, 10 s, 16 kHz). Each solo watermark embedded '
        f'alone; `{STACKED}` = AudioSeal -> AWARE -> Timbre stacked in one file, so its diff '
        'is the combined watermark. '
        f'Numbers are means over the {len(clips)} clips; per-clip values are in '
        '`data/watermark_stats.csv`. Images: `figures/<clip>/<watermark>/`.\n',
        'Timbre runs at 22.05 kHz, so it (and the stacked file) use a clean reference that went '
        'through the same 16k -> 22.05k -> 16k round trip; its diff is the watermark only.\n',
        '## Share of watermark energy by band (%)\n', f(t_where), '',
        '## Watermark loudness vs speech, by band (dB)\n', f(t_loud), '',
        '## Shape, masking and timing\n', f(mean[rest]), '',
        EXPLAIN,
        '## Observations\n',
        '_TODO: describe each watermark in words after looking at the diff images._\n',
    ]
    open(os.path.join(out, 'summary.md'), 'w').write('\n'.join(md))


def latest_probe():
    runs = sorted(glob.glob(os.path.join(REPO, 'results', '*_deepfake-probe-emilia16k')))
    if not runs:
        sys.exit('no results/*_deepfake-probe-emilia16k run found -- run run_probe.py first')
    return runs[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe-results', default=None,
                    help='run_probe.py results dir (default: latest *_deepfake-probe-emilia16k)')
    ap.add_argument('--work', default=os.path.join(HERE, 'work', 'emilia16k'))
    ap.add_argument('--n-clips', type=int, default=2)
    ap.add_argument('--variants', default=','.join(TOOLS),
                    help=f'comma list of {",".join(TOOLS + (STACKED,))}; '
                         f'{STACKED} = all three stacked in one file')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    variants = [v.strip() for v in a.variants.split(',') if v.strip()]
    bad = set(variants) - set(TOOLS + (STACKED,))
    if bad:
        sys.exit(f'unknown --variants {sorted(bad)}')
    slug = 'watermark-spectrograms' + ('-stacked' if variants == [STACKED] else '')
    a.out = a.out or os.path.join(REPO, 'results', f'{datetime.date.today():%Y-%m-%d}_{slug}')
    probe = a.probe_results or latest_probe()
    clips = pd.read_csv(os.path.join(probe, 'data', 'clips.csv'))['clip_id'].head(a.n_clips).tolist()
    data = os.path.join(a.out, 'data')
    if os.path.exists(os.path.join(data, 'watermark_stats.csv')):
        sys.exit(f'{data} already has results -- results/ is never overwritten; use --out')
    os.makedirs(data, exist_ok=True)
    print(f'clips {clips} from {probe}')

    rows = []
    for clip_id in clips:
        for tool, (ref_p, wm_p) in wav_pairs(clip_id, a.work, variants).items():
            x, y = load_pair(ref_p, wm_p)
            make_images(x, y, os.path.join(a.out, 'figures', clip_id, tool), clip_id, tool)
            rows.append(dict(clip=clip_id, watermark=tool, **stats(x, y)))
            print(f'  [{clip_id}] {tool}: 3 images + stats', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(data, 'watermark_stats.csv'), index=False)
    write_summary(a.out, df, clips)
    print(f'\ndone -> {a.out}')


if __name__ == '__main__':
    main()
