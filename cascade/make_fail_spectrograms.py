"""
make_fail_spectrograms.py  --  Spectrograms of AWARE's failure cases.

Renders, for the AWARE watermark:
  1. the ORIGINAL (clean) clip,
  2. the WATERMARKED clip,
  3. the watermark RESIDUAL (watermarked - original, amplified) so you can see the mark,
  4. every ATTACK where AWARE drops below the detection threshold (bit_acc < 0.8).

The failing attacks are exactly the (attack, strength) cells that show detected=0
for AWARE in vox_out/vox_results.csv:
    encodec 1.5 / 3.0 / 6.0 kbps,  quantization 4-level,  highpass 0.2 / 0.5
We re-run AWARE detection on each attacked clip so the printed bit-accuracy is live,
not copied from the CSV.

Runs on the Great Lakes setup (aware/src on PYTHONPATH via cascade_lib, WM_COMPARE_BASE
pointing at the checkout). Outputs PNGs to  vox_out/figs/aware_fail/  plus one montage.

Usage (from the cascade/ directory):
    python make_fail_spectrograms.py
    python make_fail_spectrograms.py --use-wm ../audio/aware_wm.wav   # skip the slow embed
    python make_fail_spectrograms.py --seconds 6                      # crop display window
    python make_fail_spectrograms.py --mel                            # mel instead of linear STFT
"""
import os, sys, argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')                      # headless (cluster) backend
import matplotlib.pyplot as plt

import cascade_lib as L
from cascade_lib import BASE, AUDIO, SR_MASTER, get_adapter, read_wav, write_wav
import vox_attacks as V

OUTDIR = os.path.join(BASE, 'vox_out', 'figs', 'aware_fail')

# (attack, strength_label, param)  -- the AWARE detected=0 cells from vox_results.csv.
# param must match vox_attacks.VOX_GRID:  encodec->bandwidth float, quantization->levels
# int, highpass->cutoff ratio float.
FAILING = [
    ('encodec',      '1.5kbps', 1.5),
    ('encodec',      '3.0kbps', 3.0),
    ('encodec',      '6.0kbps', 6.0),
    ('quantization', '4lvl',    4),
    ('highpass',     '0.2',     0.2),
    ('highpass',     '0.5',     0.5),
]


# --------------------------------------------------------------------------- #
#  Spectrogram helpers
# --------------------------------------------------------------------------- #
def _stft_db(y, sr, n_fft=1024, hop=256, mel=False):
    import librosa
    if mel:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=128)
        return librosa.power_to_db(S + 1e-10, ref=np.max), 'mel'
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, window='hann'))
    return librosa.amplitude_to_db(S + 1e-10, ref=np.max), 'linear'


def _crop(y, sr, seconds):
    if seconds and len(y) > int(seconds * sr):
        return y[: int(seconds * sr)]
    return y


def _panel(ax, y, sr, title, seconds, mel, vmin=-80, cmap='magma'):
    import librosa.display
    yc = _crop(y, sr, seconds)
    D, ytype = _stft_db(yc, sr, mel=mel)
    img = librosa.display.specshow(D, sr=sr, hop_length=256, x_axis='time',
                                   y_axis=ytype, ax=ax, cmap=cmap, vmin=vmin, vmax=0)
    ax.set_title(title, fontsize=10)
    return img


def save_single(y, sr, title, fname, seconds, mel):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    img = _panel(ax, y, sr, title, seconds, mel)
    fig.colorbar(img, ax=ax, format='%+0.0f dB')
    fig.tight_layout()
    out = os.path.join(OUTDIR, fname)
    fig.savefig(out, dpi=140); plt.close(fig)
    print('  wrote', out, flush=True)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.path.join(AUDIO, 'client_original_16k.wav'),
                    help='clean master clip')
    ap.add_argument('--use-wm', default=None,
                    help='path to a precomputed AWARE-watermarked wav (skips the slow embed)')
    ap.add_argument('--seconds', type=float, default=6.0,
                    help='display window length in seconds (0 = whole clip)')
    ap.add_argument('--mel', action='store_true', help='mel spectrogram instead of linear STFT')
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    # 1. original -----------------------------------------------------------
    master = read_wav(args.src, SR_MASTER)
    print(f'master: {args.src}  ({len(master)} samp @ {SR_MASTER})', flush=True)

    aware = get_adapter('aware')

    # 2. watermarked (embed, or reuse a cached wav) -------------------------
    if args.use_wm:
        wm = read_wav(args.use_wm, SR_MASTER)
        n = min(len(wm), len(master)); master, wm = master[:n], wm[:n]
        print(f'watermarked: loaded {args.use_wm}', flush=True)
    else:
        print('embedding AWARE (slow: ~400-iter optimization) ...', flush=True)
        wm = aware.embed(master)
        n = min(len(wm), len(master)); master, wm = master[:n], wm[:n]
        write_wav(os.path.join(OUTDIR, 'aware_watermarked.wav'), wm, SR_MASTER)

    conf0, _, acc0 = aware.detect(wm)
    print(f'clean watermarked detection: bit_acc={acc0:.3f} conf={conf0:.3f}', flush=True)

    save_single(master, SR_MASTER, 'Original (clean)', '01_original.png', args.seconds, args.mel)
    save_single(wm, SR_MASTER, f'AWARE watermarked  (bit_acc={acc0:.2f})',
                '02_watermarked.png', args.seconds, args.mel)

    # 3. watermark residual (amplified) -------------------------------------
    resid = wm - master
    amp = float(np.max(np.abs(master))) / (float(np.max(np.abs(resid))) + 1e-12)
    save_single(resid * amp, SR_MASTER,
                f'Watermark residual (x{amp:.0f} gain)  =  watermarked - original',
                '03_residual.png', args.seconds, args.mel)

    # 4. failing attacks ----------------------------------------------------
    results = []
    for attack, label, param in FAILING:
        try:
            y2 = V.apply(attack, param, wm, SR_MASTER)
        except Exception as e:
            print(f'  {attack} {label}: ERROR {str(e)[:60]}', flush=True); continue
        if y2 is None:
            print(f'  {attack} {label}: SKIP (dependency missing)', flush=True); continue
        conf, _, acc = aware.detect(y2)
        det = int(acc >= 0.8)
        results.append((attack, label, param, y2, acc, conf, det))
        tag = 'FAIL' if not det else 'pass'
        print(f'  {attack:12s} {label:8s} bit_acc={acc:.3f} conf={conf:.3f} [{tag}]', flush=True)
        title = f'{attack} {label}   bit_acc={acc:.2f}  conf={conf:.2f}  [{"DETECTED" if det else "FAILED"}]'
        save_single(y2, SR_MASTER, title,
                    f'fail_{attack}_{label.replace(".","p")}.png', args.seconds, args.mel)

    # 5. montage: original + watermarked + each failing attack --------------
    panels = [('Original', master, None), ('AWARE watermarked', wm, acc0)]
    panels += [(f'{a} {lb}', y2, acc) for (a, lb, p, y2, acc, cf, dt) in results]
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 3.0 * nrow), squeeze=False)
    img = None
    for i, (name, y, acc) in enumerate(panels):
        ax = axes[i // ncol][i % ncol]
        ttl = name if acc is None else f'{name}  (bit_acc={acc:.2f})'
        img = _panel(ax, y, SR_MASTER, ttl, args.seconds, args.mel)
    for j in range(len(panels), nrow * ncol):     # blank any unused axes
        axes[j // ncol][j % ncol].axis('off')
    fig.suptitle('AWARE watermark: clean vs. failure-case attacks (bit_acc < 0.80 = failed)',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    montage = os.path.join(OUTDIR, '00_montage_aware_failures.png')
    fig.savefig(montage, dpi=130); plt.close(fig)
    print('\nMONTAGE ->', montage, flush=True)
    print('DONE. PNGs in', OUTDIR, flush=True)


if __name__ == '__main__':
    main()
