"""Regenerate the 4-model Vox heatmap locally (data hardcoded from the GL sweep)."""
import os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLS = ['AudioSeal', 'AWARE', 'Timbre', 'AURA']

# (cond, AudioSeal, AWARE, Timbre, AURA)
DATA = [
    ('time_stretch:0.7', 0.62, 1.00, 1.00, 0.84),
    ('time_stretch:0.9', 0.62, 1.00, 1.00, 0.85),
    ('time_stretch:1.1', 0.50, 1.00, 1.00, 0.81),
    ('time_stretch:1.3', 0.81, 0.95, 1.00, 0.78),
    ('time_stretch:1.5', 0.75, 0.95, 1.00, 0.76),
    ('gaussian_noise:40dB', 1.00, 1.00, 1.00, 0.84),
    ('gaussian_noise:30dB', 1.00, 1.00, 1.00, 0.79),
    ('gaussian_noise:20dB', 1.00, 1.00, 1.00, 0.70),
    ('gaussian_noise:10dB', 0.88, 1.00, 1.00, 0.60),
    ('gaussian_noise:5dB', 0.88, 0.95, 0.80, 0.58),
    ('background_noise:40dB', 1.00, 1.00, 1.00, 0.91),
    ('background_noise:30dB', 1.00, 1.00, 1.00, 0.89),
    ('background_noise:20dB', 1.00, 1.00, 1.00, 0.87),
    ('background_noise:10dB', 0.88, 1.00, 1.00, 0.85),
    ('background_noise:5dB', 0.88, 1.00, 1.00, 0.84),
    ('opus:16k', 1.00, 1.00, 1.00, 0.77),
    ('opus:32k', 1.00, 1.00, 1.00, 0.82),
    ('opus:64k', 1.00, 1.00, 1.00, 0.90),
    ('opus:128k', 1.00, 1.00, 1.00, 0.91),
    ('opus:256k', 1.00, 1.00, 1.00, 0.91),
    ('encodec:1.5kbps', 0.62, 0.70, 0.60, 0.49),
    ('encodec:3.0kbps', 0.62, 0.75, 0.80, 0.50),
    ('encodec:6.0kbps', 0.81, 0.75, 0.80, 0.50),
    ('encodec:12.0kbps', 1.00, 0.80, 0.90, 0.51),
    ('encodec:24.0kbps', 1.00, 0.80, 1.00, 0.52),
    ('quantization:4lvl', 0.62, 0.70, 0.70, 0.49),
    ('quantization:8lvl', 0.44, 0.85, 0.50, 0.50),
    ('quantization:16lvl', 0.94, 0.90, 1.00, 0.52),
    ('quantization:32lvl', 0.94, 1.00, 1.00, 0.57),
    ('quantization:64lvl', 0.88, 1.00, 1.00, 0.62),
    ('highpass:0.1', 0.56, 0.85, 1.00, 0.93),
    ('highpass:0.2', 0.50, 0.65, 0.80, 0.91),
    ('highpass:0.3', 0.44, 0.85, 0.60, 0.82),
    ('highpass:0.4', 0.44, 0.95, 0.70, 0.50),
    ('highpass:0.5', 0.50, 0.65, 0.40, 0.47),
    ('lowpass:0.1', 1.00, 0.95, 0.90, 0.70),
    ('lowpass:0.2', 1.00, 1.00, 1.00, 0.84),
    ('lowpass:0.3', 1.00, 1.00, 1.00, 0.91),
    ('lowpass:0.4', 1.00, 1.00, 1.00, 0.92),
    ('lowpass:0.5', 1.00, 1.00, 1.00, 0.92),
    ('smooth:w6', 1.00, 1.00, 1.00, 0.87),
    ('smooth:w10', 1.00, 1.00, 1.00, 0.85),
    ('smooth:w14', 1.00, 1.00, 1.00, 0.84),
    ('smooth:w18', 1.00, 1.00, 0.90, 0.84),
    ('smooth:w22', 0.56, 1.00, 0.90, 0.83),
    ('echo:d0.1', 1.00, 1.00, 1.00, 0.91),
    ('echo:d0.3', 1.00, 1.00, 1.00, 0.93),
    ('echo:d0.5', 1.00, 1.00, 1.00, 0.93),
    ('echo:d0.7', 1.00, 1.00, 1.00, 0.93),
    ('echo:d0.9', 1.00, 1.00, 1.00, 0.93),
    ('mp3:8k', 1.00, 0.85, 1.00, 0.61),
    ('mp3:16k', 1.00, 1.00, 1.00, 0.77),
    ('mp3:24k', 1.00, 1.00, 1.00, 0.83),
    ('mp3:32k', 1.00, 1.00, 1.00, 0.89),
    ('mp3:40k', 1.00, 1.00, 1.00, 0.90),
    ('aac:8k', 1.00, 0.95, 1.00, 0.68),
    ('aac:40k', 1.00, 1.00, 1.00, 0.90),
    ('dynamic_compression:t-10_r2', 1.00, 1.00, 1.00, 0.90),
    ('dynamic_compression:t-30_r8', 1.00, 0.95, 1.00, 0.89),
    ('dynamic_expansion:t-10_r2', 1.00, 1.00, 1.00, 0.87),
    ('dynamic_expansion:t-30_r8', 1.00, 1.00, 1.00, 0.86),
    ('inverse_polarity:neg', 0.12, 1.00, 1.00, 0.92),
    ('time_jitter:s0.01', 1.00, 1.00, 1.00, 0.91),
    ('time_jitter:s0.5', 1.00, 1.00, 1.00, 0.78),
    ('phase_shift:1', 1.00, 1.00, 1.00, 0.92),
    ('phase_shift:-1000', 1.00, 1.00, 1.00, 0.92),
]

conds = [d[0] for d in DATA]
M = np.array([[d[1], d[2], d[3], d[4]] for d in DATA])

fig, ax = plt.subplots(figsize=(1.3 * len(COLS) + 2, max(6, 0.22 * len(conds) + 1)))
im = ax.imshow(M, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(len(COLS))); ax.set_xticklabels(COLS, fontsize=8)
ax.set_yticks(range(len(conds))); ax.set_yticklabels(conds, fontsize=6)
for i in range(len(conds)):
    for j in range(len(COLS)):
        ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=5.5)
ax.axvline(2.5, color='black', lw=1.2)
ax.set_title('Combined file + AURA (4 watermarks)', fontsize=10)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', 'vox_out', 'figs')
os.makedirs(out, exist_ok=True)
p = os.path.abspath(os.path.join(out, 'heatmap_combined_aura.png'))
fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
print('wrote', p)
