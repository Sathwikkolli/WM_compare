"""
vox_attacks.py  --  VoxWatermark no-box perturbation suite (17 attacks), transcribed
from the official repo (Splitting_Dataset/no_box_funcs.py) so results match the paper.

Each attack is applied at a sweep of strengths. Functions operate on mono float32
numpy at the working sample rate (we use SR_MASTER = 22050 to stay consistent with the
cascade study; VoxWatermark's reference rate is 16 kHz, the only effect is that filter
"ratio of sample rate" cutoffs and the smoothing window scale to 22.05 k instead of 16 k).

Optional dependencies (each attack degrades to SKIP if its dep is missing, so one
missing package never blocks the whole run):
  * transformers + torch  -> encodec   (pip install transformers)
  * julius                -> highpass/lowpass/smooth/echo exact match (else scipy fallback)
  * pydub + ffmpeg        -> dynamic_compression / dynamic_expansion
  * ffmpeg                -> mp3 / aac / opus
  * a noise wav in cascade/noises/ -> background_noise (else SKIP)

VOX_GRID maps attack name -> list of (label, param). apply(name, param, y, sr) -> y2
(returns None if the attack is unavailable -> caller records SKIP).
"""
import os, subprocess, tempfile, warnings
import numpy as np

warnings.filterwarnings('ignore')
NOISE_DIR = os.path.join(os.path.dirname(__file__), 'noises')

# --------------------------------------------------------------------------- #
#  Parameter grids  (verbatim from no_box_funcs.apply_no_box_pert)
# --------------------------------------------------------------------------- #
VOX_GRID = {
    'time_stretch':        [(f'{s}', s)   for s in [0.7, 0.9, 1.1, 1.3, 1.5]],
    'gaussian_noise':      [(f'{s}dB', s) for s in [40, 30, 20, 10, 5]],
    'background_noise':    [(f'{s}dB', s) for s in [40, 30, 20, 10, 5]],
    'opus':                [(f'{b*16}k', b*16) for b in [1, 2, 4, 8, 16, 31]],   # ->16..496k
    'encodec':             [(f'{b}kbps', b) for b in [1.5, 3.0, 6.0, 12.0, 24.0]],
    'quantization':        [(f'{l}lvl', l) for l in [4, 8, 16, 32, 64]],
    'highpass':            [(f'{r}', r) for r in [0.1, 0.2, 0.3, 0.4, 0.5]],
    'lowpass':             [(f'{r}', r) for r in [0.1, 0.2, 0.3, 0.4, 0.5]],
    'smooth':              [(f'w{w}', w) for w in [6, 10, 14, 18, 22]],
    'echo':                [(f'd{d}', d) for d in [0.1, 0.3, 0.5, 0.7, 0.9]],
    'mp3':                 [(f'{b}k', b) for b in [8, 16, 24, 32, 40]],
    'aac':                 [('8k', 8), ('40k', 40)],
    'dynamic_compression': [('t-10_r2', (-10, 2.0)), ('t-30_r8', (-30, 8.0))],
    'dynamic_expansion':   [('t-10_r2', (-10, 2.0)), ('t-30_r8', (-30, 8.0))],
    'inverse_polarity':    [('neg', None)],
    'time_jitter':         [(f's{s}', s) for s in [0.01, 0.5]],
    'phase_shift':         [(f'{s}', s) for s in [1, -1000]],
}
# numeric x-axis value for plotting robustness-vs-strength curves
def strength_x(name, param):
    if name == 'aac':                return float(param)
    if name == 'dynamic_compression' or name == 'dynamic_expansion': return float(param[1])
    if name == 'inverse_polarity':   return 0.0
    try:    return float(param)
    except Exception: return 0.0


# --------------------------------------------------------------------------- #
#  ffmpeg codec round-trip helper
# --------------------------------------------------------------------------- #
def _ff_codec(y, sr, enc_args, ext):
    import soundfile as sf
    d = tempfile.mkdtemp()
    src = os.path.join(d, 'in.wav'); enc = os.path.join(d, 'c.' + ext); out = os.path.join(d, 'out.wav')
    sf.write(src, y.astype('float32'), sr)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', src] + enc_args + [enc], check=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', enc, '-ar', str(sr), '-ac', '1', out], check=True)
    z, _ = sf.read(out)
    return np.asarray(z, dtype='float32')


# --------------------------------------------------------------------------- #
#  Individual attacks (numpy float32 in/out, mono, at sr)
# --------------------------------------------------------------------------- #
def _fix_len(y, n):
    if len(y) < n: return np.pad(y, (0, n - len(y)))
    return y[:n]

def a_time_stretch(y, sr, rate):
    import librosa
    z = librosa.effects.time_stretch(y, rate=rate)
    return _fix_len(z, len(y)).astype('float32')      # repo pads/trims to original length

def a_gaussian(y, sr, snr_db):
    sp = float(np.mean(y**2)); npow = sp / (10 ** (snr_db / 10.0))
    n = np.random.RandomState(0).randn(len(y)).astype('float32') * np.sqrt(npow)
    return (y + n).astype('float32')

def a_background(y, sr, snr_db):
    import soundfile as sf, glob
    if not os.path.isdir(NOISE_DIR):
        return None
    wavs = sorted(glob.glob(os.path.join(NOISE_DIR, '**', '*.wav'), recursive=True))  # any depth
    if not wavs:
        return None
    noise, nsr = sf.read(wavs[0])
    if getattr(noise, 'ndim', 1) > 1: noise = noise.mean(1)
    if nsr != sr:
        import librosa; noise = librosa.resample(noise.astype('float32'), orig_sr=nsr, target_sr=sr)
    if len(noise) < len(y):
        noise = np.tile(noise, len(y) // len(noise) + 1)
    noise = noise[:len(y)].astype('float32')
    sp = float(np.mean(y**2)); npow = float(np.mean(noise**2)) + 1e-12
    scale = np.sqrt(sp / ((10 ** (snr_db / 10.0)) * npow))
    return (y + noise * scale).astype('float32')

def a_opus(y, sr, kbps):  return _ff_codec(y, sr, ['-c:a', 'libopus', '-b:a', f'{kbps}k'], 'opus')
def a_mp3(y, sr, kbps):   return _ff_codec(y, sr, ['-codec:a', 'libmp3lame', '-b:a', f'{kbps}k'], 'mp3')
def a_aac(y, sr, kbps):   return _ff_codec(y, sr, ['-c:a', 'aac', '-b:a', f'{kbps}k'], 'm4a')

_ENC = {}
def a_encodec(y, sr, bandwidth):
    try:
        import torch, librosa
        from transformers import EncodecModel, AutoProcessor
    except Exception:
        return None
    if 'm' not in _ENC:
        _ENC['m'] = EncodecModel.from_pretrained('facebook/encodec_24khz').eval()
        _ENC['p'] = AutoProcessor.from_pretrained('facebook/encodec_24khz')
    model, proc = _ENC['m'], _ENC['p']
    y24 = librosa.resample(y.astype('float32'), orig_sr=sr, target_sr=24000)
    inp = proc(raw_audio=y24, sampling_rate=24000, return_tensors='pt')
    with torch.no_grad():
        enc = model.encode(inp['input_values'], inp['padding_mask'], bandwidth)
        dec = model.decode(enc.audio_codes, enc.audio_scales, inp['padding_mask'])[0]
    z = dec.squeeze().detach().cpu().numpy()
    z = librosa.resample(z.astype('float32'), orig_sr=24000, target_sr=sr)
    return _fix_len(z, len(y)).astype('float32')

def a_quantization(y, sr, levels):
    lo, hi = float(y.min()), float(y.max()); rng = (hi - lo) or 1e-9
    nz = (y - lo) / rng
    q = np.round(nz * (levels - 1)) / (levels - 1)
    return (q * rng + lo).astype('float32')

def _julius():
    try:
        import julius, torch; return julius, torch
    except Exception:
        return None, None

def a_highpass(y, sr, ratio):
    julius, torch = _julius()
    if julius is not None:
        x = torch.from_numpy(y).float().unsqueeze(0)
        return julius.highpass_filter(x, cutoff=ratio).squeeze(0).numpy().astype('float32')
    from scipy.signal import butter, filtfilt          # fallback: ratio is fraction of sr
    b, a = butter(4, min(2 * ratio, 0.99), btype='high')
    return filtfilt(b, a, y).astype('float32')

def a_lowpass(y, sr, ratio):
    julius, torch = _julius()
    if julius is not None:
        x = torch.from_numpy(y).float().unsqueeze(0)
        return julius.lowpass_filter(x, cutoff=ratio).squeeze(0).numpy().astype('float32')
    from scipy.signal import butter, filtfilt
    b, a = butter(4, min(2 * ratio, 0.99), btype='low')
    return filtfilt(b, a, y).astype('float32')

def a_smooth(y, sr, window):
    k = np.ones(int(window), dtype='float32') / int(window)
    return np.convolve(y, k, mode='same').astype('float32')   # uniform moving average

def a_echo(y, sr, duration, volume=0.4):
    n = max(1, int(sr * duration))
    ir = np.zeros(n, dtype='float32'); ir[0] = 1.0; ir[-1] = volume
    z = np.convolve(y, ir, mode='full')
    z = z / (np.max(np.abs(z)) + 1e-12) * (np.max(np.abs(y)) + 1e-12)
    return _fix_len(z, len(y)).astype('float32')

def _pydub_seg(y, sr):
    from pydub import AudioSegment
    i16 = (np.clip(y, -1, 1) * 32767).astype('int16')
    return AudioSegment(i16.tobytes(), sample_width=2, frame_rate=sr, channels=1)

def _seg_to_np(seg):
    return np.array(seg.get_array_of_samples(), dtype='float32') / 32767.0

def a_dyn_comp(y, sr, params):
    try:
        from pydub.effects import compress_dynamic_range
    except Exception:
        return None
    thr, ratio = params
    mx = float(np.max(np.abs(y))) or 1.0
    seg = _pydub_seg(y / mx, sr)
    seg = compress_dynamic_range(seg, threshold=thr, ratio=ratio)
    return (_seg_to_np(seg) * mx).astype('float32')

def a_dyn_exp(y, sr, params):
    # Vectorised expansion approximating pydub's frame loop (the repo's python loop is
    # impractical on a 100s+ clip). Attenuate samples whose short-time RMS is below
    # threshold by (ratio-1)/ratio of how far under they are.
    thr_db, ratio = params
    mx = float(np.max(np.abs(y))) or 1.0
    yn = y / mx
    win = max(1, int(0.005 * sr))
    env = np.sqrt(np.convolve(yn**2, np.ones(win)/win, mode='same') + 1e-12)
    thr = 10 ** (thr_db / 20.0)
    under_db = np.where(env < thr, 20 * np.log10((thr + 1e-9) / (env + 1e-9)), 0.0)
    atten_db = (ratio - 1.0) / ratio * under_db
    gain = 10 ** (-atten_db / 20.0)
    return (yn * gain * mx).astype('float32')

def a_polarity(y, sr, _):  return (-y).astype('float32')

def a_jitter(y, sr, scale):
    x = np.arange(len(y), dtype='float64')
    xn = x + np.random.RandomState(0).normal(0.0, scale, size=len(y))
    return np.interp(xn, x, y).astype('float32')

def a_phase(y, sr, shift):
    n = len(y); pad = np.zeros(abs(shift), dtype='float32')
    if shift > 0:  z = np.concatenate([pad, y[shift:]]) if shift < n else pad
    else:          z = np.concatenate([y[:shift], pad]) if shift != 0 else y
    return _fix_len(z, n).astype('float32')

APPLY = {
    'time_stretch': a_time_stretch, 'gaussian_noise': a_gaussian, 'background_noise': a_background,
    'opus': a_opus, 'encodec': a_encodec, 'quantization': a_quantization,
    'highpass': a_highpass, 'lowpass': a_lowpass, 'smooth': a_smooth, 'echo': a_echo,
    'mp3': a_mp3, 'aac': a_aac, 'dynamic_compression': a_dyn_comp, 'dynamic_expansion': a_dyn_exp,
    'inverse_polarity': a_polarity, 'time_jitter': a_jitter, 'phase_shift': a_phase,
}

TEST_SET = {'time_stretch', 'encodec', 'background_noise'}   # paper's reported pipeline

def apply(name, param, y, sr):
    """Apply one attack at one strength. Returns numpy y2, or None if unavailable."""
    fn = APPLY[name]
    return fn(y, sr, param)
