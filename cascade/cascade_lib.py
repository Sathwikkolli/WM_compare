"""
cascade_lib.py  --  shared library for the multi-watermark cascade benchmark.

Holds three things:
  1. Tool ADAPTERS   : load / embed / detect for AudioSeal, AWARE, Timbre
  2. ATTACKS         : signal-processing attacks (ffmpeg + numpy), parametric on SR
  3. METRICS         : bit-accuracy + audio-quality (PESQ, SNR, SI-SNR, STOI)

Design notes
------------
* Everything is canonicalised at SR_MASTER = 22050 Hz (Timbre's native rate, the
  highest of the three). AudioSeal / AWARE run at 16 kHz, so when they embed or
  detect we resample 22050 -> 16000, do the work, and (for embedding) resample the
  watermarked result back to 22050. Resampling is cheap and survivable (the repo's
  own `resample_8k` test shows bit_acc 1.0 for all three) but we still measure its
  cost with a dedicated control file in run_cascade.py.
* Models are loaded lazily and cached, so a single process can embed/detect with
  all three tools sequentially (the cascade is inherently sequential).
* Timbre needs its repo as cwd + on sys.path; we wrap those ops in `chdir()`.

Spots that depend on the exact submodule code are tagged  # VERIFY .
Run `python run_cascade.py --selftest` first -- it solo embeds+detects each tool
on the master clip and fails fast (seconds) if any adapter is wrong.
"""

import os, sys, math, contextlib, subprocess, types
# --- Disable torch.compile / TorchInductor BEFORE torch is ever imported. ---
# The cluster's system g++ rejects PyTorch's `-std=c++20`, so any JIT C++ kernel
# build crashes (this is why the existing sbatch sets TORCHDYNAMO_DISABLE=1).
# Setting it here makes the harness robust no matter how it's launched.
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')
os.environ.setdefault('NO_TORCH_COMPILE', '1')
import numpy as np


def _stub_module(name):
    """Register a fake module whose every attribute is a no-op passthrough class.
    Used for `audiomentations`, which Timbre imports for *training-time* audio
    augmentations we never invoke at inference (decoder.robust=False)."""
    if name in sys.modules:
        return sys.modules[name]
    m = types.ModuleType(name)
    class _Any:
        def __init__(self, *a, **k): pass
        def __call__(self, *a, **k): return a[0] if a else None
    def _getattr(attr):                       # PEP 562 module-level __getattr__
        if attr.startswith('__') and attr.endswith('__'):
            raise AttributeError(attr)        # don't fake dunders (__file__, __path__...)
        return _Any
    m.__getattr__ = _getattr
    sys.modules[name] = m
    return m


_ORIG_PAD_CENTER = None

def _compat_pad_center(data, size=None, axis=-1, **kw):
    """Accept the old positional `pad_center(window, n_fft)` call and forward to
    the real (keyword-only) librosa implementation captured in _ORIG_PAD_CENTER."""
    orig = _ORIG_PAD_CENTER
    if orig is None:
        import librosa.util as U; orig = U.pad_center
    if size is None:
        return orig(data, **kw)
    try:
        return orig(data, size=size, axis=axis, **kw)   # modern keyword-only API
    except TypeError:
        return orig(data, size, axis, **kw)             # very old positional API
_compat_pad_center._cascade_shim = True

_ORIG_MEL = None
def _compat_mel(*args, **kw):
    """Accept Timbre's positional `mel(sr, n_fft, n_mels, fmin, fmax)` and forward
    to newer librosa.filters.mel, which made every argument keyword-only."""
    orig = _ORIG_MEL
    if orig is None:
        import librosa; orig = librosa.filters.mel
    names = ['sr', 'n_fft', 'n_mels', 'fmin', 'fmax', 'htk', 'norm', 'dtype']
    for i, a in enumerate(args):
        kw.setdefault(names[i], a)
    return orig(**kw)
_compat_mel._cascade_shim = True

def _patch_librosa_pad_center():
    """Bridge Timbre's positional pad_center call against newer librosa.
    Patches librosa.util globally AND (crucially) rebinds the name inside any
    already-imported `frequency.py`, which did `from librosa.util import pad_center`
    and therefore holds its own copy that a librosa-level patch can't reach."""
    global _ORIG_PAD_CENTER, _ORIG_MEL
    try:
        import librosa, librosa.util as U
    except Exception:
        return
    if _ORIG_PAD_CENTER is None and not getattr(U.pad_center, '_cascade_shim', False):
        _ORIG_PAD_CENTER = U.pad_center
    if _ORIG_MEL is None and not getattr(librosa.filters.mel, '_cascade_shim', False):
        _ORIG_MEL = librosa.filters.mel
    U.pad_center = _compat_pad_center
    librosa.util.pad_center = _compat_pad_center
    librosa.filters.mel = _compat_mel
    # rebind in any module that already imported the old positional names
    for mod in list(sys.modules.values()):
        f = getattr(mod, '__file__', None)
        if isinstance(f, str) and f.endswith('frequency.py'):
            if hasattr(mod, 'pad_center'):     mod.pad_center = _compat_pad_center
            if hasattr(mod, 'librosa_mel_fn'): mod.librosa_mel_fn = _compat_mel
            if hasattr(mod, 'mel'):            mod.mel = _compat_mel

# --------------------------------------------------------------------------- #
#  Paths / constants
# --------------------------------------------------------------------------- #
BASE        = os.path.expanduser('~/wm_compare')
AUDIO       = os.path.join(BASE, 'audio')
TIMBRE_REPO = os.path.join(BASE, 'TimbreWatermarking', 'watermarking_model')

SR_MASTER = 22050      # canonical working rate
SR_16K    = 16000      # AudioSeal / AWARE native rate

# Fixed ground-truth messages (must match what each tool actually embeds)
AUDIOSEAL_BITS = '0110111111100100'                                   # 16 bits
AWARE_BITS     = open(os.path.join(AUDIO, 'aware_bits.txt')).read().strip()  # 20 bits

def _timbre_bits():
    """Ground-truth Timbre message = first line of wmpool.txt (a python list)."""
    p = os.path.join(TIMBRE_REPO, 'results', 'wmpool.txt')
    wm = eval(open(p).read().strip().splitlines()[0])    # e.g. [1,1,0,1,...]
    return ''.join(map(str, wm)), wm

# Tool registry: short code -> (display name, native SR, payload bit length getter)
TOOLS = ('audioseal', 'aware', 'timbre')
TOOL_NAME = {'audioseal': 'AudioSeal', 'aware': 'AWARE', 'timbre': 'Timbre'}


# --------------------------------------------------------------------------- #
#  Small helpers
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def chdir(path):
    """Temporarily cd into `path` (needed for Timbre's relative config loads)."""
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)

def read_wav(path, sr=SR_MASTER):
    """Load mono float32 at `sr`."""
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype('float32')

def write_wav(path, y, sr=SR_MASTER):
    import soundfile as sf
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, np.asarray(y, dtype='float32'), sr, subtype='PCM_16')
    return path

def resample(y, sr_from, sr_to):
    if sr_from == sr_to:
        return np.asarray(y, dtype='float32')
    import librosa
    return librosa.resample(np.asarray(y, dtype='float32'),
                            orig_sr=sr_from, target_sr=sr_to).astype('float32')

def bit_acc(decoded_bits, truth_bits):
    """Fraction of matching bits over the length of the ground-truth message."""
    n = len(truth_bits)
    if not n:
        return float('nan')
    return sum(a == b for a, b in zip(decoded_bits, truth_bits)) / n


# --------------------------------------------------------------------------- #
#  ADAPTERS -- each tool exposes load() / embed(y22, ...) / detect(y22)
#  y22 is always mono float32 at SR_MASTER (22050). Adapters handle resampling.
# --------------------------------------------------------------------------- #
class AudioSealAdapter:
    code = 'audioseal'; truth = AUDIOSEAL_BITS
    def __init__(self):
        self._gen = None; self._det = None
    def load(self):
        import torch
        try:                                    # belt-and-suspenders vs torch.compile
            import torch._dynamo as _dyn
            _dyn.config.suppress_errors = True
            _dyn.disable()
        except Exception:
            pass
        from audioseal import AudioSeal
        self._gen = AudioSeal.load_generator('audioseal_wm_16bits')
        self._det = AudioSeal.load_detector('audioseal_detector_16bits'); self._det.eval()
        self._torch = torch
    def embed(self, y22):
        torch = self._torch
        y16 = resample(y22, SR_MASTER, SR_16K)
        x = torch.from_numpy(y16).unsqueeze(0).unsqueeze(0)            # [1,1,N]
        msg = torch.tensor([[int(b) for b in self.truth]], dtype=torch.int32)  # [1,16]
        with torch.no_grad():
            wm = self._gen(x, sample_rate=SR_16K, message=msg, alpha=1.0)  # VERIFY call
        y16w = wm.squeeze(0).squeeze(0).cpu().numpy().astype('float32')
        return resample(y16w, SR_16K, SR_MASTER)                      # back to master
    def detect(self, y22):
        torch = self._torch
        y16 = resample(y22, SR_MASTER, SR_16K)
        x = torch.from_numpy(y16).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            prob, msg = self._det.detect_watermark(x, sample_rate=SR_16K)
        bits = ''.join(map(str, (msg.squeeze() > 0.5).int().tolist()))
        return float(prob), bits, bit_acc(bits, self.truth)


class AwareAdapter:
    code = 'aware'; truth = AWARE_BITS
    def __init__(self):
        self._emb = None; self._det = None
    def load(self):
        src = os.path.join(BASE, 'aware', 'src')   # src-layout package, not pip-installed
        if os.path.isdir(src) and src not in sys.path:
            sys.path.insert(0, src)
        from aware.utils.models import load
        self._emb, self._det = load(name='AWARE')
    def embed(self, y22):
        from aware.service import embed_watermark
        y16 = resample(y22, SR_MASTER, SR_16K).astype('float32')
        bits = np.array([int(b) for b in self.truth], dtype=np.int64)
        y16w = embed_watermark(y16, SR_16K, bits, self._emb)          # VERIFY call
        y16w = np.asarray(y16w, dtype='float32')
        return resample(y16w, SR_16K, SR_MASTER)
    def detect(self, y22):
        from aware.service import detect_watermark
        y16 = resample(y22, SR_MASTER, SR_16K).astype('float32')
        pat, conf = detect_watermark(y16, SR_16K, self._det)
        bits = ''.join(map(str, np.asarray(pat).astype(int).ravel()[:len(self.truth)].tolist()))
        return float(conf), bits, bit_acc(bits, self.truth)


class TimbreAdapter:
    code = 'timbre'
    def __init__(self):
        self._enc = None; self._dec = None; self._dev = None
        self.truth, self._wm = _timbre_bits()
        self.NB = len(self._wm)
    def load(self):
        import torch, yaml
        self._torch = torch
        _stub_module('audiomentations')          # not installed; only used at train time
        _patch_librosa_pad_center()              # bridge old positional pad_center call
        with chdir(TIMBRE_REPO):
            sys.path.insert(0, TIMBRE_REPO)
            sys.path.insert(0, os.path.join(TIMBRE_REPO, 'distortions'))
            from model.conv2_mel_modules import Encoder, Decoder
            _patch_librosa_pad_center()          # rebind pad_center inside frequency.py
            from distortions import frequency as _freq   # ensure module is reachable
            _freq.pad_center = _compat_pad_center
            if hasattr(_freq, 'librosa_mel_fn'): _freq.librosa_mel_fn = _compat_mel
            if hasattr(_freq, 'mel'):            _freq.mel = _compat_mel
            pc = yaml.load(open('config/process.yaml'), Loader=yaml.FullLoader)
            mc = yaml.load(open('config/model.yaml'),   Loader=yaml.FullLoader)
            tc = yaml.load(open('config/train.yaml'),   Loader=yaml.FullLoader)
            win  = pc['audio']['win_len']; emb = mc['dim']['embedding']
            mlen = tc['watermark']['length']
            dev  = 'cuda' if torch.cuda.is_available() else 'cpu'
            enc = Encoder(pc, mc, mlen, win, emb,
                          nlayers_encoder=mc['layer']['nlayers_encoder'],
                          attention_heads=mc['layer']['attention_heads_encoder']).to(dev)
            dec = Decoder(pc, mc, mlen, win, emb,
                          nlayers_decoder=mc['layer']['nlayers_decoder'],
                          attention_heads=mc['layer']['attention_heads_decoder']).to(dev)
            ckd = 'results/ckpt/pth'
            ck  = sorted(os.listdir(ckd), key=lambda x: os.path.getmtime(os.path.join(ckd, x)))[-1]
            m = torch.load(os.path.join(ckd, ck), map_location=dev)
            enc.load_state_dict(m['encoder'])                 # VERIFY key
            dec.load_state_dict(m['decoder'], strict=False)
            enc.eval(); dec.eval(); dec.robust = False
            self._enc, self._dec, self._dev = enc, dec, dev
    def embed(self, y22):
        torch = self._torch
        x = torch.from_numpy(np.asarray(y22, 'float32')).unsqueeze(0).unsqueeze(0).to(self._dev)  # [1,1,N]
        msg = torch.from_numpy(np.array([[self._wm]])).float().to(self._dev) * 2 - 1               # [1,1,NB] in +-1
        with chdir(TIMBRE_REPO), torch.no_grad():
            encoded, _carrier = self._enc.test_forward(x, msg)        # VERIFY returns (encoded, carrier)
        return encoded.squeeze(0).squeeze(0).detach().cpu().numpy().astype('float32')
    def detect(self, y22):
        torch = self._torch
        x = torch.from_numpy(np.asarray(y22, 'float32')).unsqueeze(0).unsqueeze(0).to(self._dev)
        with chdir(TIMBRE_REPO), torch.no_grad():
            decoded = self._dec.test_forward(x)
        pred = (decoded.squeeze().detach().cpu().numpy().ravel()[:self.NB] >= 0).astype(int)
        bits = ''.join(map(str, pred.tolist()))
        acc  = bit_acc(bits, self.truth)
        return acc, bits, acc        # Timbre has no separate prob -> use bit_acc as conf


_ADAPTERS = {}
def get_adapter(code):
    """Return a loaded, cached adapter for a tool code."""
    if code not in _ADAPTERS:
        a = {'audioseal': AudioSealAdapter,
             'aware':     AwareAdapter,
             'timbre':    TimbreAdapter}[code]()
        a.load()
        _ADAPTERS[code] = a
    return _ADAPTERS[code]


# --------------------------------------------------------------------------- #
#  ATTACKS -- operate on a wav file at SR_MASTER, return path to attacked wav.
#  Each entry: name -> (callable(src_path, work_dir) -> out_path, do_quality)
# --------------------------------------------------------------------------- #
def _ff(src, out, af=None, sr=SR_MASTER):
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', src]
    if af:
        cmd += ['-af', af]
    cmd += ['-ar', str(sr), '-ac', '1', out]
    subprocess.run(cmd, check=True); return out

def _codec(src, enc_path, args, sr=SR_MASTER):
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', src] + args + [enc_path], check=True)
    out = enc_path + '.wav'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', enc_path, '-ar', str(sr), '-ac', '1', out], check=True)
    return out

def _seg(src, out, fn, sr=SR_MASTER):
    import soundfile as sf
    y = read_wav(src, sr)
    sf.write(out, fn(y).astype('float32'), sr, subtype='PCM_16'); return out

def _noise(src, out, snr_db, sr=SR_MASTER):
    import soundfile as sf
    y = read_wav(src, sr); rms = float(np.sqrt((y**2).mean()))
    p = rms / (10 ** (snr_db / 20.0))
    n = np.random.RandomState(0).randn(len(y)).astype('float32') * p
    sf.write(out, (y + n).astype('float32'), sr, subtype='PCM_16'); return out

def _quant(src, out, bits=8, sr=SR_MASTER):
    import soundfile as sf
    y = read_wav(src, sr); L = 2 ** (bits - 1)
    sf.write(out, (np.round(y * L) / L).astype('float32'), sr, subtype='PCM_16'); return out

def _suppress(src, out, sr=SR_MASTER):
    import soundfile as sf
    y = read_wav(src, sr).copy(); rs = np.random.RandomState(0); win = int(0.01 * sr)
    for _ in range(50):
        i = rs.randint(0, len(y) - win); y[i:i + win] = 0.0
    sf.write(out, y.astype('float32'), sr, subtype='PCM_16'); return out

# do_quality=False for attacks that change length/pitch (PESQ alignment meaningless)
def build_attacks(sr=SR_MASTER):
    P = lambda d, n: os.path.join(d, n)
    return {
        'baseline':      (lambda s, d: s, True),
        'mp3_128':       (lambda s, d: _codec(s, P(d,'m128.mp3'), ['-codec:a','libmp3lame','-b:a','128k'], sr), True),
        'mp3_64':        (lambda s, d: _codec(s, P(d,'m64.mp3'),  ['-codec:a','libmp3lame','-b:a','64k'],  sr), True),
        'aac_128':       (lambda s, d: _codec(s, P(d,'a128.m4a'), ['-c:a','aac','-b:a','128k'], sr), True),
        'opus_64':       (lambda s, d: _codec(s, P(d,'o64.opus'), ['-c:a','libopus','-b:a','64k'], sr), True),
        'resample_8k':   (lambda s, d: _codec(s, P(d,'r8.wav'),   ['-ar','8000'], sr), True),
        'lowpass_4k':    (lambda s, d: _ff(s, P(d,'lp.wav'), 'lowpass=f=4000', sr), True),
        'highpass_500':  (lambda s, d: _ff(s, P(d,'hp.wav'), 'highpass=f=500', sr), True),
        'volume_0.5':    (lambda s, d: _ff(s, P(d,'vol.wav'), 'volume=0.5', sr), True),
        'quantize_8bit': (lambda s, d: _quant(s, P(d,'q8.wav'), 8, sr), True),
        'echo':          (lambda s, d: _ff(s, P(d,'ec.wav'), 'aecho=0.8:0.88:60:0.4', sr), True),
        'sample_suppress':(lambda s, d: _suppress(s, P(d,'ss.wav'), sr), True),
        'noise_20db':    (lambda s, d: _noise(s, P(d,'n20.wav'), 20, sr), True),
        'noise_10db':    (lambda s, d: _noise(s, P(d,'n10.wav'), 10, sr), True),
        'noise_5db':     (lambda s, d: _noise(s, P(d,'n5.wav'),   5, sr), True),
        'tempo_up_10':   (lambda s, d: _ff(s, P(d,'tu.wav'), 'atempo=1.1', sr), False),
        'tempo_down_10': (lambda s, d: _ff(s, P(d,'td.wav'), 'atempo=0.9', sr), False),
        'denoise':       (lambda s, d: _ff(s, P(d,'dn.wav'), 'afftdn=nr=20', sr), True),
        'crop_30s':      (lambda s, d: _seg(s, P(d,'c30.wav'), lambda w: w[40*sr:70*sr], sr), False),
        'crop_10s':      (lambda s, d: _seg(s, P(d,'c10.wav'), lambda w: w[40*sr:50*sr], sr), False),
        'crop_5s':       (lambda s, d: _seg(s, P(d,'c5.wav'),  lambda w: w[40*sr:45*sr], sr), False),
        'crop_3s':       (lambda s, d: _seg(s, P(d,'c3.wav'),  lambda w: w[40*sr:43*sr], sr), False),
        'pitch_up':      (lambda s, d: _ff(s, P(d,'pi.wav'),
                          'asetrate=%d,aresample=%d,atempo=0.9443' % (int(round(sr*1.059)), sr), sr), False),
    }

ATTACK_ORDER = ['baseline','mp3_128','mp3_64','aac_128','opus_64','resample_8k',
                'lowpass_4k','highpass_500','volume_0.5','quantize_8bit','echo',
                'sample_suppress','noise_20db','noise_10db','noise_5db',
                'tempo_up_10','tempo_down_10','denoise',
                'crop_30s','crop_10s','crop_5s','crop_3s','pitch_up']


# --------------------------------------------------------------------------- #
#  METRICS -- audio quality of a degraded/watermarked file vs the clean ref.
#  All computed at 16 kHz (PESQ wideband requirement); ref/deg trimmed to equal len.
# --------------------------------------------------------------------------- #
def quality_metrics(clean_path, deg_path):
    """Return dict with pesq, snr_db, si_snr_db, stoi (any may be None on failure)."""
    out = {'pesq': None, 'snr_db': None, 'si_snr_db': None, 'stoi': None}
    try:
        ref = read_wav(clean_path, SR_16K); deg = read_wav(deg_path, SR_16K)
        n = min(len(ref), len(deg))
        if n < SR_16K:                     # < 1s overlap -> skip (e.g. heavy crop)
            return out
        ref, deg = ref[:n], deg[:n]
    except Exception:
        return out
    # PESQ (wideband)
    try:
        from pesq import pesq
        v = pesq(SR_16K, ref, deg, 'wb'); out['pesq'] = round(float(v), 3)
    except Exception: pass
    # SNR
    try:
        noise = deg - ref
        s = float(np.sum(ref**2)); e = float(np.sum(noise**2)) + 1e-12
        out['snr_db'] = round(10*np.log10(s/e), 2) if s > 0 else None
    except Exception: pass
    # SI-SNR (scale invariant)
    try:
        r = ref - ref.mean(); d = deg - deg.mean()
        alpha = float(np.dot(d, r) / (np.dot(r, r) + 1e-12))
        proj = alpha * r; noise = d - proj
        out['si_snr_db'] = round(10*np.log10((np.sum(proj**2)+1e-12)/(np.sum(noise**2)+1e-12)), 2)
    except Exception: pass
    # STOI (intelligibility)
    try:
        from pystoi import stoi
        out['stoi'] = round(float(stoi(ref, deg, SR_16K, extended=False)), 3)
    except Exception: pass
    return out
