"""
aura_adapter.py  --  AURA watermark adapter for the VoxWatermark benchmark.

Exposes the same interface as the AudioSeal/AWARE/Timbre adapters in
cascade_lib.py:  load() / embed(y22) / detect(y22), where y22 is mono float32 at
SR_MASTER (22050 Hz).  AURA is a fixed 2-second / 48 kHz / 32-bit model with no
sync layer, so internally we:

  embed : resample 22050 -> 48000, tile non-overlapping 2 s (96000-sample)
          windows, embed the same 32-bit message in each, concat, resample back.
  detect: resample -> 48000, decode every 2 s window, majority-vote the bits,
          report mean per-window bit accuracy as both `acc` and `conf`.

Keeping the master at 22050 (like every other tool) means AURA is attacked by the
*identical* VoxWatermark perturbations — a fair apples-to-apples comparison.
"""
import os
import sys
from pathlib import Path

import numpy as np

# canonical benchmark rate (kept in sync with cascade_lib.SR_MASTER)
SR_MASTER = 22050
SR_AURA = 48_000
WIN = 96_000          # AURA window = 2 s @ 48 kHz
N_BITS = 32

# Fixed 32-bit ground-truth message (reproducible)
AURA_BITS = '01010001000010000101000100001000'

# default checkpoint (override with $AURA_CKPT)
_DEFAULT_CKPT = os.environ.get(
    'AURA_CKPT',
    os.path.expanduser('~/projects/aura_watermark/checkpoints/run_002/step_0200000_final.pt'),
)


def _find_aura_repo():
    for cand in (Path(__file__).resolve().parents[2] / 'aura_watermark',
                 Path.home() / 'projects' / 'aura_watermark'):
        cand = cand.resolve()
        if (cand / 'aura_watermark').is_dir():
            return cand
    raise RuntimeError('aura_watermark repo not found (looked in ../../ and ~/projects)')


class AuraAdapter:
    code = 'aura'
    truth = AURA_BITS

    def __init__(self, checkpoint=None):
        self.checkpoint = checkpoint or _DEFAULT_CKPT
        self._emb = None
        self._det = None
        self._dev = None
        self._msg = None

    def load(self):
        import torch
        sys.path.insert(0, str(_find_aura_repo()))
        from infer import load_model  # AURA's own inference helpers
        self._torch = torch
        self._dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._emb, self._det, self._cfg = load_model(self.checkpoint, self._dev)
        self._msg = torch.tensor([int(b) for b in self.truth], dtype=torch.long)

    # -- helpers ------------------------------------------------------------
    def _to_aura(self, y22):
        import librosa
        y = librosa.resample(np.asarray(y22, 'float32'), orig_sr=SR_MASTER, target_sr=SR_AURA)
        return y.astype('float32')

    def _to_master(self, y48, n_target):
        import librosa
        y = librosa.resample(np.asarray(y48, 'float32'), orig_sr=SR_AURA, target_sr=SR_MASTER)
        if len(y) < n_target:
            y = np.pad(y, (0, n_target - len(y)))
        return y[:n_target].astype('float32')

    # -- interface ----------------------------------------------------------
    def embed(self, y22):
        from infer import embed_watermark
        torch = self._torch
        n_master = len(y22)
        y48 = self._to_aura(y22)
        out = y48.copy()
        n_win = max(1, len(y48) // WIN)
        for i in range(n_win):
            s = i * WIN
            chunk = torch.from_numpy(y48[s:s + WIN]).float().unsqueeze(0).to(self._dev)  # [1, WIN]
            wm = embed_watermark(self._emb, chunk, self._msg.to(self._dev))
            out[s:s + WIN] = wm.detach().cpu().numpy().ravel()
        return self._to_master(out, n_master)

    def detect(self, y22):
        from infer import detect_watermark
        torch = self._torch
        y48 = self._to_aura(y22)
        n_win = max(1, len(y48) // WIN)
        tgt = np.array([int(b) for b in self.truth])
        accs, votes = [], np.zeros(N_BITS, dtype=np.int64)
        for i in range(n_win):
            s = i * WIN
            chunk = y48[s:s + WIN]
            if len(chunk) < WIN:
                chunk = np.pad(chunk, (0, WIN - len(chunk)))
            x = torch.from_numpy(chunk).float().unsqueeze(0).to(self._dev)  # [1, WIN]
            _, bits = detect_watermark(self._emb, self._det, x)
            b = bits.long().cpu().numpy().ravel()[:N_BITS]
            accs.append(float((b == tgt).mean()))
            votes += b
        decoded = (votes >= (n_win / 2)).astype(int)
        bitstr = ''.join(map(str, decoded.tolist()))
        acc = float(np.mean(accs))
        return acc, bitstr, acc      # (conf, bits, acc) — no separate prob, reuse acc
