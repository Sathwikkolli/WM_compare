"""
detectors.py  --  three off-the-shelf speech deepfake (anti-spoofing) detectors.

Every detector exposes   p_spoof(y16) -> (mean P(spoof), [per-window P(spoof)])
where y16 is mono float32 at 16 kHz.

All three were trained on 16 kHz clips of ~4 s, so long clips are cut into
non-overlapping 64600-sample windows (4.04 s, AASIST/RawNet2's nb_samp) and the
per-window probabilities are averaged. A clip shorter than one window is
repeat-padded exactly as AASIST's own data_utils.pad() does.

Label convention: AASIST and RawNet2 both use class 1 = bona fide (ASVspoof),
so P(spoof) = 1 - softmax[:, 1]. The XLS-R model's index is read from its
config.id2label rather than assumed.
"""
import os, sys, glob, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TP   = os.path.join(HERE, 'third_party')
WIN  = 64600
XLSR_ID = 'Gustking/wav2vec2-large-xlsr-deepfake-audio-classification'


def windows(y, win=WIN):
    y = np.asarray(y, dtype='float32')
    if len(y) < win:
        return [np.tile(y, int(win / max(len(y), 1)) + 1)[:win]]
    return [y[i * win:(i + 1) * win] for i in range(len(y) // win)]


def _strip_module(sd):
    """Checkpoints saved from nn.DataParallel prefix every key with 'module.'."""
    return {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}


class _Base:
    name = None
    def __init__(self, device=None):
        import torch
        self.torch = torch
        self.dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
    def _probs(self, batch):            # batch: [B, WIN] tensor -> P(spoof) [B]
        raise NotImplementedError
    def p_spoof(self, y16, bs=8):
        torch = self.torch
        ws = windows(y16)
        out = []
        with torch.no_grad():
            for i in range(0, len(ws), bs):
                x = torch.from_numpy(np.stack(ws[i:i + bs])).to(self.dev)
                out.extend(self._probs(x).cpu().numpy().tolist())
        return float(np.mean(out)), out


class AASIST(_Base):
    name = 'AASIST'
    def load(self):
        torch = self.torch
        repo = os.path.join(TP, 'aasist')
        sys.path.insert(0, repo)
        from models.AASIST import Model
        conf = json.load(open(os.path.join(repo, 'config', 'AASIST.conf')))
        m = Model(conf['model_config']).to(self.dev)
        sd = torch.load(os.path.join(repo, 'models', 'weights', 'AASIST.pth'),
                        map_location=self.dev)
        m.load_state_dict(_strip_module(sd)); m.eval()
        sys.path.remove(repo)
        sys.modules.pop('models', None)          # don't shadow other 'models' pkgs
        self.model = m
        return self
    def _probs(self, x):
        _, logits = self.model(x)
        return 1.0 - self.torch.softmax(logits, dim=1)[:, 1]


class RawNet2(_Base):
    name = 'RawNet2'
    def load(self):
        torch, yaml = self.torch, __import__('yaml')
        d = os.path.join(TP, 'rawnet2')
        sys.path.insert(0, d)
        from model import RawNet
        cfg = yaml.safe_load(open(os.path.join(d, 'model_config_RawNet.yaml')))
        m = RawNet(cfg['model'], self.dev).to(self.dev)
        pth = sorted(glob.glob(os.path.join(d, '**', '*.pth'), recursive=True))
        if not pth:
            raise FileNotFoundError(f'no RawNet2 .pth under {d} -- run setup_detectors.sh')
        m.load_state_dict(_strip_module(torch.load(pth[0], map_location=self.dev)))
        m.eval()
        sys.path.remove(d)
        sys.modules.pop('model', None)
        self.model, self.weights = m, os.path.relpath(pth[0], HERE)
        return self
    def _probs(self, x):
        logp = self.model(x)                     # log-softmax, [B, 2]
        return 1.0 - self.torch.exp(logp)[:, 1]


class XLSR(_Base):
    name = 'XLS-R'
    def load(self):
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        self.fe = AutoFeatureExtractor.from_pretrained(XLSR_ID)
        m = AutoModelForAudioClassification.from_pretrained(XLSR_ID).to(self.dev).eval()
        labels = {int(k): v.lower() for k, v in m.config.id2label.items()}
        fake = [k for k, v in labels.items() if 'fake' in v or 'spoof' in v]
        if len(fake) != 1:
            raise RuntimeError(f'cannot find the fake class in id2label={labels}')
        self.fake_idx, self.model = fake[0], m
        return self
    def _probs(self, x):
        feats = self.fe(list(x.cpu().numpy()), sampling_rate=16000,
                        return_tensors='pt', padding=True)
        logits = self.model(feats['input_values'].to(self.dev)).logits
        return self.torch.softmax(logits, dim=1)[:, self.fake_idx]


DETECTORS = (AASIST, RawNet2, XLSR)

def load_all(device=None):
    return [cls(device).load() for cls in DETECTORS]


if __name__ == '__main__':
    # Smoke test: white noise and silence should run without error; prints P(spoof).
    rng = np.random.default_rng(0)
    for det in load_all():
        for tag, y in (('noise', 0.05 * rng.standard_normal(16000 * 6)),
                       ('silence', np.zeros(16000 * 2))):
            p, per = det.p_spoof(y.astype('float32'))
            print(f'{det.name:8s} {tag:8s} P(spoof)={p:.3f}  windows={len(per)}')
