"""
mpeg_wm/make_clips.py -- pick N bona fide ASVspoof2019 LA eval clips for the benchmark.

ASVspoof2019 LA eval has 67 speakers, so 120 clips cannot be one per speaker:
speakers are taken round-robin (one clip from every speaker, then a second
from every speaker, ...) from a seeded shuffle, each clip >= --min-dur seconds.

RUN ON A LOGIN NODE BEFORE sbatch (needs the protocol + audio on Great Lakes):
    python make_clips.py                     # 120 clips, seed 0 -> clips120.json
"""
import argparse
import json
import os
import sys

import pandas as pd
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "masking"))
from common import ASV_AUDIO, ASV_PROTOCOL  # noqa: E402

CLIPS_JSON = os.path.join(HERE, "clips120.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-dur", type=float, default=3.0)
    ap.add_argument("--protocol", default=ASV_PROTOCOL)
    ap.add_argument("--audio-dir", default=ASV_AUDIO)
    ap.add_argument("--out", default=CLIPS_JSON)
    a = ap.parse_args()

    for p in (a.protocol, a.audio_dir):
        if not os.path.exists(p):
            raise SystemExit(f"not found: {p}  (set ASV_PROTOCOL / ASV_AUDIO)")

    cols = ["speaker", "utt", "system", "attack", "label"]
    df = pd.read_csv(a.protocol, sep=r"\s+", header=None, names=cols)
    df = df[df["label"] == "bonafide"].sample(frac=1.0, random_state=a.seed)
    print(f"bona fide eval utterances: {len(df)}, speakers: {df['speaker'].nunique()}")

    queues = {spk: list(g["utt"]) for spk, g in df.groupby("speaker", sort=False)}
    picked = []
    while len(picked) < a.n and any(queues.values()):
        for spk, utts in queues.items():
            while utts:                                   # next long-enough clip of this speaker
                utt = utts.pop(0)
                path = os.path.join(a.audio_dir, f"{utt}.flac")
                if os.path.exists(path) and sf.info(path).duration >= a.min_dur:
                    picked.append(dict(clip_id=f"asv{len(picked):03d}", speaker=spk, utt=utt,
                                       path=path, duration_s=round(sf.info(path).duration, 3)))
                    break
            if len(picked) == a.n:
                break
    if len(picked) < a.n:
        raise SystemExit(f"only {len(picked)} clips pass the filters")

    json.dump(dict(source="ASVspoof2019 LA eval, bonafide", seed=a.seed, min_dur=a.min_dur,
                   protocol=a.protocol, audio_dir=a.audio_dir, clips=picked),
              open(a.out, "w"), indent=2)
    n_spk = len({c["speaker"] for c in picked})
    d = sorted(c["duration_s"] for c in picked)
    print(f"wrote {a.out}: {len(picked)} clips from {n_spk} speakers, "
          f"duration min {d[0]:.2f} / median {d[len(d) // 2]:.2f} / max {d[-1]:.2f} s")
    print(f"submit with: sbatch --array=0-{len(picked) - 1} bench.sbatch")


if __name__ == "__main__":
    main()
