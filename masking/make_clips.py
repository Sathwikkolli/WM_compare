"""
masking/make_clips.py -- pick the bona fide ASVspoof2019 LA eval clips for E0/E1.

One clip per speaker (so no speaker dominates), at least MIN_DUR seconds long
(1 s detection windows in E1b and PESQ both need some length), seeded shuffle.

RUN ON A LOGIN NODE BEFORE sbatch: the array tasks only read clips.json, so a
bad path fails once here instead of N times on compute nodes.

Usage:
    python make_clips.py                 # 30 clips, seed 0
    python make_clips.py --n 30 --seed 0 --min-dur 3.0
"""
import argparse
import json
import os

import pandas as pd
import soundfile as sf

from common import ASV_AUDIO, ASV_PROTOCOL, CLIPS_JSON


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-dur", type=float, default=3.0)
    ap.add_argument("--protocol", default=ASV_PROTOCOL)
    ap.add_argument("--audio-dir", default=ASV_AUDIO)
    a = ap.parse_args()

    for p in (a.protocol, a.audio_dir):
        if not os.path.exists(p):
            raise SystemExit(f"not found: {p}  (set ASV_PROTOCOL / ASV_AUDIO)")

    cols = ["speaker", "utt", "system", "attack", "label"]
    df = pd.read_csv(a.protocol, sep=r"\s+", header=None, names=cols)
    df = df[df["label"] == "bonafide"].sample(frac=1.0, random_state=a.seed)
    print(f"bona fide eval utterances: {len(df)}, speakers: {df['speaker'].nunique()}")

    picked, seen = [], set()
    for r in df.itertuples():
        if r.speaker in seen:
            continue
        p = os.path.join(a.audio_dir, f"{r.utt}.flac")
        if not os.path.exists(p):
            continue
        dur = sf.info(p).duration
        if dur < a.min_dur:
            continue
        picked.append(dict(clip_id=f"asv{len(picked):02d}", speaker=r.speaker,
                           utt=r.utt, path=p, duration_s=round(dur, 3)))
        seen.add(r.speaker)
        if len(picked) == a.n:
            break
    if len(picked) < a.n:
        raise SystemExit(f"only {len(picked)} clips pass the filters")

    json.dump(dict(source="ASVspoof2019 LA eval, bonafide", seed=a.seed,
                   min_dur=a.min_dur, protocol=a.protocol, audio_dir=a.audio_dir,
                   clips=picked), open(CLIPS_JSON, "w"), indent=2)
    d = [c["duration_s"] for c in picked]
    print(f"wrote {CLIPS_JSON}: {len(picked)} clips, "
          f"duration min {min(d):.2f} / median {sorted(d)[len(d)//2]:.2f} / max {max(d):.2f} s")
    print(f"submit with: sbatch --array=0-{len(picked)-1} e0_e1.sbatch")


if __name__ == "__main__":
    main()
