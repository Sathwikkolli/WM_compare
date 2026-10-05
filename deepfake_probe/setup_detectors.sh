#!/bin/bash
# Fetch the three deepfake detectors into deepfake_probe/third_party/ (gitignored).
# Run ONCE on Great Lakes (login node is fine -- needs internet, no GPU):
#     conda activate wmcompare
#     bash setup_detectors.sh
#
#   1. AASIST   -- clovaai/aasist, official ASVspoof2019-LA weights (ships in repo)
#   2. RawNet2  -- ASVspoof2021 baseline code + official pre-trained DF weights
#   3. XLS-R    -- Gustking/wav2vec2-large-xlsr-deepfake-audio-classification (HF);
#                 downloaded into the HF cache on first use, so only `transformers`
#                 is needed here.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TP="$HERE/third_party"
mkdir -p "$TP"

# --- AASIST ----------------------------------------------------------------- #
if [ ! -f "$TP/aasist/models/weights/AASIST.pth" ]; then
    git clone --depth 1 https://github.com/clovaai/aasist.git "$TP/aasist"
fi

# --- RawNet2 (ASVspoof2021 baseline) ----------------------------------------- #
RN="$TP/rawnet2"
mkdir -p "$RN"
B=https://raw.githubusercontent.com/asvspoof-challenge/2021/main/LA/Baseline-RawNet2
[ -f "$RN/model.py" ]               || curl -sSfL "$B/model.py"               -o "$RN/model.py"
[ -f "$RN/model_config_RawNet.yaml" ] || curl -sSfL "$B/model_config_RawNet.yaml" -o "$RN/model_config_RawNet.yaml"
if [ -z "$(find "$RN" -name '*.pth' -print -quit)" ]; then
    curl -sSfL https://www.asvspoof.org/asvspoof2021/pre_trained_DF_RawNet2.zip -o "$RN/rawnet2_df.zip"
    (cd "$RN" && unzip -o -q rawnet2_df.zip && rm rawnet2_df.zip)
fi

# --- XLS-R (HuggingFace) ----------------------------------------------------- #
python -c "import transformers" 2>/dev/null || pip install "transformers>=4.30"
python - <<'EOF'
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
m = 'Gustking/wav2vec2-large-xlsr-deepfake-audio-classification'
AutoFeatureExtractor.from_pretrained(m); AutoModelForAudioClassification.from_pretrained(m)
print('xlsr cached')
EOF

echo "--- weights found ---"
find "$TP" -name '*.pth' | sed "s|$HERE/||"
