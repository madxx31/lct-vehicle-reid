#!/usr/bin/env bash
# Full training pipeline. Run from anywhere once data/ is in place and `uv sync` is done:
#   bash train/train.sh
set -euo pipefail
cd "$(dirname "$0")/.."

T="uv run python -m train.encoder.train"

# encoder trained on folds 0,3,4 -> oof_preds/embeddings12oof.npy
$T "data.eval_folds=[1,2]"
# encoder trained on folds 0,1,2 -> oof_preds/embeddings34oof.npy
$T "data.eval_folds=[3,4]"
# encoder trained on all of train.csv -> models/vehicle_encoder.pt
$T "data.eval_folds=[]"
# confidence model fitted on the oof embeddings -> models/confidence_model.cbm
uv run python -m train.confidence_model
