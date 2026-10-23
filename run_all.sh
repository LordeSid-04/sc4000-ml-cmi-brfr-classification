#!/usr/bin/env bash
# Full experiment suite. Run on a GPU box (Kaggle T4/P100 notebook: !bash run_all.sh). Needs data/processed/features.npz
# and canonical/ (python run_pipeline.py --data <competition zip>). Folds run as parallel processes on the one GPU.
set -e
python make_aug.py --copies 6                                    # once: augmented copies (CPU, ~minutes)

nn() {   # nn <name> <core> [extra args]  -> trains the all-sensor and motion-only variants, 5 folds in parallel
  n=$1; c=$2; shift 2
  for v in full imu; do
    for k in 0 1 2 3 4; do python train_nn.py --name $n --core $c --variant $v --folds $k --seeds 0 1 2 "$@" & done; wait
  done
}
nn cnn_gru cnn_gru                                                # main model
nn cnn cnn; nn gru gru; nn transformer transformer                # sequence baselines
for m in lgbm xgb rf mlp; do python train_trees.py --model $m --seeds 0 1 2; done   # summary-feature models

# ablations of the main model (each = new name, 3 seeds)
nn abl_noaug cnn_gru --aug 0
nn abl_nophase cnn_gru --phase 0
nn abl_notofreg cnn_gru --drop-sub _r0 _r1 _r2

python evaluate.py table rf lgbm xgb mlp cnn gru transformer cnn_gru
python evaluate.py hybrid cnn_gru lgbm
python evaluate.py table cnn_gru abl_noaug abl_nophase abl_notofreg 'cnn_gru:blank'
