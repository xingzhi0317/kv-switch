#!/bin/bash
# 8B->14B: layer-by-layer energy-constrained calibration (fit_cb6.py), then QuALITY 200 q with the calibrated map
cd ~/kvcodesign
source ~/kvbase/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
f=""; for g in 3 4 5 6 7; do ok=1; for i in 1 2 3; do m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1000 ] || ok=0; sleep 3; done; [ $ok = 1 ] && f="$f $g"; done
set -- $f; [ $# -ge 3 ] || { echo "not enough free GPUs: $f"; exit 1; }
G=$1,$2,$3; echo "$(date) calibrate on $G"
(cd ~/kvbase && DIR=8to14 CUDA_VISIBLE_DEVICES=$G python fit_cb6.py > logs/fit_cb6_8to14.log 2>&1) || { echo "CALIB FAILED"; exit 1; }
echo "$(date) eval on $G"
CUDA_VISIBLE_DEVICES=$G DIR=8to14 TASK=quality N=200 METHODS=cb_official \
  MAPPER_CB=$HOME/kvbase/mappers/cachebridge_8to14_ec6 OUT=$HOME/kvcodesign/results/8to14/quality_quality_cbec.json \
  python eval_quality.py > logs_8to14_evalcal.log 2>&1
echo "EVAL EXIT $?"; echo ALLDONECAL
