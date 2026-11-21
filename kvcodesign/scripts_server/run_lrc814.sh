#!/bin/bash
# low-rank 2048 + 1% calibration on Qwen3-8B -> 14B, then QuALITY (paper layout) and HotpotQA-E, 200 q each (3 GPUs)
cd ~/kvcodesign
source ~/kvbase/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=$HOME/kvcodesign/results/8to14
freegpus() { local f=""; for g in 4 5 6 7 0 2 3 1; do ok=1; for i in 1 2 3; do m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1000 ] || ok=0; sleep 1; done; [ $ok = 1 ] && f="$f $g"; done; echo $f; }
get() { while true; do set -- $(freegpus); [ $# -ge 5 ] && break; sleep 60; done; echo $@ | tr " " "
" | head -3 | paste -sd, ; }   # start only if 2 GPUs stay idle afterwards
G=$(get); echo "$(date) calibrate on $G"
RK=2048 RV=2048 FRAC=0.01 TAG=f1 CUDA_VISIBLE_DEVICES=$G python fit_lrc_big.py > logs_lrc814_fit.log 2>&1; rc=$?
[ $rc = 0 ] || { echo "CALIB FAILED rc=$rc"; exit 1; }
for spec in "quality_cb lrcq" "hotpotqa_e lrch"; do
  set -- $spec
  G=$(get); echo "$(date) eval $1 on $G"
  CUDA_VISIBLE_DEVICES=$G DIR=8to14 TASK=$1 N=200 METHODS=lrc_k2048_v2048_f1 OUT=$R/quality_$1_$2.json python eval_quality.py > logs_lrc814_$2.log 2>&1; rc=$?
  echo "$(date) eval $1 rc=$rc"
done
echo ALLDONELRC814
