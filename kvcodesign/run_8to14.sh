#!/bin/bash
# fit the uncalibrated CacheBridge map for Qwen3-8B -> 14B, then QuALITY 100 q (native, none, cb_official)
cd ~/kvcodesign
source ~/kvbase/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
freegpus() {   # GPUs 3..7 idle (< 1 GB) on 3 checks
  local f=""; for g in 3 4 5 6 7; do ok=1; for i in 1 2 3; do m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1000 ] || ok=0; sleep 3; done; [ $ok = 1 ] && f="$f $g"; done; echo $f; }
if [ "$1" = smoke ]; then
  set -- $(freegpus); G="$1,$2,$3"; echo "smoke on $G"
  CUDA_VISIBLE_DEVICES=$G DIR=8to14 TASK=quality N=2 METHODS=native,none OUT=$HOME/kvcodesign/results/8to14/smoke.json python eval_quality.py > logs_8to14_smoke.log 2>&1; echo EXIT $? >> logs_8to14_smoke.log; exit
fi
set -- $(freegpus); echo "$(date) fit on $1,$2"
CUDA_VISIBLE_DEVICES=$1,$2 DIR=8to14 python fit_cb_init_big.py > logs_8to14_fit.log 2>&1 || { echo FIT FAILED; exit 1; }
while true; do set -- $(freegpus); [ $# -ge 3 ] && break; sleep 60; done
echo "$(date) eval on $1,$2,$3"
CUDA_VISIBLE_DEVICES=$1,$2,$3 DIR=8to14 TASK=quality N=100 METHODS=native,none,cb_official \
  MAPPER_CB=$HOME/kvbase/mappers/cachebridge_8to14_init OUT=$HOME/kvcodesign/results/8to14/quality_quality_cbinit.json \
  python eval_quality.py > logs_8to14_eval.log 2>&1
echo EVAL EXIT $?
echo ALLDONE8TO14
