#!/bin/bash
# CacheBridge 8B->14B with every public detail aligned: refit init (full 1024-token texts, all positions, FineWeb budget
# stats), calibrate (paper prompt, complete articles, FineWeb M/E0) at 1% and unconstrained, evaluate quality_cb 200 q
cd ~/kvcodesign
source ~/kvbase/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MP=$HOME/kvbase/mappers; R=$HOME/kvcodesign/results/8to14
freegpus() { local f=""; for g in 0 1 2 3 4 5 6 7; do ok=1; for i in 1 2 3; do m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1000 ] || ok=0; sleep 2; done; [ $ok = 1 ] && f="$f $g"; done; echo $f; }
get() { while true; do set -- $(freegpus); [ $# -ge $1 ] 2>/dev/null; [ $# -ge $NEED ] && break; sleep 60; done; echo $@ | tr " " "\n" | head -$NEED | paste -sd, ; }
NEED=2; G=$(get); echo "$(date) init fit on $G"
(cd ~/kvbase && DIR=8to14 FULL=1 CUDA_VISIBLE_DEVICES=$G python ~/kvcodesign/fit_cb_init_big.py > logs/fit_init2_8to14.log 2>&1) || { echo "INIT FAILED"; exit 1; }
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
ev() {   # METHODS MAPPER TAG GPUS
  CUDA_VISIBLE_DEVICES=$4 DIR=8to14 TASK=quality_cb N=200 METHODS=$1 MAPPER_CB=$2 OUT=$R/quality_quality_cb_$3.json \
    python eval_quality.py > logs_al_eval_$3.log 2>&1; echo "$(date) EVAL $3 exit $?"; }
cal() {  # FRAC TAG GPUS
  (cd ~/kvbase && DIR=8to14 FRAC=$1 PROMPT=cb TRUNC=0 STATS=$MP/cachebridge_8to14_init2_stats.pt INIT=$MP/cachebridge_8to14_init2 \
     OUT=$MP/cachebridge_8to14_al_$2 CUDA_VISIBLE_DEVICES=$3 python fit_cb6.py > logs/fit_al_$2.log 2>&1) || { echo "CALIB FAILED $2"; return; }
  ev cb_official $MP/cachebridge_8to14_al_$2 $2 $3; }
NEED=3
G=$(get); echo "$(date) eval init on $G"; ev native,none,cb_official $MP/cachebridge_8to14_init2 init $G & sleep 150
G=$(get); echo "$(date) cal f1 on $G"; cal 0.01 f1 $G & sleep 150
G=$(get); echo "$(date) cal nocon on $G"; cal 1e12 nocon $G &
wait; echo ALLDONEALIGNED
