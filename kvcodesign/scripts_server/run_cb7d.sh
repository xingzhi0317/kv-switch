#!/bin/bash
# 8B->14B all-head k=8 calibration with the 1% budget (fp64 M fix), then QuALITY 200 q
cd ~/kvcodesign
source ~/kvbase/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MP=$HOME/kvbase/mappers; R=$HOME/kvcodesign/results/8to14
freegpus() { local f=""; for g in 6 7 0 3 4 5 1 2; do ok=1; for i in 1 2 3; do m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1000 ] || ok=0; sleep 2; done; [ $ok = 1 ] && f="$f $g"; done; echo $f; }
get() { while true; do set -- $(freegpus); [ $# -ge 3 ] && break; sleep 60; done; echo $@ | tr " " "\n" | head -3 | paste -sd, ; }
G=$(get); echo "$(date) 1% calibration on $G"
DIR=8to14 INIT=$MP/kvbridge8_8to14 OUT=$MP/kvbridge8cb_8to14 FRAC=0.01 PROMPT=cb TRUNC=0 STATS=fineweb CUDA_VISIBLE_DEVICES=$G python fit_cb7.py > logs_cb7d_f1.log 2>&1 || { echo "CALIB FAILED"; exit 1; }
CUDA_VISIBLE_DEVICES=$G DIR=8to14 TASK=quality_cb N=200 METHODS=kb8cb OUT=$R/quality_quality_cb_kb8cb.json python eval_quality.py > logs_cb7d_eval.log 2>&1
echo "$(date) EVAL exit $?"; echo ALLDONECB7D
