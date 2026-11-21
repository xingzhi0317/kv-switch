#!/bin/bash
# low-rank fit for Qwen3-8B -> 14B, then QuALITY (paper layout, 200 q) for rank 2048 (fp32) and K768/V1536 (bf16)
cd ~/kvcodesign
source ~/kvbase/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=$HOME/kvcodesign/results/8to14
freegpus() { local f=""; for g in 4 5 6 7 0 2 3 1; do ok=1; for i in 1 2 3; do m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1000 ] || ok=0; sleep 1; done; [ $ok = 1 ] && f="$f $g"; done; echo $f; }
get() { while true; do set -- $(freegpus); [ $# -ge $NEED ] && break; sleep 60; done; echo $@ | tr " " "\n" | head -$NEED | paste -sd, ; }
NEED=2; G=$(get); echo "$(date) fit on $G"
DIR=8to14 CUDA_VISIBLE_DEVICES=$G python fit_lowrank_big.py > logs_lr814_fit.log 2>&1; rc=$?
[ $rc = 0 ] || { echo "FIT FAILED rc=$rc"; exit 1; }
NEED=3; G=$(get); echo "$(date) eval on $G"
CUDA_VISIBLE_DEVICES=$G DIR=8to14 TASK=quality_cb N=200 METHODS=rrr_k2048_v2048,rrr_k768_v1536_bf16 OUT=$R/quality_quality_cb_lowrank.json python eval_quality.py > logs_lr814_eval.log 2>&1; rc=$?
echo "$(date) EVAL rc=$rc"; echo ALLDONELR814
