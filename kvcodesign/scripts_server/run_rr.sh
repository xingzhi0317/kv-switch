#!/bin/bash
# RaReCache on its paper's pair (Qwen3-0.6B -> 14B, GSM8K, Table 5): GSM8K-calibrated K=8 map + r=128 subspace (fit_rr.py),
# then gsm8k_rr (0-shot CoT, <=1024 new tokens), first 200 test questions: 14B, 0.6B, map only, RaReCache rho 0.3 / 0.5
cd ~/kvcodesign
source ~/kvbase/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=$HOME/kvcodesign/results/06to14; mkdir -p $R
freegpus() { local f=""; for g in 0 1 2 3 4 5 6 7; do ok=1; for i in 1 2 3; do m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1000 ] || ok=0; sleep 1; done; [ $ok = 1 ] && f="$f $g"; done; echo $f; }
get() { while true; do set -- $(freegpus); [ $# -ge 4 ] && break; sleep 60; done; echo $@ | tr " " "\n" | head -2 | paste -sd, ; }   # 2 GPUs stay idle
G=$(get); echo "$(date) fit on $G"
CUDA_VISIBLE_DEVICES=$G python fit_rr.py > logs_rr_fit.log 2>&1; rc=$?
[ $rc = 0 ] || { echo "FIT FAILED rc=$rc"; exit 1; }
G=$(get); echo "$(date) eval on $G"
CUDA_VISIBLE_DEVICES=$G DIR=06to14 TASK=gsm8k_rr N=200 METHODS=native,src_native,kvbridge,rare30,rare50 OUT=$R/quality_gsm8k_rr.json python eval_quality.py > logs_rr_eval.log 2>&1; rc=$?
echo "$(date) eval rc=$rc"
echo ALLDONERR
