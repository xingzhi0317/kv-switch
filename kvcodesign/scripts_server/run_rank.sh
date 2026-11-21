#!/bin/bash
cd ~/kvcodesign
source ~/kvbase/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
f=""; for g in 3 4 5 6 7 0 1 2; do ok=1; for i in 1 2 3; do m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1000 ] || ok=0; sleep 2; done; [ $ok = 1 ] && f="$f $g"; done
set -- $f; [ $# -ge 2 ] || { echo "not enough free GPUs: $f"; exit 1; }
echo "$(date) rank check on $1,$2"
CUDA_VISIBLE_DEVICES=$1,$2 python rank_check.py > logs_rank_check.log 2>&1
echo "EXIT $?"
