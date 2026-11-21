#!/bin/bash
# HotpotQA-E 200 q, 8B -> 14B, all-head maps one per job (three together ran out of memory on GPU 0 at 16K histories)
cd ~/kvcodesign
source ~/kvbase/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=$HOME/kvcodesign/results/8to14
freegpus() { local f=""; for g in 4 5 6 7 0 2 3 1; do ok=1; for i in 1 2 3; do m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1000 ] || ok=0; sleep 1; done; [ $ok = 1 ] && f="$f $g"; done; echo $f; }
get() { while true; do set -- $(freegpus); [ $# -ge 5 ] && break; sleep 60; done; echo $@ | tr " " "\n" | head -3 | paste -sd, ; }   # 2 GPUs stay idle
for m in kb8 kb8cb kvbridge; do
  while [ $(jobs -rp | wc -l) -ge 2 ]; do sleep 20; done
  G=$(get); echo "$(date) hotpotqa $m on $G"
  (CUDA_VISIBLE_DEVICES=$G DIR=8to14 TASK=hotpotqa_e N=200 METHODS=$m OUT=$R/quality_hotpotqa_e_b_$m.json python eval_quality.py > logs_hq814_b_$m.log 2>&1; rc=$?; echo "$(date) $m rc=$rc") &
  sleep 90
done
wait; echo ALLDONEHQB
