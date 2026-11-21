#!/bin/bash
# HotpotQA-E 200 q on Qwen3-8B -> 14B (not the calibration task): two method groups, one after the other (3 GPUs)
cd ~/kvcodesign
source ~/kvbase/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MP=$HOME/kvbase/mappers; R=$HOME/kvcodesign/results/8to14
freegpus() { local f=""; for g in 4 5 6 7 0 2 3 1; do ok=1; for i in 1 2 3; do m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1000 ] || ok=0; sleep 1; done; [ $ok = 1 ] && f="$f $g"; done; echo $f; }
get() { while true; do set -- $(freegpus); [ $# -ge 3 ] && break; sleep 60; done; echo $@ | tr " " "\n" | head -3 | paste -sd, ; }
for spec in "a native,none,rrr_k2048_v2048,cb_official" "b kb8,kb8cb,kvbridge"; do
  set -- $spec
  G=$(get); echo "$(date) hotpotqa group $1 on $G"
  CUDA_VISIBLE_DEVICES=$G DIR=8to14 TASK=hotpotqa_e N=200 METHODS=$2 MAPPER_CB=$MP/cachebridge_8to14_al_f1 OUT=$R/quality_hotpotqa_e_$1.json python eval_quality.py > logs_hq814_$1.log 2>&1; rc=$?
  echo "$(date) group $1 rc=$rc"
done
echo ALLDONEHQ
