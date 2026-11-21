#!/bin/bash
# test set (the plotted 200 questions per task) for the validation-chosen budgets: low-rank 2048 at 3%, variant at 10%
cd ~/kvcodesign
source ~/kvbase/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
freegpus() { local f=""; for g in 3 4 5 6 7 0 1 2; do ok=1; for i in 1 2 3; do m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1000 ] || ok=0; sleep 1; done; [ $ok = 1 ] && f="$f $g"; done; echo $f; }
get1() { while true; do set -- $(freegpus); [ $# -ge 1 ] && break; sleep 30; done; echo $1; }
METH=lrc_k2048_v2048_b03,kb12cbe1
for spec in "gsm8k_h 200" "quality 200" "hotpotqa_e 200" "squad 200" "kvret 200" "passkey_2k 200" "passkey_8k 100" "klchat_2k 200"; do
  set -- $spec
  while [ $(jobs -rp | wc -l) -ge 4 ]; do sleep 20; done
  G=$(get1); echo "$(date) $1 on $G"
  (CUDA_VISIBLE_DEVICES=$G DIR=17to4 TASK=$1 N=$2 METHODS=$METH OUT=$HOME/kvcodesign/results/quality_$1_tuned.json python eval_quality.py > logs_tuned_$1.log 2>&1; rc=$?; echo "$(date) $1 exit $rc") &
  sleep 60
done
wait; echo ALLDONETUNED
