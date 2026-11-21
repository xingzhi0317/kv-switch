#!/bin/bash
# validation evals for the energy-budget sweep, at most 4 GPUs at once (replaces the eval stage of run_val.sh)
cd ~/kvcodesign
source ~/kvbase/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
V=$HOME/kvcodesign/results/val; mkdir -p $V
true
for t in e003 e01 e03 e1 e3 e10 enc; do grep -q FITCB7DONE logs_val_cb_$t.log || echo "VARIANT FAILED $t"; done
echo "$(date) calibrations done"
freegpus() { local f=""; for g in 3 4 5 6 7 0 1 2; do ok=1; for i in 1 2 3; do m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1000 ] || ok=0; sleep 1; done; [ $ok = 1 ] && f="$f $g"; done; echo $f; }
get1() { while true; do set -- $(freegpus); [ $# -ge 1 ] && break; sleep 30; done; echo $1; }
L=lrc_k2048_v2048
GRP=("native,rrr_k2048_v2048,${L}_f1,${L}_nc,${L}_b003" "${L}_b03,${L}_b1,${L}_b3,${L}_b10" "kvbridge,kb12cbe003,kb12cbe01,kb12cbe03" "kb12cbe1,kb12cbe3,kb12cbe10,kb12cbenc")
for spec in "quality 1645 1545" "hotpotqa_e 300 200" "squad 300 200" "passkey_2k 300 200" "klchat_2k 300 200"; do
  set -- $spec
  for gi in 0 1 2 3; do
    while [ $(jobs -rp | wc -l) -ge 4 ]; do sleep 20; done   # at most 4 evals (4 GPUs) at a time
    G=$(get1); echo "$(date) eval $1 group $gi on $G"
    (CUDA_VISIBLE_DEVICES=$G DIR=17to4 TASK=$1 N=$2 OFFSET=$3 METHODS=${GRP[$gi]} OUT=$V/quality_$1_g$gi.json python eval_quality.py > logs_val_eval_$1_g$gi.log 2>&1; rc=$?; echo "$(date) eval $1 g$gi exit $rc") &
    sleep 60
  done
done
wait; echo ALLDONEVAL
