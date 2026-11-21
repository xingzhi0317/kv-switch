#!/bin/bash
# Energy-budget sweep on a VALIDATION split (disjoint from the 200 plotted test questions), Qwen3-1.7B -> 4B:
#   low-rank 2048 (fit_lrc.py) and CacheBridge variant all-head 12 layers (fit_cb7.py, init kvbridge_17to4),
#   budgets 0.3% 1% 3% 10% 30% 100% none; validation = 100 q each of QuALITY (last 100 of the pool), HotpotQA-E,
#   SQuAD, 2K passkey, continuation KL (items 201-300).  Results -> results/val/
cd ~/kvcodesign
source ~/kvbase/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MP=$HOME/kvbase/mappers; V=$HOME/kvcodesign/results/val; mkdir -p $V
freegpus() { local f=""; for g in 3 4 5 6 7 0 1 2; do ok=1; for i in 1 2 3; do m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1000 ] || ok=0; sleep 1; done; [ $ok = 1 ] && f="$f $g"; done; echo $f; }
get1() { while true; do set -- $(freegpus); [ $# -ge 1 ] && break; sleep 30; done; echo $1; }
# ---- calibrations (one GPU each)
for spec in "0.003 b003" "0.03 b03" "0.1 b1" "0.3 b3" "1.0 b10"; do
  set -- $spec; G=$(get1); echo "$(date) lowrank $2 on $G"
  (RK=2048 RV=2048 FRAC=$1 TAG=$2 CUDA_VISIBLE_DEVICES=$G python fit_lrc.py > logs_val_lrc_$2.log 2>&1; echo "$(date) lowrank $2 exit $?") &
  sleep 75
done
for spec in "0.003 e003" "0.01 e01" "0.03 e03" "0.1 e1" "0.3 e3" "1.0 e10" "1e12 enc"; do
  set -- $spec; G=$(get1); echo "$(date) variant $2 on $G"
  (DIR=17to4 INIT=$MP/kvbridge_17to4 OUT=$MP/kvbridge12cb$2_17to4 FRAC=$1 PROMPT=old TRUNC=6000 CUDA_VISIBLE_DEVICES=$G python fit_cb7.py > logs_val_cb_$2.log 2>&1; echo "$(date) variant $2 exit $?") &
  sleep 75
done
wait
for t in b003 b03 b1 b3 b10; do grep -q LRCDONE logs_val_lrc_$t.log || echo "LOWRANK FAILED $t"; done
for t in e003 e01 e03 e1 e3 e10 enc; do grep -q FITCB7DONE logs_val_cb_$t.log || echo "VARIANT FAILED $t"; done
# ---- validation evals: 5 tasks x 4 method groups, one GPU each
L=lrc_k2048_v2048
GROUPS=("native,rrr_k2048_v2048,${L}_f1,${L}_nc,${L}_b003" "${L}_b03,${L}_b1,${L}_b3,${L}_b10" "kvbridge,kb12cbe003,kb12cbe01,kb12cbe03" "kb12cbe1,kb12cbe3,kb12cbe10,kb12cbenc")
for spec in "quality 1645 1545" "hotpotqa_e 300 200" "squad 300 200" "passkey_2k 300 200" "klchat_2k 300 200"; do
  set -- $spec
  for gi in 0 1 2 3; do
    G=$(get1); echo "$(date) eval $1 group $gi on $G"
    (CUDA_VISIBLE_DEVICES=$G DIR=17to4 TASK=$1 N=$2 OFFSET=$3 METHODS=${GROUPS[$gi]} OUT=$V/quality_$1_g$gi.json python eval_quality.py > logs_val_eval_$1_g$gi.log 2>&1; echo "$(date) eval $1 g$gi exit $?") &
    sleep 60
  done
done
wait; echo ALLDONEVAL
