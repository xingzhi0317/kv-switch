#!/bin/bash
# CFM source-layer count sweep on 8B->14B: k = 1, 2, 4, 8 (k = 12 done) -> R^2 vs paper Table 13 -> QuALITY 200 q vs 44.27%
cd ~/kvcodesign
source ~/kvbase/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MP=$HOME/kvbase/mappers; R=$HOME/kvcodesign/results/8to14
freegpus() { local f=""; for g in 3 4 5 6 7 0 1 2; do ok=1; for i in 1 2 3; do m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1000 ] || ok=0; sleep 2; done; [ $ok = 1 ] && f="$f $g"; done; echo $f; }
get() { while true; do set -- $(freegpus); [ $# -ge $NEED ] && break; sleep 60; done; echo $@ | tr " " "\n" | head -$NEED | paste -sd, ; }
NEED=2; G=$(get); echo "$(date) fit k=1,2,4,8 on $G"
DIR=8to14 TOPKS=1,2,4,8 OUT="$MP/kvbridge{k}_8to14" CUDA_VISIBLE_DEVICES=$G python fit_cfm_big.py > logs_cfmk_fit.log 2>&1 || { echo "FIT FAILED"; exit 1; }
cp $R/diag_r2.json $R/diag_r2_cfm0.json
NEED=4; G=$(get); echo "$(date) diag on $G"
CUDA_VISIBLE_DEVICES=$G python diag_r2.py kb:$MP/kvbridge1_8to14 kb:$MP/kvbridge2_8to14 kb:$MP/kvbridge4_8to14 kb:$MP/kvbridge8_8to14 kb:$MP/kvbridge_8to14 > logs_cfmk_diag.log 2>&1
echo "$(date) DIAG exit $?"; cp $R/diag_r2.json $R/diag_r2_cfmk.json
NEED=3; G=$(get); echo "$(date) eval k=1,2,4 on $G"
(CUDA_VISIBLE_DEVICES=$G DIR=8to14 TASK=quality_cb N=200 METHODS=kb1,kb2,kb4 OUT=$R/quality_quality_cb_cfmk_a.json python eval_quality.py > logs_cfmk_eval_a.log 2>&1; echo "$(date) EVAL a exit $?") &
sleep 150
G=$(get); echo "$(date) eval k=8 on $G"
CUDA_VISIBLE_DEVICES=$G DIR=8to14 TASK=quality_cb N=200 METHODS=kb8 OUT=$R/quality_quality_cb_cfmk_b.json python eval_quality.py > logs_cfmk_eval_b.log 2>&1
echo "$(date) EVAL b exit $?"
wait; echo ALLDONECFMK
