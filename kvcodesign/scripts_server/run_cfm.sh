#!/bin/bash
# CFM (official kvbridge functions) on 8B->14B, then K/V R^2 diagnostics vs paper Table 13 and CFM QuALITY retention
cd ~/kvcodesign
source ~/kvbase/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MP=$HOME/kvbase/mappers; R=$HOME/kvcodesign/results/8to14
freegpus() { local f=""; for g in 3 4 5 6 7 0 1 2; do ok=1; for i in 1 2 3; do m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1000 ] || ok=0; sleep 2; done; [ $ok = 1 ] && f="$f $g"; done; echo $f; }
get() { while true; do set -- $(freegpus); [ $# -ge $NEED ] && break; sleep 60; done; echo $@ | tr " " "\n" | head -$NEED | paste -sd, ; }
NEED=2; G=$(get); echo "$(date) CFM fit on $G"
DIR=8to14 CUDA_VISIBLE_DEVICES=$G python fit_cfm_big.py > logs_cfm_fit.log 2>&1 || { echo "CFM FIT FAILED"; exit 1; }
echo "$(date) CFM fit done"
NEED=4; G=$(get); echo "$(date) R2 diagnostics on $G"
(CUDA_VISIBLE_DEVICES=$G python diag_r2.py kb:$MP/kvbridge_8to14 cb:$MP/cachebridge_8to14_init2 cb:$MP/cachebridge_8to14_fast_cov \
   cb:$MP/cachebridge_8to14_al_f1 cb:$MP/cachebridge_8to14_al_cov cb:$MP/cachebridge_8to14_al_nocon > logs_diag_r2.log 2>&1; echo "$(date) DIAG exit $?") &
sleep 150
NEED=3; G=$(get); echo "$(date) CFM QuALITY eval on $G"
CUDA_VISIBLE_DEVICES=$G DIR=8to14 TASK=quality_cb N=200 METHODS=kvbridge OUT=$R/quality_quality_cb_cfm.json python eval_quality.py > logs_cfm_eval.log 2>&1
echo "$(date) EVAL CFM exit $?"
wait; echo ALLDONECFM
