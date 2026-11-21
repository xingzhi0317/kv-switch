#!/bin/bash
# CFM with near-zero ridge: lam = 0.01 / 128000 on the averaged Gram (= 0.01 on the summed Gram, CacheBridge's fit_layer
# convention).  Fit -> K/V R^2 on QuALITY (vs paper Table 13: 0.617 / 0.503) -> QuALITY 200 q (paper: 44.27%)
cd ~/kvcodesign
source ~/kvbase/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MP=$HOME/kvbase/mappers; R=$HOME/kvcodesign/results/8to14
freegpus() { local f=""; for g in 3 4 5 6 7 0 1 2; do ok=1; for i in 1 2 3; do m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1000 ] || ok=0; sleep 2; done; [ $ok = 1 ] && f="$f $g"; done; echo $f; }
get() { while true; do set -- $(freegpus); [ $# -ge $NEED ] && break; sleep 60; done; echo $@ | tr " " "\n" | head -$NEED | paste -sd, ; }
NEED=2; G=$(get); echo "$(date) fit on $G"
DIR=8to14 LAM=7.8125e-8 OUT=$MP/kvbridge0_8to14 CUDA_VISIBLE_DEVICES=$G python fit_cfm_big.py > logs_cfm0_fit.log 2>&1 || { echo "FIT FAILED"; exit 1; }
NEED=4; G=$(get); echo "$(date) diag on $G"
(CUDA_VISIBLE_DEVICES=$G python diag_r2.py kb:$MP/kvbridge_8to14 kb:$MP/kvbridge0_8to14 > logs_cfm0_diag.log 2>&1; echo "$(date) DIAG exit $?") &
sleep 150
NEED=3; G=$(get); echo "$(date) eval on $G"
CUDA_VISIBLE_DEVICES=$G DIR=8to14 TASK=quality_cb N=200 METHODS=kb0 OUT=$R/quality_quality_cb_cfm0.json python eval_quality.py > logs_cfm0_eval.log 2>&1
echo "$(date) EVAL exit $?"
wait; echo ALLDONECFM0
