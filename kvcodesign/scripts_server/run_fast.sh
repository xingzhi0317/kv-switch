#!/bin/bash
# one-pass statistics -> verify (sum selection vs official fit_layer mapper) -> CFM-scale selection -> calibrate -> QuALITY
cd ~/kvcodesign
source ~/kvbase/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MP=$HOME/kvbase/mappers; R=$HOME/kvcodesign/results/8to14
freegpus() { local f=""; for g in 3 4 5 6 7 0 1 2; do ok=1; for i in 1 2 3; do m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1000 ] || ok=0; sleep 2; done; [ $ok = 1 ] && f="$f $g"; done; echo $f; }
get() { while true; do set -- $(freegpus); [ $# -ge $NEED ] && break; sleep 60; done; echo $@ | tr " " "\n" | head -$NEED | paste -sd, ; }
NEED=2; G=$(get); echo "$(date) collect on $G"
MODE=collect CUDA_VISIBLE_DEVICES=$G python cb_fast.py > logs_fast_collect.log 2>&1 || { echo "COLLECT FAILED"; exit 1; }
G1=${G%%,*}
MODE=fit SEL=sum OUT=$MP/cachebridge_8to14_fast_sum REF=$MP/cachebridge_8to14_init2 CUDA_VISIBLE_DEVICES=$G1 python cb_fast.py > logs_fast_fit_sum.log 2>&1 || { echo "FIT SUM FAILED"; exit 1; }
MODE=fit SEL=cov OUT=$MP/cachebridge_8to14_fast_cov CUDA_VISIBLE_DEVICES=$G1 python cb_fast.py > logs_fast_fit_cov.log 2>&1 || { echo "FIT COV FAILED"; exit 1; }
echo "$(date) fits done"
NEED=3; G=$(get); echo "$(date) eval no-cal cov on $G"
(CUDA_VISIBLE_DEVICES=$G DIR=8to14 TASK=quality_cb N=200 METHODS=cb_official MAPPER_CB=$MP/cachebridge_8to14_fast_cov \
   OUT=$R/quality_quality_cb_covinit.json python eval_quality.py > logs_fast_eval_covinit.log 2>&1; echo "$(date) EVAL covinit exit $?") &
sleep 150
G=$(get); echo "$(date) calibrate cov on $G"
(cd ~/kvbase && DIR=8to14 FRAC=0.01 PROMPT=cb TRUNC=0 STATS=$MP/cachebridge_8to14_fast_cov_stats.pt INIT=$MP/cachebridge_8to14_fast_cov \
   OUT=$MP/cachebridge_8to14_al_cov CUDA_VISIBLE_DEVICES=$G python fit_cb6.py > logs/fit_al_cov.log 2>&1) || { echo "CALIB FAILED"; wait; exit 1; }
CUDA_VISIBLE_DEVICES=$G DIR=8to14 TASK=quality_cb N=200 METHODS=cb_official MAPPER_CB=$MP/cachebridge_8to14_al_cov \
   OUT=$R/quality_quality_cb_covf1.json python eval_quality.py > logs_fast_eval_covf1.log 2>&1; echo "$(date) EVAL covf1 exit $?"
wait; echo ALLDONEFAST
