#!/bin/bash
# all-head + CacheBridge calibration, fast (fit_cb7.py): (A) check against fit_cb5.py on 1.7B->4B (3 layers);
# (B) Qwen3-8B->14B, 8 source layers (CFM k=8 init), paper-aligned data, 1% budget and no budget -> QuALITY 200 q
cd ~/kvcodesign
source ~/kvbase/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MP=$HOME/kvbase/mappers; R=$HOME/kvcodesign/results/8to14
freegpus() { local f=""; for g in 3 4 5 6 7 0 1 2; do ok=1; for i in 1 2 3; do m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1000 ] || ok=0; sleep 2; done; [ $ok = 1 ] && f="$f $g"; done; echo $f; }
get() { while true; do set -- $(freegpus); [ $# -ge $NEED ] && break; sleep 60; done; echo $@ | tr " " "\n" | head -$NEED | paste -sd, ; }
NEED=1; G=$(get); echo "$(date) A: check on $G"
(DIR=17to4 INIT=$MP/kvbridge3_17to4 OUT=$MP/kvbridge3cbv_17to4 FRAC=0.01 PROMPT=old TRUNC=6000 CUDA_VISIBLE_DEVICES=$G python fit_cb7.py > logs_cb7_check.log 2>&1 \
  && python cmp_kb.py $MP/kvbridge3_17to4 $MP/kvbridge3cb_17to4 $MP/kvbridge3cbv_17to4 >> logs_cb7_check.log 2>&1 \
  && CUDA_VISIBLE_DEVICES=$G DIR=17to4 TASK=quality N=50 METHODS=kb3cbv OUT=$HOME/kvcodesign/results/check_cb7_quality.json python eval_quality.py > logs_cb7_check_eval.log 2>&1; echo "$(date) A exit $?") &
sleep 120
NEED=3; G=$(get); echo "$(date) B1: 1% on $G"
(DIR=8to14 INIT=$MP/kvbridge8_8to14 OUT=$MP/kvbridge8cb_8to14 FRAC=0.01 PROMPT=cb TRUNC=0 STATS=fineweb CUDA_VISIBLE_DEVICES=$G python fit_cb7.py > logs_cb7_f1.log 2>&1; echo "$(date) B1 exit $?") &
sleep 150
G=$(get); echo "$(date) B2: no budget on $G"
DIR=8to14 INIT=$MP/kvbridge8_8to14 OUT=$MP/kvbridge8cbn_8to14 FRAC=1e12 PROMPT=cb TRUNC=0 STATS=fineweb CUDA_VISIBLE_DEVICES=$G python fit_cb7.py > logs_cb7_nocon.log 2>&1; echo "$(date) B2 exit $?"
wait
G=$(get); echo "$(date) eval on $G"
CUDA_VISIBLE_DEVICES=$G DIR=8to14 TASK=quality_cb N=200 METHODS=kb8cb,kb8cbn OUT=$R/quality_quality_cb_kb8cb.json python eval_quality.py > logs_cb7_eval.log 2>&1
echo "$(date) EVAL exit $?"; echo ALLDONECB7
