#!/bin/bash
# energy-budget sweep for 8B->14B: calibrate (fit_cb6.py, FRAC) then QuALITY 200 q; one pipeline per 3 idle GPUs
cd ~/kvcodesign
source ~/kvbase/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
freegpus() { local f=""; for g in 0 1 2 3 4 5 6 7; do ok=1; for i in 1 2 3; do m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1000 ] || ok=0; sleep 2; done; [ $ok = 1 ] && f="$f $g"; done; echo $f; }
one() {   # FRAC TAG GPUS
  (cd ~/kvbase && DIR=8to14 FRAC=$1 OUT=$HOME/kvbase/mappers/cachebridge_8to14_ec6_$2 CUDA_VISIBLE_DEVICES=$3 python fit_cb6.py > logs/fit_cb6_8to14_$2.log 2>&1) || { echo "CALIB FAILED $2"; return; }
  CUDA_VISIBLE_DEVICES=$3 DIR=8to14 TASK=quality N=200 METHODS=cb_official MAPPER_CB=$HOME/kvbase/mappers/cachebridge_8to14_ec6_$2 \
    OUT=$HOME/kvcodesign/results/8to14/quality_quality_cbec_$2.json python eval_quality.py > logs_8to14_eval_$2.log 2>&1
  echo "DONE $2 exit $?"
}
for spec in 0.1:f10 1.0:f100 1e12:nocon; do
  frac=${spec%%:*}; tag=${spec##*:}
  while true; do set -- $(freegpus); [ $# -ge 3 ] && break; sleep 60; done
  echo "$(date) $tag (FRAC=$frac) on $1,$2,$3"
  one $frac $tag $1,$2,$3 &
  sleep 150   # let the models load before the next GPU scan
done
wait; echo ALLDONEBUDGET
