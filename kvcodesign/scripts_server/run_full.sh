#!/bin/bash
# QuALITY (quality_cb) on all 1645 evaluation questions for 8B->14B; each job resumes its 200-question file
cd ~/kvcodesign
source ~/kvbase/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MP=$HOME/kvbase/mappers; R=$HOME/kvcodesign/results/8to14
freegpus() { local f=""; for g in 3 4 5 6 7 0 1 2; do ok=1; for i in 1 2 3; do m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1000 ] || ok=0; sleep 2; done; [ $ok = 1 ] && f="$f $g"; done; echo $f; }
get() { while true; do set -- $(freegpus); [ $# -ge 3 ] && break; sleep 60; done; echo $@ | tr " " "\n" | head -3 | paste -sd, ; }
job() {   # TAG METHODS MAPPER
  G=$(get); echo "$(date) $1 on $G"
  CUDA_VISIBLE_DEVICES=$G DIR=8to14 TASK=quality_cb N=1645 METHODS=$2 MAPPER_CB=$3 OUT=$R/quality_quality_cb_$1.json \
    python eval_quality.py > logs_full_$1.log 2>&1
  echo "$(date) $1 exit $?"
}
job init native,none,cb_official $MP/cachebridge_8to14_init2 & sleep 150
job f1 cb_official $MP/cachebridge_8to14_al_f1 & sleep 150
(job cfm kvbridge $MP/cachebridge_8to14_al_f1; job nocon cb_official $MP/cachebridge_8to14_al_nocon) &
wait; echo ALLDONEFULL
