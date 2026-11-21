#!/bin/bash
# CacheBridge-style V calibration of our low-rank maps (1.7B->4B): 4 calibrations, then 8 tasks x 4 calibrated maps
cd ~/kvcodesign
source ~/kvbase/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
freegpus() { local f=""; for g in 3 4 5 1 2 6 7 0; do ok=1; for i in 1 2 3; do m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1000 ] || ok=0; sleep 1; done; [ $ok = 1 ] && f="$f $g"; done; echo $f; }
get1() { while true; do set -- $(freegpus); [ $# -ge 1 ] && break; sleep 30; done; echo $1; }
for spec in "2048 2048 0.01 f1" "2048 2048 1e12 nc" "768 1536 0.01 f1" "768 1536 1e12 nc"; do
  set -- $spec; G=$(get1); echo "$(date) calibrate k$1 v$2 $4 on $G"
  (RK=$1 RV=$2 FRAC=$3 TAG=$4 CUDA_VISIBLE_DEVICES=$G python fit_lrc.py > logs_lrc_k$1_$4.log 2>&1; echo "$(date) calib k$1 $4 exit $?") &
  sleep 75
done
wait
for f in 2048_f1 2048_nc 768_f1 768_nc; do grep -q LRCDONE logs_lrc_k$f.log || { echo "CALIB FAILED $f"; exit 1; }; done
M=lrc_k2048_v2048_f1,lrc_k2048_v2048_nc,lrc_k768_v1536_f1_bf16,lrc_k768_v1536_nc_bf16
for spec in "quality 200" "hotpotqa_e 200" "squad 200" "gsm8k_h 200" "kvret 200" "passkey_2k 200" "passkey_8k 100" "klchat_2k 200"; do
  set -- $spec; G=$(get1); echo "$(date) eval $1 on $G"
  (CUDA_VISIBLE_DEVICES=$G DIR=17to4 TASK=$1 N=$2 METHODS=$M OUT=$HOME/kvcodesign/results/quality_$1_lrc.json python eval_quality.py > logs_lrc_eval_$1.log 2>&1; echo "$(date) eval $1 exit $?") &
  sleep 75
done
wait; echo ALLDONELRC
