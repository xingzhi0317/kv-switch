#!/bin/bash
# kvbridge's own benchmark (author's code) on our server: 4B standalone and 1.7B -> 4B mapped, one GPU each
cd ~/kvcodesign
source ~/kvbase/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
freegpus() { local f=""; for g in 4 5 6 7 0 2 3 1; do ok=1; for i in 1 2 3; do m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1000 ] || ok=0; sleep 1; done; [ $ok = 1 ] && f="$f $g"; done; echo $f; }
get1() { while true; do set -- $(freegpus); [ $# -ge 3 ] && break; sleep 60; done; echo $1; }   # 2 GPUs stay idle
for mode in standalone mapped; do
  G=$(get1); echo "$(date) $mode on $G"
  (MODE=$mode N=2000 CUDA_VISIBLE_DEVICES=$G python bench_kvbridge_local.py > logs_kbbench_$mode.log 2>&1; rc=$?; echo "$(date) $mode rc=$rc") &
  sleep 90
done
wait; echo ALLDONEKB
