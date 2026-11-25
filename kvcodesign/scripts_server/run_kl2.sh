#!/bin/bash
# KL-trained translator, paper recipe (train_kl2.py): LR sweep half a decade apart (KV-Lingo App. E), one GPU per LR,
# 2 GPUs always left idle.  usage: DIR=17to4 ./run_kl2.sh 1e-5 3e-5 1e-4 3e-4 1e-3
cd ~/kvcodesign
source ~/kvbase/.venv/bin/activate
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
idle() { m=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1000 ]; }
nfree() { local n=0; for g in 0 1 2 3 4 5 6 7; do idle $g && n=$((n+1)); done; echo $n; }
for lr in "$@"; do
  while true; do
    g=""; for c in 1 2 3 4 5 0 6 7; do ok=1; for i in 1 2 3; do idle $c || ok=0; sleep 1; done; [ $ok = 1 ] && { g=$c; break; }; done
    [ -n "$g" ] && [ $(nfree) -ge 3 ] && break; sleep 60
  done
  echo "$(date) LR $lr on GPU $g"
  DIR=$DIR LR=$lr TAG=lr$lr CUDA_VISIBLE_DEVICES=$g setsid nohup python train_kl2.py > logs_kl2_${DIR}_lr$lr.log 2>&1 < /dev/null &
  sleep 90   # let it allocate before the next GPU check
done
echo LAUNCHED
