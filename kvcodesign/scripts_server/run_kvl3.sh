#!/bin/bash
# KV-Lingo retrain on the paper-style stream (kvlingo_data2.py), the paper's learning rates only (Table 5, nu=1, depth):
# 1.7B->4B 1e-5 on GPUs 0-1, 4B->1.7B 3e-4 on GPUs 2-3 (at most 4 GPUs for long runs); checkpoint every 250 steps,
# restart this script to resume.  Closed forms: kvlingo/cf2_{PAIR}_nu1_b0.pt (kvlingo_train2.py MODE=fit, 11-22).
cd ~/kvcodesign
source ~/kvbase/.venv/bin/activate
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CKPT_EVERY=250 RESUME=1
idle() { m=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1000 ]; }
for g in 0 1 2 3; do idle $g || { echo "$(date) GPU $g busy, abort"; exit 1; }; done
(PAIR=17to4 LRS=1e-5 TAG=kvl3_17to4 CUDA_VISIBLE_DEVICES=0,1 setsid nohup torchrun --nnodes=1 --nproc_per_node=2 \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:29551 kvlingo_train3.py >> logs_kvl3_17to4.log 2>&1 < /dev/null &)
(PAIR=4to17 LRS=3e-4 TAG=kvl3_4to17 CUDA_VISIBLE_DEVICES=2,3 setsid nohup torchrun --nnodes=1 --nproc_per_node=2 \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:29552 kvlingo_train3.py >> logs_kvl3_4to17.log 2>&1 < /dev/null &)
echo "$(date) KVL3LAUNCHED"
