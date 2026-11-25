#!/bin/bash
# Replaces run_kvl2.sh after the data build (same steps): closed form per direction (GPU 0 / 1), then 3-LR training,
# 1.7B->4B data-parallel on GPUs 0-2 and 4B->1.7B on GPUs 3-4; GPU 5 stays free for evaluation work, 6/7 idle.
cd ~/kvcodesign
source ~/kvbase/.venv/bin/activate
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
until grep -q KVL2DATADONE logs_kvl2_data.log; do
  pgrep -f "python kvlingo_data2.py" > /dev/null || { echo "$(date) data build died"; exit 1; }
  sleep 20
done
echo "$(date) data built"
idle() { m=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1000 ]; }
for g in 0 1 2 3 4; do idle $g || { echo "$(date) GPU $g busy, abort"; exit 1; }; done
(MODE=fit PAIR=17to4 CUDA_VISIBLE_DEVICES=0 python kvlingo_train2.py > logs_kvl2_fit_17to4.log 2>&1) &
(MODE=fit PAIR=4to17 CUDA_VISIBLE_DEVICES=1 python kvlingo_train2.py > logs_kvl2_fit_4to17.log 2>&1) &
wait
grep -q FITDONE logs_kvl2_fit_17to4.log && grep -q FITDONE logs_kvl2_fit_4to17.log || { echo "$(date) fit failed"; exit 1; }
echo "$(date) fits done"
(MODE=train PAIR=17to4 LRS=3e-6,1e-5,3e-5 TAG=kvl2_17to4 CUDA_VISIBLE_DEVICES=0,1,2 setsid nohup torchrun --nnodes=1 --nproc_per_node=3 \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:29531 kvlingo_train2.py > logs_kvl2_17to4.log 2>&1 < /dev/null &)
(MODE=train PAIR=4to17 LRS=1e-4,3e-4,1e-3 TAG=kvl2_4to17 CUDA_VISIBLE_DEVICES=3,4 setsid nohup torchrun --nnodes=1 --nproc_per_node=2 \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:29532 kvlingo_train2.py > logs_kvl2_4to17.log 2>&1 < /dev/null &)
echo "$(date) KVL2LAUNCHED"
