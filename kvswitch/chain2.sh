#!/bin/bash
cd ~/kvswitch/src; source ~/kvmap/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=2 HF_ENDPOINT=https://hf-mirror.com
while pgrep -f "train.py" > /dev/null; do sleep 20; done
PAIR=P2 STEPS=600 RANK=64 python train.py > ~/kvswitch/logs/train_P2.log 2>&1
for T in uc:200 coqa1:100 coqa5:100 coqa10:100 lb8k:200 niah8k:50 niah16k:50 niah32k:50 niah4k:50 gsm:200; do
  PAIR=P1 TASK=${T%%:*} N=${T##*:} METHODS=ours_r64 SUFFIX=_ours python eval.py >> ~/kvswitch/logs/q_P1_ours.log 2>&1
done
echo CHAIN2DONE >> ~/kvswitch/logs/q_P1_ours.log
