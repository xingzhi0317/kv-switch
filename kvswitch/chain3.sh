#!/bin/bash
cd ~/kvswitch/src; source ~/kvmap/.venv/bin/activate
while pgrep -f "timing.py" > /dev/null; do sleep 20; done
export CUDA_VISIBLE_DEVICES=3 HF_ENDPOINT=https://hf-mirror.com
PAIR=P2 python timing.py > ~/kvswitch/logs/timing_P2.log 2>&1
A=Llama-3.2-3B-Instruct B=Qwen3-1.7B TAG=P3 python fit_x.py > ~/kvswitch/logs/fit_P3.log 2>&1
for T in uc:200 coqa1:100 lb8k:200 niah8k:50 gsm:100; do
  PAIR=P3 TASK=${T%%:*} N=${T##*:} METHODS=native,none,B4,B4k1 python eval.py >> ~/kvswitch/logs/q_P3.log 2>&1
done
echo CHAIN3DONE >> ~/kvswitch/logs/q_P3.log
