#!/bin/bash
cd ~/kvswitch/src; source ~/kvmap/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=3
until grep -q CHAIN3DONE ~/kvswitch/logs/q_P3.log 2>/dev/null; do sleep 30; done
python _p4t.py > ~/kvswitch/logs/p4test.log 2>&1
MODE=fit python p4.py > ~/kvswitch/logs/fit_P4.log 2>&1
for T in uc:100 lb8k:100 coqa1:100 niah8k:50; do
  MODE=eval TASK=${T%%:*} N=${T##*:} python p4.py >> ~/kvswitch/logs/q_P4.log 2>&1
done
MODE=time python p4.py > ~/kvswitch/logs/timing_P4.log 2>&1
echo CHAIN4DONE >> ~/kvswitch/logs/q_P4.log
