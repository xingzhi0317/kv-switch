#!/bin/bash
# after the M1 queues on GPU0/GPU1 finish: ablation training + guard calibration (GPU0), ours eval for P2 (GPU1)
cd ~/kvswitch/src; source ~/kvmap/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
g0() {
  until grep -q QUEUEDONE ~/kvswitch/logs/q_P1.log; do sleep 30; done
  export CUDA_VISIBLE_DEVICES=0
  PAIR=P1 TASK=coqa1 N=100 METHODS=native,none,B2,B1,B3 python eval.py >> ~/kvswitch/logs/q_P1_fix.log 2>&1
  PAIR=P1 TASK=coqa5 N=100 METHODS=native,B2,B1,B3 python eval.py >> ~/kvswitch/logs/q_P1_fix.log 2>&1
  PAIR=P1 TASK=coqa10 N=100 METHODS=native,B2,B1,B3 python eval.py >> ~/kvswitch/logs/q_P1_fix.log 2>&1
  for T in coqa1:100 lb8k:200 gsm:100; do
    PAIR=P1 TASK=${T%%:*} N=${T##*:} METHODS=noise0.5,noise2 SUFFIX=_noise python eval.py >> ~/kvswitch/logs/q_P1_noise.log 2>&1
  done
  PAIR=P1 STEPS=600 RANK=16 python train.py > ~/kvswitch/logs/train_P1_r16.log 2>&1
  PAIR=P1 STEPS=600 RANK=64 NMIN=2048 NMAX=2048 TAG=ours_r64_fix2k python train.py > ~/kvswitch/logs/train_P1_fix2k.log 2>&1
  for T in uc:200 lb8k:200; do
    PAIR=P1 TASK=${T%%:*} N=${T##*:} METHODS=ours_r16,ours_r64_fix2k SUFFIX=_abl python eval.py >> ~/kvswitch/logs/q_P1_abl.log 2>&1
  done
  echo G0DONE >> ~/kvswitch/logs/q_P1_abl.log
}
g1() {
  until grep -q QUEUEDONE ~/kvswitch/logs/q_P2.log; do sleep 30; done
  until [ -f ~/kvswitch/results/train_P2_ours_r64.json ]; do sleep 30; done
  export CUDA_VISIBLE_DEVICES=1
  PAIR=P2 TASK=coqa1 N=100 METHODS=native,none,B2,B1,B3 python eval.py >> ~/kvswitch/logs/q_P2_fix.log 2>&1
  PAIR=P2 TASK=coqa5 N=100 METHODS=native,B2,B1,B3 python eval.py >> ~/kvswitch/logs/q_P2_fix.log 2>&1
  PAIR=P2 TASK=coqa10 N=100 METHODS=native,B2,B1,B3 python eval.py >> ~/kvswitch/logs/q_P2_fix.log 2>&1
  for T in uc:200 coqa1:100 coqa5:100 coqa10:100 lb8k:200 niah8k:50 niah16k:50 niah4k:50 gsm:200 niah32k:50; do
    M=ours_r64; [[ $T == coqa* ]] && M=ours_r64,ours_r64_c3,B1_c3
    PAIR=P2 TASK=${T%%:*} N=${T##*:} METHODS=$M SUFFIX=_ours python eval.py >> ~/kvswitch/logs/q_P2_ours.log 2>&1
  done
  echo G1DONE >> ~/kvswitch/logs/q_P2_ours.log
}
g0 & g1 & wait
