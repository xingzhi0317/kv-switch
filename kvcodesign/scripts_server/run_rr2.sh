#!/bin/bash
# resume gsm8k_rr 06to14 from the saved 100 questions with batched generation (BATCH=16)
cd ~/kvcodesign
source ~/kvbase/.venv/bin/activate
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0,1 DIR=06to14 TASK=gsm8k_rr N=200 BATCH=16 METHODS=native,src_native,kvbridge,rare30,rare50 OUT=results/06to14/quality_gsm8k_rr.json python eval_quality.py > logs_rr_eval2.log 2>&1; rc=$?
echo "$(date) rc=$rc"; echo ALLDONERR2
