#!/bin/bash
# usage: launch.sh GPU LOGNAME script.py [env...]
cd ~/kvswitch/src; source ~/kvmap/.venv/bin/activate
G=$1; N=$2; S=$3; shift 3
env CUDA_VISIBLE_DEVICES=$G HF_ENDPOINT=https://hf-mirror.com "$@" setsid nohup python $S > ~/kvswitch/logs/$N.log 2>&1 < /dev/null &
echo launched $N pid $!
