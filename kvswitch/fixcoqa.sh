#!/bin/bash
# resume the two coqa1 runs that crashed (cache headroom bug, fixed) once the main M1 queues are done
cd ~/kvswitch/src; source ~/kvmap/.venv/bin/activate
f() { until grep -q QUEUEDONE ~/kvswitch/logs/q_$1.log; do sleep 30; done
      CUDA_VISIBLE_DEVICES=$2 PAIR=$1 TASK=coqa1 N=100 METHODS=native,none,B2,B1,B3 python eval.py >> ~/kvswitch/logs/q_$1_fix.log 2>&1; echo FIXDONE >> ~/kvswitch/logs/q_$1_fix.log; }
f P1 0 & f P2 1 & wait
