#!/bin/bash
# PLAN_v5 stage 1: launch all exact-information evaluations through the polite GPU queue (free cards only,
# one job per card).  Evaluations resume from their OUT file, so re-running this after a kill continues the work.
#   bash ~/kvcodesign/run_stage1.sh            # all jobs
#   JOBS=squad_a,squad_b bash run_stage1.sh    # a subset
# Finished when ~/kvcodesign/logs/queue_stage1.log contains QUEUEDONE.
cd ~/kvcodesign && mkdir -p logs results
~/kvbase/.venv/bin/python test_tasks.py || exit 1
if ps aux | grep -q "[g]pu_queue.py"; then echo "a queue is already running"; exit 1; fi
SET=stage1 nohup ~/kvbase/.venv/bin/python gpu_queue.py > logs/queue_stage1.log 2>&1 &
echo "queue started, pid $!"
