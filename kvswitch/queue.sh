#!/bin/bash
# usage: queue.sh GPU NAME "ENV1 ENV2..;ENV..;..."   runs eval.py jobs sequentially
cd ~/kvswitch/src; source ~/kvmap/.venv/bin/activate
G=$1; NAME=$2; JOBS=$3
(
IFS=';'
for j in $JOBS; do
  echo "=== $j $(date)"
  eval env CUDA_VISIBLE_DEVICES=$G $j python ${SCRIPT:-eval.py}
done
echo QUEUEDONE
) > ~/kvswitch/logs/$NAME.log 2>&1 < /dev/null &
