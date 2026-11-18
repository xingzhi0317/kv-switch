#!/bin/bash
# Time to first token, one method per process (no other method's weights in memory).
#   DIR=17to4 bash run_ttft.sh native none kvbridge ...      -> results/ttft_{DIR}.json (resumable)
cd ~/kvcodesign
for m in "$@"; do
  echo "$(date +%T) $DIR $m"
  DIR=$DIR MODE=ttft TASK=kvret N=1 METHODS=$m REPS=${REPS:-7} HF_HUB_OFFLINE=1 ~/kvbase/.venv/bin/python eval_quality.py 2>&1 | grep "^ttft\|Error\|Traceback"
done
echo TTFTALLDONE
