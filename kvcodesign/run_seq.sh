#!/bin/bash
# Run eval_quality.py for several tasks one after another on the GPU this script was given (CUDA_VISIBLE_DEVICES).
#   DIR=17to4 METHODS=a,b TAG=x bash run_seq.sh task1:N1 task2:N2 ...
# Output: results/[DIR/]quality_{task}_{TAG}.json   (resumable: re-running continues where it stopped)
cd ~/kvcodesign
for tn in "$@"; do
  t=${tn%%:*}; n=${tn##*:}
  sub=""; [ "$DIR" != "17to4" ] && sub="$DIR/"
  echo "$(date +%T) $DIR $t N=$n $METHODS"
  TASK=$t N=$n OUT=$HOME/kvcodesign/results/${sub}quality_${t}_${TAG}.json ~/kvbase/.venv/bin/python eval_quality.py 2>&1 | grep -v "Warn\|warn\|Loading"
done
echo SEQDONE
