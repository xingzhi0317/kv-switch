#!/bin/bash
# Run ONE command on the first GPU that has been idle (< 1 GB used) on CHECKS consecutive checks, INTERVAL s apart
# (default 12 x 30 s = 6 minutes).  The long window matters: other users run chains of short jobs with ~1 minute
# gaps, and a 1-minute window twice put us on a card at the same moment as their next job.
# Never touches other users' processes.   LOG=logs/x.log bash run_when_free.sh <command ...>   (launch with nohup)
cd ~/kvcodesign
while true; do
  g=""
  for i in 0 1 2 3 4 5 6 7; do
    ok=1
    for k in $(seq 1 ${CHECKS:-12}); do
      m=$(nvidia-smi -i $i --query-gpu=memory.used --format=csv,noheader,nounits)
      [ "$m" -lt 1000 ] || { ok=0; break; }
      sleep ${INTERVAL:-30}
    done
    [ $ok = 1 ] && { g=$i; break; }
  done
  [ -n "$g" ] && break
  sleep 60
done
echo "$(date +%F_%T) launch on GPU $g: $*" >> "${LOG:-logs/run_when_free.log}"
CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_ENDPOINT=https://hf-mirror.com "$@" >> "${LOG:-logs/run_when_free.log}" 2>&1
