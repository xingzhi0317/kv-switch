#!/bin/bash
# after the LR sweep of a direction finishes: pick the LR with the lowest FINAL val KL (KV-Lingo App. E), link its mapper
# as kvbridge3cklbest_{DIR} (eval method kb3cklbest), then run the same task set the old KL translator was scored on.
# One GPU per task, 2 GPUs always left idle.  usage: DIR=17to4 ./auto_kl2_eval.sh 1e-6 3e-6 1e-5 3e-5
cd ~/kvcodesign
source ~/kvbase/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LRS="$@"
while true; do   # every run of the sweep has written its step-2000 validation
  n=0; for lr in $LRS; do python -c "import json,sys; h=json.load(open('results/train_kl2_${DIR}_lr$lr.json')); sys.exit(0 if h['val'][-1][0]==h['config']['steps'] else 1)" 2>/dev/null && n=$((n+1)); done
  [ $n = $# ] && break; sleep 120
done
BEST=$(python -c "
import json
r={lr: json.load(open(f'results/train_kl2_${DIR}_lr{lr}.json'))['val'][-1][1] for lr in '$LRS'.split()}
print(min(r, key=r.get)); print(r, file=open('results/train_kl2_${DIR}_choice.txt','w'))")
echo "$(date) best LR $BEST"
ln -sfn $HOME/kvbase/mappers/kvbridge3ckllr${BEST}_$DIR $HOME/kvbase/mappers/kvbridge3cklbest_$DIR
R=results; [ $DIR = 17to4 ] || R=results/$DIR
idle() { m=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1000 ]; }
nfree() { local n=0; for g in 0 1 2 3 4 5 6 7; do idle $g && n=$((n+1)); done; echo $n; }
for spec in passkey_2k:200 passkey_8k:100 klchat_2k:200 klchat_8k:100 quality:200 hotpotqa_e:200 squad:200 kvret:200 gsm8k_h:200; do
  t=${spec%%:*}; n=${spec##*:}
  while true; do
    g=""; for c in 0 1 2 3 4 5 6 7; do ok=1; for i in 1 2 3; do idle $c || ok=0; sleep 1; done; [ $ok = 1 ] && { g=$c; break; }; done
    [ -n "$g" ] && [ $(nfree) -ge 3 ] && break; sleep 60
  done
  echo "$(date) $t on GPU $g"
  CUDA_VISIBLE_DEVICES=$g DIR=$DIR TASK=$t N=$n BATCH=16 METHODS=kb3cklbest OUT=$R/quality_${t}_kb3cklbest.json setsid nohup python eval_quality.py > logs_kl2eval_${DIR}_$t.log 2>&1 < /dev/null &
  sleep 90
done
wait
echo "$(date) KL2EVALDONE"
