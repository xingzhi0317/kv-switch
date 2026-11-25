#!/bin/bash
# Refresh kvlingo/REPORT_kvlingo.md every 10 minutes until all four alignment runs have evaluated (or 14 h).
cd ~/kvcodesign
source ~/kvbase/.venv/bin/activate
for i in $(seq 1 84); do
  python kvlingo_report.py > /dev/null 2>> logs_kvlingo_report.err
  n=$(grep -l KVLINGODONE logs_kvl_17to4.log logs_kvl_4to17.log logs_hkv_8to4.log logs_hkv_4to8.log 2>/dev/null | wc -l)
  [ "$n" -ge 4 ] && break
  sleep 600
done
echo "$(date) REPORTLOOPDONE"
