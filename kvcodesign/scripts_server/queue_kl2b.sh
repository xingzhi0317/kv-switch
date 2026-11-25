#!/bin/bash
# lower end of the LR sweep (val KL keeps improving as LR drops), then the remaining 4B->1.7B rate
cd ~/kvcodesign
DIR=17to4 ./run_kl2.sh 1e-6
DIR=4to17 ./run_kl2.sh 1e-6 3e-5
echo QUEUEDONE
