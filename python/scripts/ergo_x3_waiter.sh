#!/bin/bash
cd ~/MorionGen
L=develop_logs/test_runs/20261105_ergo/logs/x3_waiter.log
OS=~/.local/share/opensim-api/bin/python
AP=/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/envs/motiogen-ardy/bin/python
while [ ! -f develop_logs/test_runs/20261104_x3/logs/chain_done ]; do sleep 60; done
echo "chain_done seen $(date)" >> $L
PYTHONNOUSERSITE=1 $OS python/scripts/run_ergo.py --set x3 --workers 3 >> $L 2>&1
for s in confirm top_up; do
  PYTHONNOUSERSITE=1 $AP python/scripts/analyze_ergo.py --set x3 --stage $s >> $L 2>&1
  echo "analyze $s exit=$? $(date)" >> $L
done
touch develop_logs/test_runs/20261105_ergo/logs/x3_done
